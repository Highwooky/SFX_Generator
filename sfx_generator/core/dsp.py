"""단일 레이어 변형 엔진.

설계 의도(Why):
- 모든 변형은 mono float32 1D 배열을 받아 mono float32 1D 배열을 반환한다.
  (스테레오/팬은 mixer 단계에서 일괄 처리 → 변형 로직을 단순하게 유지)
- 피치/필터/리버브/디스토션 등은 pedalboard로 처리한다. 고품질이면서
  Apple Silicon 네이티브이고 완전 오프라인이라 에어갭 환경에 적합하다.
- 알 수 없는 op는 조용히 무시하지 않고 명시적으로 예외를 던진다(오류 은폐 금지).
"""

from __future__ import annotations

import numpy as np
from pedalboard import (
    Bitcrush,
    Chorus,
    Delay,
    Distortion,
    Gain,
    HighpassFilter,
    LowpassFilter,
    PeakFilter,
    Pedalboard,
    Reverb,
    time_stretch,
)
from scipy.signal import istft, stft

from .recipe import TransformUnion

# 리버브 종류 → (room_size, damping) 매핑. 방송 효과음에서 자주 쓰는 3종.
_REVERB_PRESETS = {
    "room": (0.25, 0.6),
    "hall": (0.7, 0.4),
    "plate": (0.5, 0.2),
}


def _run_board(board: Pedalboard, audio: np.ndarray, sr: int) -> np.ndarray:
    """mono 1D 배열을 pedalboard로 처리. (1, N) 형태로 넘기고 다시 1D로 환원."""
    buf = np.ascontiguousarray(audio.reshape(1, -1).astype(np.float32))
    out = board(buf, sr)
    return np.asarray(out, dtype=np.float32).reshape(-1)


def _fade(audio: np.ndarray, sr: int, fade_in: float, fade_out: float) -> np.ndarray:
    """선형 페이드 인/아웃. 클릭 노이즈 방지 및 자연스러운 시작/끝 처리."""
    out = audio.copy()
    n = len(out)
    fi = min(int(sr * fade_in), n)
    fo = min(int(sr * fade_out), n)
    if fi > 0:
        out[:fi] *= np.linspace(0.0, 1.0, fi, dtype=np.float32)
    if fo > 0:
        out[n - fo :] *= np.linspace(1.0, 0.0, fo, dtype=np.float32)
    return out


def _normalize_peak(audio: np.ndarray, peak_db: float) -> np.ndarray:
    """피크 기준 정규화. 무음 입력은 그대로 반환(0 나눗셈 방지)."""
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak < 1e-9:
        return audio
    target = 10 ** (peak_db / 20.0)
    return (audio * (target / peak)).astype(np.float32)


def _map_stretch_rate(rate: float) -> float:
    """레시피 rate(>1=길어짐)를 pedalboard time_stretch 인자로 변환.

    Why: pedalboard의 stretch_factor는 '속도 배율'이라 2.0이면 길이가 절반(빨라짐)이
    된다. 레시피 rate는 '길이 배율'(>1=길어짐)이므로 역수를 취해야 의미가 맞는다.
    """
    return 1.0 / rate


def _granular(audio: np.ndarray, sr: int, tf) -> np.ndarray:
    """그래뉼러 합성: 입력을 잘게 쪼개(그레인) 윈도우·피치변형 후 재배치(overlap-add).

    Why: 짧은 샘플 하나로 길이·질감이 전혀 다른 텍스처/스웰을 무한히 파생할 수 있다.
    난수는 길이 기반으로 시드해 결과를 결정적으로(재현 가능) 만든다.
    """
    n = len(audio)
    if n < 16:
        return audio
    rng = np.random.default_rng(1234 + n)
    grain = max(16, int(sr * tf.grain_ms / 1000.0))
    win = np.hanning(grain).astype(np.float32)
    out_len = max(grain, int(n * tf.stretch))
    out = np.zeros(out_len + grain, dtype=np.float32)
    hop = max(1, int(grain / max(tf.density, 0.1)))
    spray = int(sr * tf.spray_ms / 1000.0)

    pos = 0
    while pos < out_len:
        # 출력 위치를 입력 위치로 역매핑(stretch) + 분산(spray)
        src = int(pos / max(tf.stretch, 1e-6)) + (rng.integers(-spray, spray + 1) if spray > 0 else 0)
        src = int(np.clip(src, 0, max(n - grain, 0)))
        seg = audio[src:src + grain]
        if len(seg) < grain:
            seg = np.pad(seg, (0, grain - len(seg)))
        # 그레인별 피치 흔들림(리샘플)
        if tf.pitch_jitter > 0:
            st = rng.uniform(-tf.pitch_jitter, tf.pitch_jitter)
            factor = 2 ** (st / 12.0)
            idx = np.clip(np.arange(grain) * factor, 0, grain - 1)
            seg = np.interp(idx, np.arange(grain), seg).astype(np.float32)
        out[pos:pos + grain] += seg * win
        pos += hop

    out = out[:out_len]
    peak = float(np.max(np.abs(out))) or 1.0
    return (out / peak * 0.9).astype(np.float32)


def _spectral(audio: np.ndarray, sr: int, tf) -> np.ndarray:
    """스펙트럴 처리(STFT 기반): freeze/stretch/blur로 드론·패드·번짐 텍스처 생성."""
    n = len(audio)
    nperseg = 1024
    if n < nperseg:
        audio = np.pad(audio, (0, nperseg - n))
    rng = np.random.default_rng(99 + len(audio))
    f, t, Z = stft(audio.astype(np.float32), fs=sr, nperseg=nperseg, noverlap=nperseg * 3 // 4)
    mag = np.abs(Z)

    if tf.mode == "freeze":
        # 중앙 프레임의 스펙트럼을 길게 유지 → 정적인 드론
        frames = max(1, int(mag.shape[1] * tf.amount))
        col = mag[:, mag.shape[1] // 2:mag.shape[1] // 2 + 1]
        mag2 = np.repeat(col, frames, axis=1)
    elif tf.mode == "stretch":
        # 시간축으로 스펙트럼 프레임을 보간(늘림)
        frames = max(1, int(mag.shape[1] * tf.amount))
        xs = np.linspace(0, mag.shape[1] - 1, frames)
        mag2 = np.stack([np.interp(xs, np.arange(mag.shape[1]), mag[b]) for b in range(mag.shape[0])])
    else:  # blur: 시간축 이동평균으로 번짐
        k = max(1, int(tf.amount))
        kernel = np.ones(k, dtype=np.float32) / k
        mag2 = np.stack([np.convolve(mag[b], kernel, mode="same") for b in range(mag.shape[0])])

    # 무작위 위상으로 재합성(텍스처/패드 성격) → ISTFT
    phase = rng.uniform(-np.pi, np.pi, size=mag2.shape).astype(np.float32)
    Z2 = mag2 * np.exp(1j * phase)
    _, y = istft(Z2, fs=sr, nperseg=nperseg, noverlap=nperseg * 3 // 4)
    y = np.asarray(y, dtype=np.float32)
    peak = float(np.max(np.abs(y))) or 1.0
    return (y / peak * 0.85).astype(np.float32)


def _envelope(audio: np.ndarray, tf) -> np.ndarray:
    """엔벨로프 폴로잉: (시간0~1, 게인) 브레이크포인트를 신호 길이에 보간해 곱한다."""
    pts = sorted(tf.points, key=lambda p: p[0])
    ts = np.array([p[0] for p in pts], dtype=np.float32)
    gs = np.array([p[1] for p in pts], dtype=np.float32)
    x = np.linspace(0.0, 1.0, len(audio), dtype=np.float32)
    env = np.interp(x, ts, gs).astype(np.float32)
    return (audio * env).astype(np.float32)


def apply_transform(audio: np.ndarray, sr: int, tf: TransformUnion) -> np.ndarray:
    """변형 1개를 적용. op 별 분기를 명확히 나열(가독성 우선)."""
    op = tf.op

    if op == "pitch":
        # 피치만 이동(길이 보존). time_stretch에 stretch_factor=1 + 반음 지정.
        buf = np.ascontiguousarray(audio.reshape(1, -1).astype(np.float32))
        out = time_stretch(buf, sr, 1.0, float(tf.semitones))
        return np.asarray(out, dtype=np.float32).reshape(-1)

    if op == "stretch":
        buf = np.ascontiguousarray(audio.reshape(1, -1).astype(np.float32))
        out = time_stretch(buf, sr, _map_stretch_rate(tf.rate), 0.0)
        return np.asarray(out, dtype=np.float32).reshape(-1)

    if op == "reverse":
        return audio[::-1].copy()

    if op == "gain":
        return _run_board(Pedalboard([Gain(gain_db=tf.db)]), audio, sr)

    if op == "fade":
        return _fade(audio, sr, tf.in_, tf.out)

    if op == "filter":
        plugin = (
            HighpassFilter(cutoff_frequency_hz=tf.cutoff_hz)
            if tf.kind == "highpass"
            else LowpassFilter(cutoff_frequency_hz=tf.cutoff_hz)
        )
        return _run_board(Pedalboard([plugin]), audio, sr)

    if op == "eq":
        return _run_board(
            Pedalboard([PeakFilter(cutoff_frequency_hz=tf.freq_hz, gain_db=tf.gain_db, q=tf.q)]),
            audio,
            sr,
        )

    if op == "reverb":
        room_size, damping = _REVERB_PRESETS[tf.kind]
        rev = Reverb(room_size=room_size, damping=damping, wet_level=tf.wet, dry_level=1.0 - tf.wet)
        return _run_board(Pedalboard([rev]), audio, sr)

    if op == "distortion":
        return _run_board(Pedalboard([Distortion(drive_db=tf.drive_db)]), audio, sr)

    if op == "chorus":
        return _run_board(
            Pedalboard([Chorus(rate_hz=tf.rate_hz, depth=tf.depth, mix=tf.mix)]), audio, sr
        )

    if op == "delay":
        return _run_board(
            Pedalboard([Delay(delay_seconds=tf.seconds, feedback=tf.feedback, mix=tf.mix)]),
            audio,
            sr,
        )

    if op == "bitcrush":
        return _run_board(Pedalboard([Bitcrush(bit_depth=tf.bit_depth)]), audio, sr)

    if op == "normalize":
        return _normalize_peak(audio, tf.peak_db)

    if op == "granular":
        return _granular(audio, sr, tf)

    if op == "spectral":
        return _spectral(audio, sr, tf)

    if op == "envelope":
        return _envelope(audio, tf)

    raise ValueError(f"지원하지 않는 변형 op: {op}")


def apply_chain(audio: np.ndarray, sr: int, transforms: list[TransformUnion]) -> np.ndarray:
    """변형 체인을 순서대로 적용. 각 단계 실패 시 어떤 op에서 났는지 알려준다."""
    out = audio
    for i, tf in enumerate(transforms):
        try:
            out = apply_transform(out, sr, tf)
        except Exception as e:  # noqa: BLE001 - 어떤 단계에서 실패했는지 맥락 보강 후 재전파
            raise RuntimeError(f"변형 체인 {i}번째({tf.op}) 적용 실패: {e}") from e
    return out
