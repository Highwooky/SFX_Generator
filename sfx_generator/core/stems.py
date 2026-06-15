"""효과 스템 → 개별 효과음 클립 자동 분할.

설계 의도(Why):
- 사용자가 실제 방송에서 쓴 'SFX 스템'(효과음만 있는 트랙)을 넣으면, 무음 구간을 기준으로
  개별 효과음을 잘라 라이브러리에 넣는다. = '내가 쓰던 소리'를 그대로 재료화(검색·합치기·노브).
- 믹스된 영상/마스터(대사·음악 섞임)는 분리가 사실상 불가하므로, '효과 스템'을 전제로 한다.
- AI 학습 없이, 신호 처리(에너지 게이팅)만으로 안정적으로 동작한다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

_TARGET_SR = 48000


def _safe(name: str) -> str:
    for ch in '/\\:*?"<>|':
        name = name.replace(ch, "_")
    return name.strip() or "clip"


def split_stem(path: Path, out_dir: Path, *, threshold_db: float = -40.0,
               min_silence_s: float = 0.15, min_clip_s: float = 0.08,
               pad_s: float = 0.02, max_clips: int = 300) -> list[Path]:
    """스템을 무음 기준으로 잘라 개별 클립(24bit/48kHz WAV)으로 저장. 경로 목록 반환."""
    path = Path(path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if data.size == 0:
        return []
    mono = data.mean(axis=1)
    sr = int(sr)

    # 프레임 RMS로 활성/무음 판정
    hop, win = 256, 1024
    n_fr = max(1, (len(mono) - win) // hop)
    rms = np.array([np.sqrt(np.mean(mono[i * hop:i * hop + win] ** 2) + 1e-12) for i in range(n_fr)])
    peak = float(rms.max()) or 1e-9
    thr = peak * (10 ** (threshold_db / 20.0))
    active = rms > thr

    # 활성 구간 검출 + 짧은 무음(min_silence)으로 끊긴 곳은 한 클립으로 병합
    gap_frames = int(min_silence_s * sr / hop)
    regions: list[tuple[int, int]] = []
    i = 0
    while i < len(active):
        if active[i]:
            start = i
            silence = 0
            j = i
            while j < len(active):
                if active[j]:
                    silence = 0
                else:
                    silence += 1
                    if silence > gap_frames:
                        break
                j += 1
            end = j - silence
            regions.append((start, end))
            i = j
        else:
            i += 1

    pad = int(pad_s * sr)
    min_len = int(min_clip_s * sr)
    g = np.gcd(sr, _TARGET_SR)
    up, down = _TARGET_SR // g, sr // g
    stem_name = _safe(path.stem)
    out_paths: list[Path] = []
    idx = 0
    for (fs, fe) in regions:
        s = max(0, fs * hop - pad)
        e = min(len(mono), fe * hop + win + pad)
        if e - s < min_len:
            continue
        clip = data[s:e]  # 원본 채널 유지
        # 가장자리 5ms 페이드(클릭 방지)
        fade = min(int(0.005 * sr), clip.shape[0] // 4) or 1
        env = np.ones(clip.shape[0], dtype=np.float32)
        env[:fade] = np.linspace(0, 1, fade)
        env[-fade:] = np.linspace(1, 0, fade)
        clip = clip * env[:, None]
        if sr != _TARGET_SR:  # 48kHz로 리샘플
            clip = resample_poly(clip, up, down, axis=0).astype(np.float32)
        idx += 1
        out = out_dir / f"{stem_name}_{idx:03d}.wav"
        sf.write(str(out), clip, _TARGET_SR, subtype="PCM_24")
        out_paths.append(out)
        if idx >= max_clips:
            break
    return out_paths
