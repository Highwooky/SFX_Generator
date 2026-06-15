"""레이어 믹싱 + 마스터링 엔진.

설계 의도(Why):
- 내부 표준 버퍼는 (num_samples, 2) float32 스테레오로 고정한다.
  pyloudnorm은 (samples, channels), pedalboard는 (channels, samples)를 쓰므로
  변환 지점을 이 모듈 안에 가둬 혼선을 막는다.
- 마스터링은 방송 규약(LUFS 라우드니스 + 리미터)을 따른다. 기존 믹스다운 툴의
  24bit/48kHz·소프트 리미터 철학을 그대로 계승한다.
"""

from __future__ import annotations

import warnings

import numpy as np
import pyloudnorm as pyln
from pedalboard import Limiter, Pedalboard

from .recipe import Master


def _equal_power_pan(mono: np.ndarray, pan: float) -> np.ndarray:
    """등파워(equal-power) 패닝으로 mono → (N, 2) 스테레오 변환.

    Why: 단순 선형 팬은 센터에서 음량이 -3dB 꺼지는 'hole' 현상이 있다.
    cos/sin 기반 등파워 팬은 좌우 어디로 움직여도 체감 음량을 일정하게 유지한다.
    """
    angle = (pan + 1.0) * 0.25 * np.pi  # pan -1→0, 0→π/4, +1→π/2
    left = np.cos(angle)
    right = np.sin(angle)
    return np.stack([mono * left, mono * right], axis=1).astype(np.float32)


def mix_layers(rendered: list[tuple[np.ndarray, float, float, float]], sr: int) -> np.ndarray:
    """변형까지 끝난 레이어들을 시작 시점에 배치해 합산한다.

    rendered: (mono_audio, start_sec, gain_db, pan) 튜플 리스트
    반환: (num_samples, 2) float32 스테레오 믹스
    """
    if not rendered:
        raise ValueError("믹스할 레이어가 없습니다.")

    # 전체 길이 = 가장 늦게 끝나는 레이어 기준
    total = 0
    placements = []
    for mono, start, gain_db, pan in rendered:
        start_n = int(round(start * sr))
        gain = 10 ** (gain_db / 20.0)
        stereo = _equal_power_pan(mono.astype(np.float32) * gain, pan)
        placements.append((stereo, start_n))
        total = max(total, start_n + len(stereo))

    buf = np.zeros((total, 2), dtype=np.float32)
    for stereo, start_n in placements:
        end_n = start_n + len(stereo)
        buf[start_n:end_n] += stereo  # 겹치는 구간은 누적 합산(레이어링)
    return buf


def _measure_loudness(stereo: np.ndarray, sr: int) -> float | None:
    """통합 라우드니스(LUFS) 측정. 너무 짧거나 무음이면 None(정규화 건너뜀)."""
    # pyloudnorm은 약 0.4초 이상이어야 측정 가능. 그 이하는 정규화 스킵.
    if len(stereo) < int(sr * 0.4):
        return None
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(stereo)
    if not np.isfinite(loudness):
        return None
    return float(loudness)


def master(stereo: np.ndarray, sr: int, cfg: Master) -> np.ndarray:
    """라우드니스 정규화 → 리미터 순으로 마스터링.

    Why this order: 먼저 목표 LUFS로 게인을 맞춘 뒤, 그 결과의 순간 피크를
    리미터로 눌러 0dBFS 오버를 막는다(클리핑 방지). 방송 납품 안전 마진.
    """
    out = stereo.copy()

    if cfg.loudness_lufs is not None:
        measured = _measure_loudness(out, sr)
        if measured is not None:
            # 정규화로 일부 피크가 0dBFS를 넘을 수 있으나 바로 아래 리미터가 처리한다.
            # 따라서 pyloudnorm의 클리핑 경고는 무의미하므로 의도적으로 억제한다.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out = pyln.normalize.loudness(out, measured, cfg.loudness_lufs).astype(np.float32)

    if cfg.limiter:
        # pedalboard는 (channels, samples) → 전치 후 처리, 다시 환원
        board = Pedalboard([Limiter(threshold_db=-1.0, release_ms=100.0)])
        processed = board(np.ascontiguousarray(out.T), sr)
        out = np.ascontiguousarray(np.asarray(processed, dtype=np.float32).T)

    # 최종 안전장치: 리미터 후에도 미세하게 넘칠 수 있으니 하드 클립으로 봉인
    np.clip(out, -1.0, 1.0, out=out)
    return out
