"""렌더 오케스트레이션: 레시피 → 최종 WAV.

설계 의도(Why):
- 이 모듈은 '지휘자'다. 직접 DSP를 하지 않고 synth/dsp/mixer를 순서대로 호출만 한다.
- 라이브러리 음원 해석은 외부 resolver(콜러블)에 위임한다. 그래서 검색 구현
  (태그/임베딩)을 교체해도 렌더 로직은 바뀌지 않는다(의존성 역전).
- 합성은 항상 동작하므로 resolver 없이도 synth-only 레시피는 완전히 렌더된다.
"""

from __future__ import annotations

import re
from math import gcd
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from . import dsp, mixer, synth
from .recipe import Layer, Recipe

# query(검색어), pick("best"|"random") → 음원 파일 경로(없으면 None)
Resolver = Callable[[str, str], Optional[Path]]

# 비트 심도 → soundfile subtype. 32비트는 프로덕션 표준인 float로 처리.
_SUBTYPE = {16: "PCM_16", 24: "PCM_24", 32: "FLOAT"}


def _load_sample(path: Path, target_sr: int) -> np.ndarray:
    """음원을 mono float32로 로드 후 목표 샘플레이트로 리샘플링."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)  # 다채널은 평균으로 모노화
    if sr != target_sr:
        # 정수비 리샘플링(resample_poly)이 FFT 방식보다 에일리어싱이 적고 빠르다.
        g = gcd(int(sr), int(target_sr))
        mono = resample_poly(mono, target_sr // g, sr // g).astype(np.float32)
    return mono.astype(np.float32)


def _render_layer(layer: Layer, sr: int, rng: np.random.Generator, resolver: Optional[Resolver]) -> np.ndarray:
    """레이어 1개를 출처(합성/라이브러리)에서 만들고 변형 체인까지 적용한 mono 반환."""
    if layer.synth is not None:
        base = synth.synthesize(layer.synth, sr, rng)
    else:
        assert layer.source is not None  # 스키마 검증으로 보장됨
        if resolver is None:
            raise RuntimeError(
                f"라이브러리 음원('{layer.source.query}')을 쓰려면 resolver가 필요합니다."
            )
        path = resolver(layer.source.query, layer.source.pick)
        if path is None:
            raise FileNotFoundError(f"라이브러리에서 '{layer.source.query}'를 찾지 못했습니다.")
        base = _load_sample(Path(path), sr)

    return dsp.apply_chain(base, sr, layer.transforms)


def _safe_filename(prefix: str, name: str) -> str:
    """파일시스템에 안전한 이름 생성. prefix는 기존 규약([SFX] 등) 유지."""
    clean = re.sub(r'[\\/:*?"<>|]+', "_", name).strip()
    prefix = prefix.strip()
    return f"{prefix} {clean}.wav" if prefix else f"{clean}.wav"


def _fit_length(stereo: np.ndarray, sr: int, length_sec: float) -> np.ndarray:
    """최종 스테레오를 정확히 length_sec 길이로 맞춘다(짧으면 무음 패딩, 길면 잘라냄).

    Why: 사용자가 '2초' 같은 정확한 길이를 지정할 수 있게. 끝에 짧은 페이드를
    넣어 잘릴 때 클릭 노이즈를 방지한다.
    """
    target = max(1, int(length_sec * sr))
    n = stereo.shape[0]
    if n > target:
        stereo = stereo[:target].copy()
        # 끝 5ms 페이드아웃으로 클릭 방지
        f = min(int(sr * 0.005), target)
        if f > 0:
            stereo[-f:] *= np.linspace(1.0, 0.0, f)[:, None]
    elif n < target:
        pad = np.zeros((target - n, stereo.shape[1]), dtype=stereo.dtype)
        stereo = np.vstack([stereo, pad])
    return stereo


def render_recipe(
    recipe: Recipe, out_dir: Path, resolver: Optional[Resolver] = None,
    length_sec: Optional[float] = None,
) -> Path:
    """레시피 1건을 렌더해 WAV 파일로 저장하고 경로를 반환.

    length_sec를 주면 최종 길이를 정확히 그 초로 맞춘다(미지정 시 자연 길이).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sr = recipe.master.format.rate
    rng = np.random.default_rng(recipe.seed)

    rendered: list[tuple[np.ndarray, float, float, float]] = []
    for i, layer in enumerate(recipe.layers):
        try:
            mono = _render_layer(layer, sr, rng, resolver)
        except Exception as e:  # noqa: BLE001 - 레이어 인덱스를 붙여 디버깅 용이하게
            raise RuntimeError(f"레이어 {i} 렌더 실패: {e}") from e
        rendered.append((mono, layer.start, layer.gain_db, layer.pan))

    stereo = mixer.mix_layers(rendered, sr)
    stereo = mixer.master(stereo, sr, recipe.master)
    if length_sec is not None:
        stereo = _fit_length(stereo, sr, length_sec)

    out_path = out_dir / _safe_filename(recipe.master.prefix, recipe.name)
    sf.write(str(out_path), stereo, sr, subtype=_SUBTYPE[recipe.master.format.bit])
    return out_path


def render_variations(
    recipe: Recipe, count: int, out_dir: Path, resolver: Optional[Resolver] = None,
    length_sec: Optional[float] = None,
) -> list[Path]:
    """동일 레시피로 seed만 바꿔 변주를 일괄 생성(현장에서 골라 쓰기용)."""
    if count < 1:
        raise ValueError("count는 1 이상이어야 합니다.")
    paths: list[Path] = []
    for i in range(count):
        # 원본 seed에서 파생 → 재현 가능하면서도 서로 다른 변주
        variant = recipe.model_copy(deep=True)
        variant.seed = recipe.seed + i
        variant.name = f"{recipe.name}_v{i + 1}"
        paths.append(render_recipe(variant, out_dir, resolver, length_sec=length_sec))
    return paths
