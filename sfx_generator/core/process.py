"""원본 오디오 가공: 입력 음원을 '실제 소스'로 받아 변형·보강 레이어를 입힌다.

설계 의도(Why):
- 레퍼런스를 합성으로 흉내내면 전혀 다른 소리가 났다. 대신 원본 오디오를 그대로
  소스로 쓰고 DSP만 입히면, 사운드는 진짜이면서 다양한 변주/효과를 얻을 수 있다.
- 소스는 센티넬 쿼리(SOURCE_QUERY)로 참조하고, 렌더 시 resolver가 실제 파일을 반환한다.
"""

from __future__ import annotations

import copy

from .recipe import Recipe

# 렌더 resolver가 '선택한 원본 파일'로 해석하는 센티넬 쿼리
SOURCE_QUERY = "__srcfile__"

_FMT = {"loudness_lufs": -16, "limiter": True, "format": {"bit": 24, "rate": 48000}, "prefix": "[SFX]"}

# 스타일 → 원본에 적용할 변형 체인(tf) + 보강 레이어(extra)
# tf는 원본 소스 레이어에 입히고, extra는 원본과 함께 섞이는 합성 레이어.
_STYLES: dict[str, dict] = {
    "원본 정리": {"tf": []},
    "공간감(홀)": {"tf": [{"op": "reverb", "kind": "hall", "wet": 0.3}]},
    "룸 잔향": {"tf": [{"op": "reverb", "kind": "room", "wet": 0.25}]},
    "어둡게": {"tf": [{"op": "filter", "kind": "lowpass", "cutoff_hz": 2500}, {"op": "pitch", "semitones": -2}]},
    "밝게": {"tf": [{"op": "eq", "freq_hz": 6000, "gain_db": 4, "q": 0.9}]},
    "리버스": {"tf": [{"op": "reverse"}, {"op": "reverb", "kind": "room", "wet": 0.2}]},
    "더블/슬랩": {"tf": [{"op": "delay", "seconds": 0.09, "feedback": 0.2, "mix": 0.3}]},
    "와이드(코러스)": {"tf": [{"op": "chorus", "rate_hz": 0.8, "depth": 0.3, "mix": 0.4}]},
    "텍스처(그래뉼러)": {"tf": [{"op": "granular", "grain_ms": 80, "density": 2.5,
                            "pitch_jitter": 3, "stretch": 1.5, "spray_ms": 30},
                           {"op": "reverb", "kind": "hall", "wet": 0.2}]},
    "묵직하게+서브": {"tf": [], "extra": [
        {"synth": {"kind": "sub_impact", "freq": 55, "decay": 0.6}, "gain_db": -7}]},
    "반짝임 추가": {"tf": [], "extra": [
        {"synth": {"kind": "tone", "freq": 2400, "decay": 0.5},
         "transforms": [{"op": "delay", "seconds": 0.1, "feedback": 0.3, "mix": 0.4}],
         "gain_db": -12, "start": 0.02}]},
    "긴장 라이저+": {"tf": [], "extra": [
        {"synth": {"kind": "riser", "freq": 300, "decay": 1.2}, "gain_db": -6}]},
}

# '자동 변주'에서 순환 사용할 스타일(서로 충분히 다른 것들)
_VARIATION_CYCLE = ["원본 정리", "공간감(홀)", "어둡게", "밝게", "더블/슬랩", "와이드(코러스)", "룸 잔향", "텍스처(그래뉼러)"]


def list_styles() -> list[str]:
    """선택 가능한 가공 스타일 이름."""
    return list(_STYLES)


def build_recipe(style: str, name: str = "원본_가공", seed: int = 0) -> Recipe:
    """선택한 스타일로 원본 소스 가공 레시피를 만든다."""
    spec = _STYLES.get(style, {"tf": []})
    src_layer: dict = {"source": {"query": SOURCE_QUERY, "pick": "best"},
                       "transforms": copy.deepcopy(spec.get("tf", []))}
    layers = [src_layer] + copy.deepcopy(spec.get("extra", []))
    return Recipe.model_validate({"name": name, "seed": seed, "layers": layers, "master": _FMT})


def variation_recipes(n: int, base_name: str = "원본_변주") -> list[tuple[str, Recipe]]:
    """원본을 서로 다른 스타일로 가공한 변주 N개. (스타일명, Recipe) 목록."""
    out: list[tuple[str, Recipe]] = []
    for i in range(max(1, n)):
        style = _VARIATION_CYCLE[i % len(_VARIATION_CYCLE)]
        out.append((style, build_recipe(style, name=f"{base_name}_{style}", seed=i)))
    return out


def make_source_resolver(src_path, base_resolver=None):
    """SOURCE_QUERY를 선택한 원본 파일로 해석하는 resolver. 그 외 쿼리는 base에 위임."""
    def _res(query, pick, _src=src_path, _base=base_resolver):
        if query == SOURCE_QUERY:
            return _src
        return _base(query, pick) if _base else None
    return _res
