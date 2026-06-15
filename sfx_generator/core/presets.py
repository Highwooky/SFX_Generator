"""프리셋 갤러리: 검증된 레시피 모음.

설계 의도(Why):
- 프롬프트 작성이 막힐 때 '골라 쓰는' 출발점을 제공한다(프롬프트 한계 보완).
- 모두 합성 전용이라 라이브러리가 없어도 항상 렌더된다. 새 합성/변형 엔진
  (modal·fm·pluck·wind·rain·fire·granular·spectral)의 사용 예시도 겸한다.
- 사용자는 이 레시피를 그대로 쓰거나 GUI의 JSON 편집창에서 미세조정한다.
"""

from __future__ import annotations

import copy
import random
from typing import Optional

from .recipe import Recipe

_FMT = {"loudness_lufs": -16, "limiter": True, "format": {"bit": 24, "rate": 48000}, "prefix": "[SFX]"}

# 이름 → (설명, 레이어 리스트)
_PRESETS: dict[str, tuple[str, list]] = {
    "공포_임팩트": ("긴장 라이저 뒤 묵직한 저음 쿵", [
        {"synth": {"kind": "riser", "freq": 200, "decay": 2.0},
         "transforms": [{"op": "reverb", "kind": "hall", "wet": 0.35}], "gain_db": -3, "pan": -0.2},
        {"synth": {"kind": "sub_impact", "freq": 55, "decay": 1.2},
         "transforms": [{"op": "distortion", "drive_db": 6}], "start": 1.8, "gain_db": -2},
    ]),
    "유리_깨짐": ("밝은 크랙 + 여러 파편 공진(다층)", [
        {"synth": {"kind": "noise", "decay": 0.12},
         "transforms": [{"op": "filter", "kind": "highpass", "cutoff_hz": 4000}, {"op": "fade", "out": 0.1}],
         "gain_db": -4},
        {"synth": {"kind": "modal", "material": "glass", "freq": 2600, "decay": 0.28},
         "transforms": [{"op": "filter", "kind": "highpass", "cutoff_hz": 1500}], "start": 0.01, "gain_db": -3, "pan": -0.2},
        {"synth": {"kind": "modal", "material": "glass", "freq": 3900, "decay": 0.22},
         "start": 0.04, "gain_db": -5, "pan": 0.25},
        {"synth": {"kind": "modal", "material": "glass", "freq": 5400, "decay": 0.16},
         "start": 0.07, "gain_db": -7, "pan": 0.0},
    ]),
    "금속_타격": ("어택 트랜지언트 + 금속 모달 링 + 홀 잔향", [
        {"synth": {"kind": "noise", "decay": 0.05},
         "transforms": [{"op": "filter", "kind": "highpass", "cutoff_hz": 2000}, {"op": "fade", "out": 0.04}],
         "gain_db": -8},
        {"synth": {"kind": "modal", "material": "metal", "freq": 320, "decay": 1.8},
         "transforms": [{"op": "reverb", "kind": "hall", "wet": 0.3}], "start": 0.005},
    ]),
    "전자_확인음": ("밝은 FM 확인음(UI)", [
        {"synth": {"kind": "fm", "freq": 880, "decay": 0.35, "ratio": 2.0, "index": 4},
         "transforms": [{"op": "eq", "freq_hz": 4000, "gain_db": 3, "q": 1.0}]},
    ]),
    "레이저_발사": ("하강 스윕(리버스 라이저) + FM 톤", [
        {"synth": {"kind": "riser", "freq": 500, "decay": 0.45},
         "transforms": [{"op": "reverse"}, {"op": "filter", "kind": "highpass", "cutoff_hz": 400},
                        {"op": "fade", "out": 0.15}], "gain_db": -3},
        {"synth": {"kind": "fm", "freq": 1100, "decay": 0.4, "ratio": 2.0, "index": 7},
         "transforms": [{"op": "pitch", "semitones": -8}, {"op": "delay", "seconds": 0.1, "feedback": 0.25, "mix": 0.25}],
         "gain_db": -4},
    ]),
    "현_뜯기": ("Karplus-Strong 현 + 룸 잔향", [
        {"synth": {"kind": "pluck", "freq": 330, "decay": 1.4},
         "transforms": [{"op": "reverb", "kind": "room", "wet": 0.25}]},
    ]),
    "바람_텍스처": ("거센 바람 베드", [
        {"synth": {"kind": "wind", "freq": 600, "decay": 4.0},
         "transforms": [{"op": "reverb", "kind": "hall", "wet": 0.25}, {"op": "fade", "in": 0.5, "out": 1.0}]},
    ]),
    "빗소리_배경": ("빗소리 앰비언스", [
        {"synth": {"kind": "rain", "decay": 5.0}, "transforms": [{"op": "fade", "in": 0.4, "out": 1.0}]},
    ]),
    "모닥불": ("불 럼블 + 크래클", [
        {"synth": {"kind": "fire", "decay": 4.0}, "transforms": [{"op": "fade", "in": 0.3, "out": 0.8}]},
    ]),
    "스펙트럴_드론": ("톤을 스펙트럴 프리즈해 만든 긴장 드론", [
        {"synth": {"kind": "tone", "freq": 196, "decay": 1.0},
         "transforms": [{"op": "spectral", "mode": "freeze", "amount": 4.0},
                        {"op": "filter", "kind": "lowpass", "cutoff_hz": 3000},
                        {"op": "fade", "in": 0.5, "out": 1.0}]},
    ]),
    "그래뉼러_구름": ("노이즈를 그래뉼러로 흩뿌린 텍스처", [
        {"synth": {"kind": "noise", "decay": 0.4},
         "transforms": [{"op": "granular", "grain_ms": 60, "density": 3, "pitch_jitter": 5, "stretch": 8.0, "spray_ms": 40},
                        {"op": "reverb", "kind": "hall", "wet": 0.3}]},
    ]),
    # ── 예능 자막 큐 ──
    "두둥_반전": ("낮은 두 번의 임팩트 + 홀 잔향(반전/등장)", [
        {"synth": {"kind": "sub_impact", "freq": 60, "decay": 1.2},
         "transforms": [{"op": "reverb", "kind": "hall", "wet": 0.4}], "gain_db": -2},
        {"synth": {"kind": "sub_impact", "freq": 85, "decay": 1.6},
         "transforms": [{"op": "reverb", "kind": "hall", "wet": 0.45}], "start": 0.35, "gain_db": -1},
    ]),
    "짜잔_등장": ("FM 팡파레 + 반짝이는 꼬리(짜잔)", [
        {"synth": {"kind": "fm", "freq": 523, "decay": 1.0, "ratio": 2.0, "index": 6},
         "transforms": [{"op": "eq", "freq_hz": 3000, "gain_db": 4, "q": 1.0}, {"op": "reverb", "kind": "room", "wet": 0.2}], "gain_db": -3},
        {"synth": {"kind": "tone", "freq": 1568, "decay": 1.2},
         "transforms": [{"op": "delay", "seconds": 0.12, "feedback": 0.3, "mix": 0.3}], "start": 0.05, "gain_db": -7},
    ]),
    "두구두구_정답공개": ("드럼롤 긴장 + 끝에 딩동댕", [
        {"synth": {"kind": "noise", "decay": 1.6},
         "transforms": [{"op": "filter", "kind": "lowpass", "cutoff_hz": 2500}, {"op": "fade", "in": 0.15, "out": 0.1}], "gain_db": -6},
        {"synth": {"kind": "tone", "freq": 1046, "decay": 0.9},
         "transforms": [{"op": "eq", "freq_hz": 2000, "gain_db": 3, "q": 1.0}], "start": 1.5, "gain_db": -1},
    ]),
    "두근두근_심장": ("저음 더블 비트 두 번(긴장/설렘)", [
        {"synth": {"kind": "sub_impact", "freq": 55, "decay": 0.3},
         "transforms": [{"op": "filter", "kind": "lowpass", "cutoff_hz": 220}], "start": 0.0, "gain_db": -3},
        {"synth": {"kind": "sub_impact", "freq": 48, "decay": 0.35},
         "transforms": [{"op": "filter", "kind": "lowpass", "cutoff_hz": 220}], "start": 0.2, "gain_db": -3},
        {"synth": {"kind": "sub_impact", "freq": 55, "decay": 0.3},
         "transforms": [{"op": "filter", "kind": "lowpass", "cutoff_hz": 220}], "start": 0.85, "gain_db": -3},
        {"synth": {"kind": "sub_impact", "freq": 48, "decay": 0.35},
         "transforms": [{"op": "filter", "kind": "lowpass", "cutoff_hz": 220}], "start": 1.05, "gain_db": -3},
    ]),
    "와장창_깨짐": ("크랙 + 유리/금속 파편(사고/충돌)", [
        {"synth": {"kind": "noise", "decay": 0.18},
         "transforms": [{"op": "filter", "kind": "highpass", "cutoff_hz": 2500}, {"op": "fade", "out": 0.15}], "gain_db": -2},
        {"synth": {"kind": "modal", "material": "glass", "freq": 2000, "decay": 0.5}, "start": 0.01, "gain_db": -3, "pan": -0.2},
        {"synth": {"kind": "modal", "material": "glass", "freq": 3400, "decay": 0.35}, "start": 0.05, "gain_db": -5, "pan": 0.25},
        {"synth": {"kind": "modal", "material": "metal", "freq": 700, "decay": 0.6}, "start": 0.08, "gain_db": -8},
    ]),
    "정적_썰렁": ("얇은 고음 귀뚜라미 + 희미한 찬바람(어색한 정적)", [
        {"synth": {"kind": "tone", "freq": 2600, "decay": 2.0},
         "transforms": [{"op": "chorus", "rate_hz": 6, "depth": 0.4, "mix": 0.5}, {"op": "fade", "in": 0.2, "out": 0.6}], "gain_db": -12},
        {"synth": {"kind": "wind", "freq": 700, "decay": 2.0},
         "transforms": [{"op": "filter", "kind": "highpass", "cutoff_hz": 500}], "gain_db": -18},
    ]),
}


def list_presets() -> list[tuple[str, str]]:
    """(이름, 설명) 목록."""
    return [(name, desc) for name, (desc, _) in _PRESETS.items()]


def get_recipe(name: str) -> Recipe:
    """프리셋 이름으로 검증된 Recipe를 만든다."""
    if name not in _PRESETS:
        raise KeyError(f"알 수 없는 프리셋: {name} (사용 가능: {', '.join(_PRESETS)})")
    _, layers = _PRESETS[name]
    return Recipe.model_validate({"name": name, "seed": 0, "layers": layers, "master": _FMT})


# ── 내부 알고리즘: 해석기 타입 → 다층 템플릿 매핑 ────────────────────────────
# UI에 프리셋 목록을 노출하지 않고, 해당 의성어/타입이 감지되면 이 다층 레시피를
# '내부적으로' 사용해 단일 합성보다 훨씬 그럴듯한 결과를 낸다.
_TYPE_TEMPLATE: dict[str, str] = {
    "dramatic": "두둥_반전",
    "fanfare": "짜잔_등장",
    "heartbeat": "두근두근_심장",
    "crash": "와장창_깨짐",
    "awkward": "정적_썰렁",
    "glass": "유리_깨짐",
    "metal": "금속_타격",
}


def template_layers(type_name: str) -> Optional[list[dict]]:
    """타입 이름에 대응하는 내부 템플릿 레이어(깊은 복사)를 반환, 없으면 None."""
    preset = _TYPE_TEMPLATE.get(type_name)
    if preset is None:
        return None
    return copy.deepcopy(_PRESETS[preset][1])


def random_template() -> tuple[str, Recipe]:
    """'영감(랜덤)'용: 내부 템플릿 중 하나를 무작위로 골라 (이름, 레시피) 반환."""
    name = random.choice(list(_PRESETS.keys()))
    return name, get_recipe(name)


# ── 컨셉 추론: 카테고리성 프롬프트 → 대표 자막 효과음 '팩' ────────────────────
def detect_concept(prompt: str) -> Optional[str]:
    """프롬프트가 특정 사운드가 아니라 '카테고리/모음'을 원하면 컨셉명을 반환.

    예: '예능에서 쓸만한 자막 효과음', '교양 프로그램 인서트 모음' → 팩 생성.
    구체적 단일 요청('밝은 자막 효과')은 None(일반 해석으로).
    """
    t = prompt
    genre = None
    if any(k in t for k in ["예능", "오락", "버라이어티"]):
        genre = "예능"
    elif any(k in t for k in ["교양", "다큐", "시사", "뉴스"]):
        genre = "교양"
    collection = any(k in t for k in ["모음", "세트", "팩", "여러", "종류", "쓸만", "쓸 만", "추천", "이런 식", "같은"])
    if genre:
        return genre
    if collection and ("자막" in t or "효과음" in t or "코드음" in t):
        return "예능"
    return None


def _mk(name: str, layers: list, seed: int = 0) -> Recipe:
    return Recipe.model_validate({"name": name, "seed": seed, "layers": layers, "master": _FMT})


def concept_pack(concept: str, n: int = 6) -> list[tuple[str, Recipe]]:
    """컨셉에 어울리는 대표 자막 효과음 묶음. (라벨, Recipe) 목록(최대 n개)."""
    if concept == "교양":
        # 다큐/교양: 절제되고 잔잔한 큐
        items: list[tuple[str, Recipe]] = [
            ("전환_휙", _mk("교양_전환", [{"synth": {"kind": "whoosh", "freq": 350, "decay": 0.7},
              "transforms": [{"op": "reverb", "kind": "room", "wet": 0.25}, {"op": "fade", "out": 0.2}], "gain_db": -4}])),
            ("포인트_띵", _mk("교양_포인트", [{"synth": {"kind": "fm", "freq": 1320, "decay": 0.6, "ratio": 2.0, "index": 2},
              "transforms": [{"op": "eq", "freq_hz": 3000, "gain_db": 2, "q": 1.0}], "gain_db": -3}])),
            ("잔잔한_차임", _mk("교양_차임", [{"synth": {"kind": "modal", "material": "metal", "freq": 740, "decay": 2.0},
              "transforms": [{"op": "reverb", "kind": "hall", "wet": 0.3}], "gain_db": -4}])),
            ("인서트_스윕", _mk("교양_인서트", [{"synth": {"kind": "noise", "decay": 0.6},
              "transforms": [{"op": "filter", "kind": "highpass", "cutoff_hz": 2000}, {"op": "fade", "in": 0.1, "out": 0.3}], "gain_db": -8}])),
            ("은은한_반짝", _mk("교양_반짝", [{"synth": {"kind": "tone", "freq": 2200, "decay": 0.9},
              "transforms": [{"op": "delay", "seconds": 0.12, "feedback": 0.25, "mix": 0.3}], "gain_db": -7}])),
            ("낮은_강조", _mk("교양_강조", [{"synth": {"kind": "tone", "freq": 196, "decay": 1.0},
              "transforms": [{"op": "spectral", "mode": "freeze", "amount": 3.0}, {"op": "filter", "kind": "lowpass", "cutoff_hz": 2000}, {"op": "fade", "in": 0.3, "out": 0.8}], "gain_db": -5}])),
        ]
    else:  # 예능(기본): 또렷하고 코믹한 아이코닉 큐
        boing = _mk("예능_띠용", [{"synth": {"kind": "fm", "freq": 420, "decay": 0.5, "ratio": 2.0, "index": 6},
                    "transforms": [{"op": "chorus", "rate_hz": 3, "depth": 0.4, "mix": 0.5}, {"op": "pitch", "semitones": 3}]}])
        accent = _mk("예능_반짝", [{"synth": {"kind": "tone", "freq": 1568, "decay": 0.6},
                     "transforms": [{"op": "delay", "seconds": 0.1, "feedback": 0.3, "mix": 0.4}, {"op": "eq", "freq_hz": 5000, "gain_db": 3, "q": 1.0}]}])
        wrong = _mk("예능_땡", [{"synth": {"kind": "fm", "freq": 180, "decay": 0.5, "ratio": 1.0, "index": 1.5},
                    "transforms": [{"op": "distortion", "drive_db": 6}, {"op": "filter", "kind": "lowpass", "cutoff_hz": 1500}]}])
        items = [
            ("두둥_반전", get_recipe("두둥_반전")),
            ("짜잔_등장", get_recipe("짜잔_등장")),
            ("두구두구_정답공개", get_recipe("두구두구_정답공개")),
            ("띠용_놀람", boing),
            ("반짝_포인트", accent),
            ("와장창_깨짐", get_recipe("와장창_깨짐")),
            ("땡_오답", wrong),
            ("정적_썰렁", get_recipe("정적_썰렁")),
        ]
    return items[:max(1, n)]
