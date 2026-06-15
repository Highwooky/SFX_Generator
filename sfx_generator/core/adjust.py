"""조절 노브: 생성된 레시피에 사용자 슬라이더 값을 변형으로 얹는다.

설계 의도(Why):
- 텍스트로 한 번에 정확히 맞히긴 어렵다. 대신 '나온 소리'를 밝기·피치·어택·공간감·
  무게·거칠기 슬라이더로 밀고 당겨 상상에 수렴시킨다(AI 없이 정확도를 올리는 현실적 방법).
- 매번 '원본 레시피'에서 새로 적용하므로 값이 누적되지 않는다(되돌리기 안전).
"""

from __future__ import annotations

from .recipe import Recipe


def apply_knobs(recipe: Recipe, *, brightness: float = 0.0, pitch: float = 0.0,
                attack: float = 0.0, space: float = 0.0, weight: float = 0.0,
                grit: float = 0.0) -> Recipe:
    """노브 값을 각 레이어 변형 체인 끝에 추가한 새 레시피를 반환.

    brightness: -12~+12 dB(고역), pitch: -12~+12 반음, attack: 0~0.5초,
    space: 0~0.9 리버브, weight: 0~12 dB(저역), grit: 0~15 dB 디스토션.
    """
    d = recipe.model_dump(by_alias=True)
    for layer in d.get("layers", []):
        tf = list(layer.get("transforms") or [])
        if abs(pitch) >= 0.1:
            tf.append({"op": "pitch", "semitones": float(pitch)})
        if weight > 0.1:
            tf.append({"op": "eq", "freq_hz": 90, "gain_db": float(min(weight, 24)), "q": 0.7})
        if brightness > 0.1:
            tf.append({"op": "eq", "freq_hz": 6000, "gain_db": float(min(brightness, 24)), "q": 0.8})
        elif brightness < -0.1:
            cutoff = max(300.0, 6000.0 + brightness * 400.0)  # 어두울수록 로우패스 ↓
            tf.append({"op": "filter", "kind": "lowpass", "cutoff_hz": cutoff})
        if grit > 0.1:
            tf.append({"op": "distortion", "drive_db": float(min(grit, 24))})
        if space > 0.01:
            tf.append({"op": "reverb", "kind": "hall", "wet": float(min(space, 0.95))})
        if attack > 0.005:
            tf.append({"op": "fade", "in": float(attack)})
        layer["transforms"] = tf
    return Recipe.model_validate(d)
