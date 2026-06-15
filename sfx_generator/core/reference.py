"""레퍼런스 음원 분석 → 비슷한 합성 레시피 생성(정밀판).

설계 의도(Why):
- 참고 음원의 거시 특징을 분석해 '근사 합성 레시피'를 만든다. 샘플을 복제하지 않으므로
  저작권에서 자유롭다. 대신 음색의 '계열'을 맞춘 합성이라 사실감은 근사치다.
- 정밀화 포인트:
  1) 감쇠 시간(decay_time)을 실측해 길이가 아닌 '울림'에 맞춘다.
  2) 어택 트랜지언트를 별도 노이즈 레이어로 분리(타격감 재현).
  3) 자기상관 기반 피치/하모니시티로 tone/modal/pluck/sub_impact를 정확히 고른다.
  4) 3밴드 스펙트럼 틸트로 밝기 균형을 EQ/필터로 맞춘다.
  5) 토널+노이즈 혼합도(flatness)에 따라 노이즈 베드를 섞는다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from .recipe import Recipe

_FMT = {"loudness_lufs": -16, "limiter": True, "format": {"bit": 24, "rate": 48000}, "prefix": "[SFX]"}


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def analyze_reference(path: Path) -> dict:
    """음원을 분석해 특징 딕셔너리를 반환."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if mono.size == 0:
        raise ValueError("빈 오디오입니다.")
    sr = int(sr or 48000)
    dur = len(mono) / sr
    peak_abs = float(np.max(np.abs(mono))) or 1e-9
    rms_all = float(np.sqrt(np.mean(mono ** 2) + 1e-12))

    # ── 엔벨로프(프레임 RMS): 어택/감쇠 시간 실측 ──
    hop = 256
    win = 1024
    n_fr = max(1, (len(mono) - 1) // hop)
    env = np.array([np.sqrt(np.mean(mono[i * hop:i * hop + win] ** 2) + 1e-12) for i in range(n_fr)])
    peak_idx = int(np.argmax(env))
    attack_time = peak_idx * hop / sr
    epk = env[peak_idx] or 1e-9
    # 피크 이후 −20dB(10%)로 떨어지는 시점 → 감쇠 시간
    after = env[peak_idx:]
    below = np.where(after < epk * 0.1)[0]
    decay_time = (below[0] * hop / sr) if below.size else max(0.05, dur - attack_time)
    decay_time = _clamp(decay_time, 0.05, 10.0)
    sustained = below.size == 0 or decay_time > 1.0
    # 타격성: 어택이 빠르고 초반 에너지 집중
    half = len(mono) // 2 or 1
    early = float(np.sum(mono[:half] ** 2))
    late = float(np.sum(mono[half:] ** 2)) + 1e-9
    percussive = (attack_time < 0.08) and (early > late * 2.5)

    # ── 분석 구간(active region): 어택~감쇠까지만. 꼬리 무음/감쇠가 밝기·피치를
    #    왜곡하므로 실제 울리는 부분만 본다. ──
    a_start = max(0, peak_idx * hop - int(0.01 * sr))
    a_len = int((attack_time + decay_time + 0.1) * sr)
    act = mono[a_start:a_start + a_len]
    if act.size < 2048:
        act = mono  # 너무 짧으면 전체 사용

    # ── 스펙트럼: centroid·flatness·3밴드 틸트 ──
    w = act * np.hanning(len(act))
    spec = np.abs(np.fft.rfft(w)) + 1e-9
    freqs = np.fft.rfftfreq(len(act), 1.0 / sr)
    centroid = float((freqs * spec).sum() / spec.sum())
    flatness = float(np.exp(np.mean(np.log(spec))) / np.mean(spec))
    total = float(spec.sum())
    low = float(spec[freqs < 300].sum()) / total
    mid = float(spec[(freqs >= 300) & (freqs < 3000)].sum()) / total
    high = float(spec[freqs >= 3000].sum()) / total

    # ── 피치: 스펙트럼 최강 피크(견고). 하모니시티는 평탄도에서 근사(요약용). ──
    band = (freqs >= 40) & (freqs <= 8000)
    dominant = float(freqs[band][np.argmax(spec[band])]) if band.any() else 220.0
    pitch = dominant
    harmonicity = _clamp(1.0 - flatness * 1.5, 0.0, 1.0)

    # ── 부분음(partials): 스펙트럼 상위 피크 → 화음/복합 타격 재현용 ──
    mag = spec.copy()
    peak_freqs: list[float] = []
    peak_mags: list[float] = []
    thr = 0.15 * float(mag.max())
    # 국소 최대 + 임계 이상만
    for i in range(1, len(mag) - 1):
        fhz = freqs[i]
        if 40 <= fhz <= 8000 and mag[i] >= thr and mag[i] > mag[i - 1] and mag[i] >= mag[i + 1]:
            # 인접 피크 병합(±25Hz)
            if peak_freqs and abs(fhz - peak_freqs[-1]) < 25:
                if mag[i] > peak_mags[-1]:
                    peak_freqs[-1], peak_mags[-1] = float(fhz), float(mag[i])
                continue
            peak_freqs.append(float(fhz))
            peak_mags.append(float(mag[i]))
    order = np.argsort(peak_mags)[::-1][:4] if peak_mags else []
    partials = [(peak_freqs[j], peak_mags[j] / (peak_mags[order[0]] if len(order) else 1.0)) for j in order]

    # ── 진폭 엔벨로프(브레이크포인트 ~24점, 시간0~1·게인0~1) ──
    env_norm = env / (float(env.max()) or 1e-9)
    npts = min(24, len(env_norm))
    idx = np.linspace(0, len(env_norm) - 1, npts).astype(int)
    envelope = [(float(i / (len(env_norm) - 1 or 1)), float(_clamp(env_norm[i], 0.0, 1.0))) for i in idx]

    return {
        "duration": dur, "centroid": centroid, "flatness": flatness,
        "dominant_freq": pitch, "percussive": percussive, "attack_time": attack_time,
        "decay_time": decay_time, "sustained": sustained, "harmonicity": harmonicity,
        "bands": {"low": low, "mid": mid, "high": high},
        "peak": peak_abs, "rms": rms_all,
        "partials": partials, "envelope": envelope,
    }


def recipe_from_features(feat: dict, name: str = "레퍼런스_매칭") -> tuple[Recipe, str]:
    """특징을 다층 합성 레시피로 매핑. (Recipe, 설명문) 반환."""
    dur = float(min(max(feat["duration"], 0.05), 10.0))
    centroid = float(feat["centroid"])
    flat = float(feat["flatness"])
    perc = bool(feat["percussive"])
    attack = _clamp(feat.get("attack_time", 0.0), 0.0, 0.3)
    decay = _clamp(feat.get("decay_time", dur), 0.05, 10.0)
    harm = float(feat.get("harmonicity", _clamp(1.0 - flat * 1.5, 0.0, 1.0)))
    sustained = bool(feat.get("sustained", not perc))
    bands = feat.get("bands")
    freq = _clamp(feat["dominant_freq"], 40.0, 8000.0)

    tonal = (flat < 0.3) or (harm > 0.55)
    layers: list[dict] = []

    # ── 바디 레이어(견고한 규칙: 밝기·감쇠·주파수 조합) ──
    material = "metal"
    if tonal:
        f = freq
        if not perc:
            kind, desc = "tone", "지속 톤"
        elif centroid > 2500:
            kind, desc, material = "modal", "공진 타격(모달/glass)", "glass"
        elif centroid > 900:
            if decay > 0.35:
                kind, desc, material = "modal", "공진 타격(모달/metal)", "metal"
            else:
                kind, desc = "pluck", "플럭/현 타격"
        else:  # 어두운 타격
            kind, desc = ("sub_impact", "저음 임팩트") if f < 180 else ("pluck", "플럭/현 타격")
    else:
        if centroid < 900 and decay > 1.0:
            kind, f, desc = "wind", _clamp(centroid, 300, 1200), "지속 바람형 텍스처"
        else:
            kind, f, desc = "noise", _clamp(centroid, 100, 8000), "노이즈 텍스처"

    envelope = feat.get("envelope")
    partials = feat.get("partials") or []
    # 엔벨로프 폴로잉을 쓸 땐 합성음을 '전체 길이'로 채우고 음량 곡선으로 모양을 만든다.
    # (합성 자체 감쇠 + 엔벨로프 이중 감쇠를 피하기 위함)
    body_decay = dur if envelope else decay

    body_synth: dict = {"kind": kind, "freq": f, "decay": body_decay}
    if kind == "modal":
        body_synth["material"] = material

    # 스펙트럼 틸트 보정(3밴드 기반, 없으면 centroid로 추정)
    body_tf: list[dict] = []
    if bands is None:
        bands = {"low": 0.34, "mid": 0.33, "high": 0.33}
        if centroid > 3500:
            bands = {"low": 0.15, "mid": 0.3, "high": 0.55}
        elif centroid < 700:
            bands = {"low": 0.6, "mid": 0.3, "high": 0.1}
    if bands["high"] > 0.45:
        body_tf.append({"op": "eq", "freq_hz": 6000, "gain_db": 4, "q": 0.9})
    if bands["low"] > 0.55:
        body_tf.append({"op": "filter", "kind": "lowpass", "cutoff_hz": 2500})
    elif bands["high"] < 0.08 and kind not in ("wind",):
        body_tf.append({"op": "filter", "kind": "lowpass", "cutoff_hz": _clamp(centroid * 3, 800, 8000)})
    # 다이내믹스: 엔벨로프 폴로잉(있으면) 또는 어택 페이드(부드러운 시작)
    if envelope:
        body_tf.append({"op": "envelope", "points": envelope})
    elif attack > 0.008 and not perc:
        body_tf.append({"op": "fade", "in": _clamp(attack, 0, decay * 0.6)})
    body_synth_layer = {"synth": body_synth, "transforms": body_tf} if body_tf else {"synth": body_synth}
    layers.append(body_synth_layer)

    # ── 다중 피치: 화음/복합 타격이면 부분음마다 톤 레이어 추가(모달은 자체 배음 → 제외) ──
    if tonal and kind in ("tone", "pluck") and len(partials) > 1:
        for pf, rel in partials[1:3]:  # 으뜸음 외 상위 2개
            pf = _clamp(pf, 40.0, 8000.0)
            if abs(pf - f) < 30:  # 으뜸음과 너무 가까우면 스킵
                continue
            ptf = [{"op": "envelope", "points": envelope}] if envelope else []
            layers.append({
                "synth": {"kind": kind, "freq": pf, "decay": body_decay},
                "transforms": ptf,
                "gain_db": _clamp(-6 + 6 * float(rel), -18, -2),
            })

    # ── 어택 트랜지언트 레이어(타격이면 별도 클릭으로 어택감 재현) ──
    if perc and attack < 0.05:
        click_hp = _clamp(max(1500.0, centroid), 200.0, 12000.0)
        layers.append({
            "synth": {"kind": "noise", "decay": min(0.08, decay)},
            "transforms": [{"op": "filter", "kind": "highpass", "cutoff_hz": click_hp},
                           {"op": "fade", "out": 0.05}],
            "gain_db": -7.0 if tonal else -2.0, "start": 0.0,
        })

    # ── 토널+노이즈 혼합이면 노이즈 베드 섞기 ──
    if tonal and 0.12 < flat < 0.5:
        layers.append({
            "synth": {"kind": "noise", "decay": decay},
            "transforms": [{"op": "filter", "kind": "highpass", "cutoff_hz": 1500},
                           {"op": "fade", "out": _clamp(decay * 0.6, 0, decay)}],
            "gain_db": _clamp(-12 + 14 * flat, -18, -4),
        })

    recipe = Recipe.model_validate({"name": name, "seed": 0, "layers": layers, "master": _FMT})
    bright = "밝음" if centroid > 3500 else "어두움" if centroid < 700 else "중간"
    extra = ""
    if envelope:
        extra += " · 엔벨로프매칭"
    if tonal and kind in ("tone", "pluck") and len(partials) > 1:
        extra += f" · 부분음{min(len(partials), 3)}개"
    summary = (f"{desc} · 길이 {dur:.2f}s · 감쇠 {decay:.2f}s · {bright}"
               f" · {'타격성' if perc else '지속성'} · 기본 {freq:.0f}Hz"
               f" · {len(layers)}레이어{extra}")
    return recipe, summary


def recipe_from_reference(path: Path, name: str = "레퍼런스_매칭") -> tuple[Recipe, str]:
    """경로 하나로 분석→레시피까지 한 번에."""
    return recipe_from_features(analyze_reference(Path(path)), name)


# 라이브러리 매칭 시 resolver가 인식하는 센티넬 쿼리
REF_MATCH_QUERY = "__refmatch__"


def feature_tags(feat: dict) -> list[str]:
    """특징을 라이브러리 자동 태그 어휘(short/bright/percussive 등)로 변환."""
    dur = feat["duration"]
    cen = feat["centroid"]
    flat = feat["flatness"]
    perc = bool(feat["percussive"])
    tags = [
        "short" if dur < 0.5 else "long" if dur > 2.5 else "medium",
        "bright" if cen > 3500 else "dark" if cen < 800 else "mid",
        "percussive" if perc else "sustained",
    ]
    if flat > 0.35:
        tags.append("noisy")
    elif flat < 0.08:
        tags.append("tonal")
    return tags


def recipe_from_library_match(matched: Path, name: str = "레퍼런스_매칭") -> Recipe:
    """매칭된 실제 샘플을 쓰는 소스 기반 레시피(센티넬 쿼리). 렌더 시 resolver가 해당 파일을 반환."""
    return Recipe.model_validate({
        "name": name, "seed": 0,
        "layers": [{"source": {"query": REF_MATCH_QUERY, "pick": "best"}}],
        "master": _FMT,
    })
