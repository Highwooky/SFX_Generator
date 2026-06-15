"""절차적 합성 프리미티브.

설계 의도(Why):
- 원본 음원 없이 수학적으로 파형을 생성하므로 저작권으로부터 완전히 자유롭다.
- 모든 함수는 (samplerate, ...) -> np.ndarray(float32, mono, [-1,1]) 규약을 지킨다.
- 난수는 외부에서 받은 rng를 사용해 seed 기반 재현성을 보장한다.
"""

from __future__ import annotations

import numpy as np

from .recipe import SynthSource


def _adsr_decay(n: int, sr: int, decay: float) -> np.ndarray:
    """간단한 지수 감쇠 엔벨로프. 타악/임팩트성 소리의 자연스러운 꼬리를 만든다."""
    t = np.arange(n) / sr
    # 감쇠 시간 동안 약 -60dB까지 떨어지도록 시상수 설정
    tau = max(decay, 1e-3) / 6.9
    return np.exp(-t / tau).astype(np.float32)


def sub_impact(sr: int, freq: float, decay: float) -> np.ndarray:
    """저음 '쿵' 임팩트: 피치가 빠르게 떨어지는 사인 + 클릭 트랜지언트."""
    n = int(sr * decay)
    t = np.arange(n) / sr
    # 시작 주파수에서 절반으로 글라이드 다운 → 묵직한 임팩트감
    f = freq * np.exp(-t * 6.0) * 0.5 + freq * 0.5
    phase = 2 * np.pi * np.cumsum(f) / sr
    body = np.sin(phase) * _adsr_decay(n, sr, decay)
    # 어택을 또렷하게 하는 짧은 클릭
    click_n = int(sr * 0.005)
    body[:click_n] += np.linspace(1.0, 0.0, click_n) * 0.5
    return (body * 0.9).astype(np.float32)


def tone(sr: int, freq: float, decay: float, rng: np.random.Generator) -> np.ndarray:
    """배음이 약간 섞인 톤(비프/알림음). UI·인서트 효과음에 적합."""
    n = int(sr * decay)
    t = np.arange(n) / sr
    sig = np.sin(2 * np.pi * freq * t)
    sig += 0.25 * np.sin(2 * np.pi * freq * 2 * t)  # 2배음
    sig += 0.12 * np.sin(2 * np.pi * freq * 3 * t)  # 3배음
    env = _adsr_decay(n, sr, decay)
    # 어택 5ms 램프(클릭 노이즈 방지)
    a = int(sr * 0.005)
    env[:a] *= np.linspace(0, 1, a)
    return (sig * env * 0.6).astype(np.float32)


def noise(sr: int, decay: float, rng: np.random.Generator) -> np.ndarray:
    """감쇠하는 화이트 노이즈 버스트. 충돌·파열·텍스처 레이어용."""
    n = int(sr * decay)
    sig = rng.standard_normal(n).astype(np.float32)
    return (sig * _adsr_decay(n, sr, decay) * 0.5).astype(np.float32)


def whoosh(sr: int, freq: float, decay: float, rng: np.random.Generator) -> np.ndarray:
    """휙 지나가는 소리: 노이즈에 진폭 스웰 + 후처리 필터 스윕을 가정한 종 모양 엔벨로프."""
    n = int(sr * decay)
    sig = rng.standard_normal(n).astype(np.float32)
    t = np.linspace(0, 1, n)
    # 종(bell) 모양 엔벨로프 → 가까이 왔다가 멀어지는 느낌
    env = np.exp(-((t - 0.5) ** 2) / (2 * 0.15**2)).astype(np.float32)
    return (sig * env * 0.6).astype(np.float32)


def riser(sr: int, freq: float, decay: float, rng: np.random.Generator) -> np.ndarray:
    """점점 고조되는 라이저: 상승 톤 + 점증 노이즈. 긴장감/전환 빌드업용."""
    n = int(sr * decay)
    t = np.arange(n) / sr
    sweep = freq * np.exp(np.linspace(0, np.log(8), n))  # 3옥타브 상승
    phase = 2 * np.pi * np.cumsum(sweep) / sr
    tone_part = np.sin(phase)
    noise_part = rng.standard_normal(n).astype(np.float32) * 0.4
    swell = np.linspace(0.0, 1.0, n) ** 2  # 끝으로 갈수록 가파르게 커짐
    return ((tone_part + noise_part) * swell * 0.5).astype(np.float32)


# 재질별 공진 모드 비율(기본 주파수 대비). 모드 비율이 재질감을 결정한다.
_MODAL_RATIOS = {
    "metal": [1.0, 2.76, 5.40, 8.93, 13.34],   # 비조화 → 금속/벨
    "glass": [1.0, 2.32, 4.25, 6.63, 9.38],     # 밝고 가는 비조화
    "wood": [1.0, 1.59, 2.14, 2.30, 2.65],      # 둔탁한 저차 모드
}


def modal(sr: int, freq: float, decay: float, material: str = "metal") -> np.ndarray:
    """모달 합성: 재질별 공진 모드의 합. 금속/유리/나무 타격을 파라미터로 생성."""
    n = int(sr * decay)
    t = np.arange(n) / sr
    ratios = _MODAL_RATIOS.get(material, _MODAL_RATIOS["metal"])
    sig = np.zeros(n, dtype=np.float32)
    for i, r in enumerate(ratios):
        f = freq * r
        if f >= sr / 2:  # 나이퀴스트 초과 모드는 스킵(에일리어싱 방지)
            continue
        # 고차 모드일수록 빠르게 감쇠하고 작게(자연스러운 음색)
        amp = 1.0 / (i + 1) ** 1.2
        mode_decay = decay / (1 + i * 0.6)
        env = np.exp(-t / (max(mode_decay, 1e-3) / 6.9))
        sig += (np.sin(2 * np.pi * f * t) * env * amp).astype(np.float32)
    # 짧은 어택 트랜지언트
    a = int(sr * 0.003)
    if a > 0:
        sig[:a] *= np.linspace(0, 1, a)
    peak = float(np.max(np.abs(sig))) or 1.0
    return (sig / peak * 0.85).astype(np.float32)


def pluck(sr: int, freq: float, decay: float, rng: np.random.Generator) -> np.ndarray:
    """Karplus-Strong 물리 모델: 현/튕김/플럭. 노이즈 버스트를 지연선으로 순환."""
    n = int(sr * decay)
    p = max(2, int(sr / max(freq, 20)))  # 지연선 길이 = 파장
    buf = (rng.standard_normal(p)).astype(np.float32)  # 초기 여기(excitation)
    out = np.zeros(n, dtype=np.float32)
    # 감쇠 계수: decay에 맞춰 조정(클수록 길게 울림)
    damp = 0.996 + 0.003 * min(decay / 2.0, 1.0)
    idx = 0
    for i in range(n):
        out[i] = buf[idx]
        nxt = (idx + 1) % p
        buf[idx] = damp * 0.5 * (buf[idx] + buf[nxt])  # 평균 → 저역통과(현 감쇠)
        idx = nxt
    peak = float(np.max(np.abs(out))) or 1.0
    return (out / peak * 0.8).astype(np.float32)


def fm(sr: int, freq: float, decay: float, ratio: float, index: float) -> np.ndarray:
    """FM 합성: 전자음/레이저/sci-fi. 변조비(ratio)·강도(index)로 음색 결정."""
    n = int(sr * decay)
    t = np.arange(n) / sr
    env = np.exp(-t / (max(decay, 1e-3) / 6.9)).astype(np.float32)
    mod = np.sin(2 * np.pi * freq * ratio * t) * index * env  # 변조 신호도 감쇠
    sig = np.sin(2 * np.pi * freq * t + mod) * env
    a = int(sr * 0.004)
    if a > 0:
        sig[:a] *= np.linspace(0, 1, a)
    return (sig * 0.7).astype(np.float32)


def _resonant_noise(sr: int, n: int, rng, center: float, q: float, gain: float) -> np.ndarray:
    """노이즈를 2차 대역통과 공진 필터에 통과(자연 텍스처의 빌딩블록)."""
    x = rng.standard_normal(n).astype(np.float32)
    w0 = 2 * np.pi * center / sr
    alpha = np.sin(w0) / (2 * q)
    b0, b1, b2 = alpha, 0.0, -alpha
    a0, a1, a2 = 1 + alpha, -2 * np.cos(w0), 1 - alpha
    y = np.zeros(n, dtype=np.float32)
    x1 = x2 = y1 = y2 = 0.0
    for i in range(n):
        xi = x[i]
        yi = (b0 * xi + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2) / a0
        x2, x1 = x1, xi
        y2, y1 = y1, yi
        y[i] = yi
    return (y * gain).astype(np.float32)


def wind(sr: int, freq: float, decay: float, rng: np.random.Generator) -> np.ndarray:
    """바람: 저역 공진 노이즈에 느린 진폭 변조(거센 정도가 출렁임)."""
    n = int(sr * decay)
    base = _resonant_noise(sr, n, rng, center=max(min(freq, 1200), 300), q=2.0, gain=3.0)
    t = np.arange(n) / sr
    lfo = 0.5 + 0.5 * np.sin(2 * np.pi * 0.4 * t + rng.uniform(0, 6.28))  # 느린 출렁임
    gust = 0.5 + 0.5 * np.sin(2 * np.pi * 0.13 * t)  # 더 느린 돌풍
    sig = base * (0.4 + 0.6 * lfo * gust)
    peak = float(np.max(np.abs(sig))) or 1.0
    return (sig / peak * 0.7).astype(np.float32)


def rain(sr: int, freq: float, decay: float, rng: np.random.Generator) -> np.ndarray:
    """비: 고역 쉬익 노이즈 베드 + 무작위 물방울 임펄스."""
    n = int(sr * decay)
    bed = _resonant_noise(sr, n, rng, center=4000, q=0.7, gain=1.4) * 0.5
    drops = np.zeros(n, dtype=np.float32)
    count = int(decay * 220)  # 밀도
    pos = rng.integers(0, max(n - 1, 1), size=count)
    drops[pos] = rng.uniform(0.3, 1.0, size=count)
    # 물방울에 짧은 감쇠 꼬리
    k = np.exp(-np.arange(int(sr * 0.02)) / (sr * 0.004)).astype(np.float32)
    drops = np.convolve(drops, k)[:n]
    sig = bed + drops * 0.6
    peak = float(np.max(np.abs(sig))) or 1.0
    return (sig / peak * 0.7).astype(np.float32)


def fire(sr: int, freq: float, decay: float, rng: np.random.Generator) -> np.ndarray:
    """불: 낮고 조용한 럼블 + 또렷하고 드문드문한 '탁탁' 크래클이 지배적."""
    n = int(sr * decay)
    # 럼블은 배경으로만 아주 낮게(일정한 hiss로 들리지 않게)
    rumble = _resonant_noise(sr, n, rng, center=160, q=1.2, gain=1.6) * 0.22
    # 크래클: 드문드문한 임펄스에 짧은 '스냅' 바디(감쇠 사인 버스트)를 입힘
    crackle = np.zeros(n, dtype=np.float32)
    count = int(decay * 28)
    pos = rng.integers(0, max(n - 1, 1), size=count)
    crackle[pos] = rng.uniform(0.5, 1.0, size=count) * rng.choice([1.0, -1.0], size=count)
    bl = int(sr * 0.012)
    bt = np.arange(bl) / sr
    snap = (np.sin(2 * np.pi * rng.uniform(800, 2200) * bt) * np.exp(-bt / 0.0025)).astype(np.float32)
    crackle = np.convolve(crackle, snap)[:n]
    # 가끔 더 큰 '팝'
    big = rng.integers(0, max(n - 1, 1), size=max(2, int(decay * 4)))
    pop = np.zeros(n, dtype=np.float32)
    pop[big] = rng.uniform(0.8, 1.0, size=len(big))
    pk = np.exp(-np.arange(int(sr * 0.04)) / (sr * 0.006)).astype(np.float32)
    pop = np.convolve(pop, pk)[:n]
    sig = rumble + crackle * 0.9 + pop * 0.5
    peak = float(np.max(np.abs(sig))) or 1.0
    return (sig / peak * 0.8).astype(np.float32)


def synthesize(spec: SynthSource, sr: int, rng: np.random.Generator) -> np.ndarray:
    """SynthSource 스펙을 받아 해당 합성 파형을 생성하는 디스패처."""
    decay = spec.duration if spec.duration is not None else spec.decay
    kind = spec.kind
    if kind == "sub_impact":
        return sub_impact(sr, spec.freq, decay)
    if kind == "tone":
        return tone(sr, spec.freq, decay, rng)
    if kind == "noise":
        return noise(sr, decay, rng)
    if kind == "whoosh":
        return whoosh(sr, spec.freq, decay, rng)
    if kind == "riser":
        return riser(sr, spec.freq, decay, rng)
    if kind == "modal":
        return modal(sr, spec.freq, decay, spec.material or "metal")
    if kind == "pluck":
        return pluck(sr, spec.freq, decay, rng)
    if kind == "fm":
        return fm(sr, spec.freq, decay, spec.ratio, spec.index)
    if kind == "wind":
        return wind(sr, spec.freq, decay, rng)
    if kind == "rain":
        return rain(sr, spec.freq, decay, rng)
    if kind == "fire":
        return fire(sr, spec.freq, decay, rng)
    # 스키마에서 Literal로 제한되므로 정상 흐름상 도달 불가(방어적 처리)
    raise ValueError(f"알 수 없는 합성 종류: {kind}")
