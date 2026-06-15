"""레시피 스키마: 프롬프트 해석기와 DSP 엔진을 잇는 단일 계약(contract).

설계 의도(Why):
- 해석기(LLM/룰)와 렌더 엔진을 이 스키마로 완전히 분리한다. 양쪽이 같은
  JSON 계약만 지키면 서로를 독립적으로 개발/교체할 수 있다.
- 변형(transform)은 'op' 필드로 구분되는 판별 유니온(discriminated union)으로
  정의한다. 잘못된 파라미터를 렌더 직전이 아니라 '레시피 검증 시점'에 잡기 위함.
- seed를 최상위에 둬서 동일 레시피의 변주(variation) 재현성을 보장한다.
"""

from __future__ import annotations

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator

# ──────────────────────────────────────────────────────────────────────────
# 변형(Transform) 연산 정의 — op 별로 타입을 분리해 검증 강도를 높인다.
# 새 op 추가 시: ① 여기에 모델 추가 ② TransformUnion에 등록 ③ dsp.py에 핸들러 추가
# ──────────────────────────────────────────────────────────────────────────


class PitchTransform(BaseModel):
    op: Literal["pitch"]
    semitones: float = Field(..., ge=-24, le=24, description="반음 단위 피치 이동")


class StretchTransform(BaseModel):
    op: Literal["stretch"]
    # rate > 1 이면 길어짐(느려짐), < 1 이면 짧아짐(빨라짐)
    rate: float = Field(..., gt=0.1, le=10.0, description="시간 신축 배율")


class ReverseTransform(BaseModel):
    op: Literal["reverse"]


class GainTransform(BaseModel):
    op: Literal["gain"]
    db: float = Field(..., ge=-60, le=24)


class FadeTransform(BaseModel):
    op: Literal["fade"]
    in_: float = Field(0.0, ge=0, alias="in", description="페이드 인 길이(초)")
    out: float = Field(0.0, ge=0, description="페이드 아웃 길이(초)")

    model_config = {"populate_by_name": True}


class FilterTransform(BaseModel):
    op: Literal["filter"]
    kind: Literal["highpass", "lowpass"]
    cutoff_hz: float = Field(..., gt=10, le=22000)


class EqTransform(BaseModel):
    """피크 EQ 한 밴드. 여러 밴드가 필요하면 transform을 여러 개 체인한다."""

    op: Literal["eq"]
    freq_hz: float = Field(..., gt=10, le=22000)
    gain_db: float = Field(..., ge=-24, le=24)
    q: float = Field(1.0, gt=0.1, le=18.0)


class ReverbTransform(BaseModel):
    op: Literal["reverb"]
    kind: Literal["room", "hall", "plate"] = "hall"
    wet: float = Field(0.3, ge=0, le=1.0)


class DistortionTransform(BaseModel):
    op: Literal["distortion"]
    drive_db: float = Field(..., ge=0, le=60)


class ChorusTransform(BaseModel):
    op: Literal["chorus"]
    rate_hz: float = Field(1.0, gt=0, le=100)
    depth: float = Field(0.25, ge=0, le=1.0)
    mix: float = Field(0.5, ge=0, le=1.0)


class DelayTransform(BaseModel):
    op: Literal["delay"]
    seconds: float = Field(..., gt=0, le=5.0)
    feedback: float = Field(0.3, ge=0, le=0.95)
    mix: float = Field(0.4, ge=0, le=1.0)


class BitcrushTransform(BaseModel):
    op: Literal["bitcrush"]
    bit_depth: float = Field(..., ge=1, le=24)


class NormalizeTransform(BaseModel):
    """레이어 단위 피크 정규화(마스터 라우드니스와 별개)."""

    op: Literal["normalize"]
    peak_db: float = Field(-1.0, ge=-24, le=0)


class GranularTransform(BaseModel):
    """그래뉼러 합성: 짧은 샘플을 잘게 쪼개 재배치 → 텍스처/구름/스웰.

    적은 샘플 하나로 무한에 가까운 변주를 만드는 핵심 도구.
    """

    op: Literal["granular"]
    grain_ms: float = Field(80.0, ge=5, le=500, description="그레인 길이(ms)")
    density: float = Field(2.0, ge=0.5, le=8.0, description="겹침 밀도(↑일수록 두터움)")
    pitch_jitter: float = Field(2.0, ge=0, le=12, description="그레인별 피치 흔들림(반음)")
    stretch: float = Field(1.0, gt=0.1, le=20.0, description="결과 길이 배율")
    spray_ms: float = Field(20.0, ge=0, le=500, description="그레인 위치 분산(ms)")


class SpectralTransform(BaseModel):
    """스펙트럴 처리(페이즈 보코더): 프리즈/스트레치/블러 → 드론·패드화."""

    op: Literal["spectral"]
    mode: Literal["freeze", "stretch", "blur"] = "freeze"
    amount: float = Field(2.0, gt=0.1, le=20.0, description="freeze/stretch 길이배율, blur 강도")


class EnvelopeTransform(BaseModel):
    """진폭 엔벨로프 폴로잉: (시간0~1, 게인0~1) 브레이크포인트 곡선을 신호에 곱한다.

    레퍼런스 음원의 실제 음량 곡선을 합성음에 입혀 어택/감쇠 '다이내믹스'를 재현한다.
    """

    op: Literal["envelope"]
    points: list[tuple[float, float]] = Field(..., min_length=2, max_length=128)


# 판별 유니온: 'op' 값으로 어떤 모델인지 자동 판별 → 명확한 검증 에러
TransformUnion = Annotated[
    Union[
        PitchTransform,
        StretchTransform,
        ReverseTransform,
        GainTransform,
        FadeTransform,
        FilterTransform,
        EqTransform,
        ReverbTransform,
        DistortionTransform,
        ChorusTransform,
        DelayTransform,
        BitcrushTransform,
        NormalizeTransform,
        GranularTransform,
        SpectralTransform,
        EnvelopeTransform,
    ],
    Field(discriminator="op"),
]


# ──────────────────────────────────────────────────────────────────────────
# 음원(Source) / 합성(Synth) — 레이어는 둘 중 하나만 가진다.
# ──────────────────────────────────────────────────────────────────────────


class LibrarySource(BaseModel):
    """보유 라이브러리에서 베이스 샘플을 가져온다(저작권 안전 경로)."""

    query: str = Field(..., min_length=1, description="라이브러리 검색어")
    pick: Literal["best", "random"] = "best"


class SynthSource(BaseModel):
    """절차적 합성으로 소리를 생성한다(원본 없음 → 저작권 프리)."""

    kind: Literal[
        "sub_impact", "tone", "noise", "whoosh", "riser",
        # 확장: 모달(재질 타격), 플럭(현/튕김), FM(전자/레이저), 자연 텍스처
        "modal", "pluck", "fm", "wind", "rain", "fire",
    ]
    freq: float = Field(220.0, gt=1, le=22000, description="기본 주파수(Hz)")
    decay: float = Field(0.8, gt=0.01, le=10.0, description="감쇠 시간(초)")
    duration: Optional[float] = Field(None, gt=0, le=30, description="명시 길이(초)")
    # 모달 합성용 재질(공진 모드 비율을 결정)
    material: Optional[Literal["metal", "wood", "glass"]] = None
    # FM 합성용: 캐리어 대비 모듈레이터 비율 / 변조 강도
    ratio: float = Field(2.0, gt=0, le=24, description="FM 모듈레이터 비율")
    index: float = Field(5.0, ge=0, le=50, description="FM 변조 강도")


class Layer(BaseModel):
    source: Optional[LibrarySource] = None
    synth: Optional[SynthSource] = None
    transforms: list[TransformUnion] = Field(default_factory=list)
    start: float = Field(0.0, ge=0, description="믹스 내 시작 시점(초)")
    gain_db: float = Field(0.0, ge=-60, le=24)
    pan: float = Field(0.0, ge=-1.0, le=1.0, description="-1=L, 0=C, +1=R")

    @model_validator(mode="after")
    def _exactly_one_origin(self) -> "Layer":
        # 레이어는 라이브러리 음원 또는 합성 중 정확히 하나만 출처로 가져야 한다.
        if (self.source is None) == (self.synth is None):
            raise ValueError("레이어는 source 또는 synth 중 정확히 하나만 가져야 합니다.")
        return self


class OutputFormat(BaseModel):
    bit: Literal[16, 24, 32] = 24
    rate: int = Field(48000, ge=8000, le=192000)


class Master(BaseModel):
    loudness_lufs: Optional[float] = Field(-16.0, ge=-40, le=0)
    limiter: bool = True
    format: OutputFormat = Field(default_factory=OutputFormat)
    prefix: str = "[SFX]"


class Recipe(BaseModel):
    name: str = Field(..., min_length=1)
    seed: int = 0
    layers: list[Layer] = Field(..., min_length=1)
    master: Master = Field(default_factory=Master)

    @classmethod
    def from_json(cls, text: str) -> "Recipe":
        """LLM이 뱉은 JSON 문자열을 검증하며 로드(에러 시 명확한 메시지)."""
        return cls.model_validate_json(text)
