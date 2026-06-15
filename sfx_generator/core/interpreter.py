"""프롬프트 → 레시피 해석기 (룰 기반).

설계 의도(Why):
- 외부 모델/서비스 없이 앱 내부에서 즉시 동작한다(에어갭·설치 0). 룰이 항상 안전망.
- 한글/영문 키워드 매핑을 '데이터 테이블'로 상단에 분리했다. 현장 용어를 추가할 때
  코드 로직이 아니라 이 테이블만 손대면 된다(유지보수성).
- Korean 조사가 붙어도 잡히도록 토큰 일치가 아니라 '부분 문자열 포함'으로 매칭한다.
  (예: '묵직한', '저음의', '삐걱대는' 모두 매칭)
- 동일 프롬프트 → 동일 결과(seed를 프롬프트 해시로 고정)로 재현성을 보장한다.
- 향후 번들 LLM은 llm 콜러블로 주입되어 동일한 interpret() 인터페이스를 공유한다.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from typing import Optional

from .library import Library
from .presets import template_layers
from .recipe import Recipe

# ──────────────────────────────────────────────────────────────────────────
# 표 1) 소리 종류: 키워드 → 베이스 레이어 실현 방법
#   - synth: 절차적 합성으로 실현(추상/전자음). library_query 없이도 동작.
#   - library_query: 구체적 실세계 음(문/유리/발걸음 등)은 라이브러리에서 가져옴.
#   - 둘 다 있으면: 라이브러리 매칭 성공 시 라이브러리, 실패 시 synth로 폴백.
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class SoundIntent:
    name: str
    keywords: list[str]
    synth_kind: Optional[str] = None
    freq: float = 220.0
    decay: float = 1.0
    library_query: Optional[str] = None
    material: Optional[str] = None  # modal 합성용(metal/wood/glass)


SOUND_TYPES: list[SoundIntent] = [
    SoundIntent("impact", ["쿵", "쾅", "임팩트", "타격", "충격", "히트", "꽝", "hit", "impact", "boom"],
                synth_kind="sub_impact", freq=55, decay=1.2, library_query="impact hit"),
    SoundIntent("riser", ["상승", "고조", "빌드업", "라이저", "점점", "riser", "rise", "buildup"],
                synth_kind="riser", freq=200, decay=2.0, library_query="riser"),
    SoundIntent("whoosh", ["휙", "스윽", "슉", "슝", "후웅", "휘익", "휘리릭", "지나가", "스와이프", "전환", "whoosh", "swoosh", "swipe"],
                synth_kind="whoosh", freq=300, decay=1.0, library_query="whoosh swoosh"),
    SoundIntent("beep", ["삐", "비프", "알림", "띵", "신호", "버튼", "beep", "tone", "ui", "notification"],
                synth_kind="tone", freq=880, decay=0.4, library_query="beep ui"),
    SoundIntent("noise", ["노이즈", "지지직", "잡음", "화이트노이즈", "static", "noise"],
                synth_kind="noise", decay=1.0, library_query="noise static"),
    # 구체 사물군: 라이브러리 우선, 폴백 synth는 가장 근접한 합성으로
    SoundIntent("creak", ["삐걱", "끼익", "creak", "squeak"],
                synth_kind="noise", decay=1.5, library_query="creak door"),
    SoundIntent("glass", ["유리", "깨", "파편", "glass", "shatter", "break"],
                synth_kind="modal", material="glass", freq=900, decay=0.8, library_query="glass break shatter"),
    SoundIntent("door", ["문", "도어", "door"],
                synth_kind="noise", decay=1.0, library_query="door"),
    SoundIntent("footstep", ["발걸음", "발소리", "걷는", "footstep", "step"],
                synth_kind="noise", decay=0.5, library_query="footstep"),
    SoundIntent("water", ["물", "물방울", "물소리", "water", "drop", "splash"],
                synth_kind="noise", decay=0.6, library_query="water drop splash"),
    SoundIntent("bell", ["종소리", "벨", "차임", "bell", "chime"],
                synth_kind="tone", freq=660, decay=2.0, library_query="bell chime"),
    # ── 방송 현장: 예능 자막 효과음 ──
    SoundIntent("applause", ["박수", "짝짝", "갈채", "applause", "clap"],
                synth_kind="noise", decay=1.5, library_query="applause clap"),
    SoundIntent("cheer", ["환호", "함성", "응원", "cheer"],
                synth_kind="noise", decay=2.0, library_query="cheer crowd"),
    SoundIntent("laugh", ["웃음", "깔깔", "방청", "laugh"],
                synth_kind="noise", decay=1.5, library_query="laugh audience"),
    SoundIntent("drumroll", ["두구두구", "드럼롤", "drumroll"],
                synth_kind="noise", decay=1.5, library_query="drumroll"),
    SoundIntent("sparkle", ["반짝", "반짝반짝", "영롱", "뿅", "뾰로롱", "별빛", "sparkle", "twinkle"],
                synth_kind="tone", freq=1500, decay=0.9, library_query="sparkle twinkle"),
    SoundIntent("correct", ["딩동댕", "딩동", "정답", "띵동", "correct"],
                synth_kind="tone", freq=1046, decay=0.8, library_query="correct ding chime"),
    SoundIntent("wrong", ["땡", "오답", "부저", "buzzer", "wrong"],
                synth_kind="tone", freq=180, decay=0.6, library_query="buzzer wrong"),
    SoundIntent("transition", ["전환음", "트랜지션", "전환", "transition"],
                synth_kind="whoosh", decay=0.8, library_query="transition swoosh"),
    SoundIntent("explosion", ["폭발", "펑", "explosion", "blast"],
                synth_kind="sub_impact", freq=70, decay=1.5, library_query="explosion blast"),
    SoundIntent("censor", ["삐처리", "음소거", "검열", "censor"],
                synth_kind="tone", freq=1000, decay=0.6, library_query="censor beep"),
    # ── 방송 현장: 드라마/뉴스 ──
    SoundIntent("thunder", ["천둥", "우레", "thunder"],
                synth_kind="noise", decay=2.5, library_query="thunder"),
    SoundIntent("rain", ["빗소리", "장대비", "rain"],
                synth_kind="rain", decay=3.0, library_query="rain"),
    SoundIntent("wind", ["바람소리", "강풍", "wind"],
                synth_kind="wind", freq=600, decay=2.5, library_query="wind"),
    SoundIntent("fire", ["불꽃", "화염", "모닥불", "불타", "fire", "flame"],
                synth_kind="fire", decay=2.5, library_query="fire flame"),
    SoundIntent("pluck", ["튕김", "현", "기타", "하프", "퉁", "pluck", "string"],
                synth_kind="pluck", freq=330, decay=1.2, library_query="pluck string"),
    SoundIntent("laser", ["레이저", "광선", "빔", "전자음", "laser", "zap"],
                synth_kind="fm", freq=600, decay=0.6, library_query="laser zap"),
    SoundIntent("metal", ["쇳덩이", "징", "clang", "쇠북"],
                synth_kind="modal", material="metal", freq=320, decay=1.6, library_query="metal clang"),
    SoundIntent("phone", ["전화벨", "벨소리", "phone", "ring"],
                synth_kind="tone", freq=1000, decay=1.0, library_query="phone ring telephone"),
    SoundIntent("knock", ["노크", "똑똑", "knock"],
                synth_kind="noise", decay=0.4, library_query="knock door"),
    SoundIntent("clock", ["초침", "시계", "똑딱", "clock", "tick"],
                synth_kind="tone", freq=2000, decay=0.1, library_query="clock tick"),
    SoundIntent("typing", ["타이핑", "키보드", "typing", "keyboard"],
                synth_kind="noise", decay=0.1, library_query="typing keyboard"),
    SoundIntent("camera", ["셔터", "찰칵", "카메라", "camera", "shutter"],
                synth_kind="noise", decay=0.2, library_query="camera shutter"),
    # 자막/포인트 강조 효과음: 짧고 밝은 액센트(예능 자막에서 가장 흔함)
    SoundIntent("accent", ["자막", "포인트", "강조", "강조음", "스팅어", "sting", "accent"],
                synth_kind="tone", freq=1320, decay=0.45, library_query="accent sting stinger"),
    # 예능 자막 효과음(아이코닉 의성어)
    SoundIntent("dramatic", ["두둥", "반전등장", "충격등장", "dramatic"],
                synth_kind="sub_impact", freq=70, decay=1.8, library_query="dramatic boom sting"),
    SoundIntent("fanfare", ["짜잔", "빠밤", "빰빠밤", "팡파레", "fanfare", "tada"],
                synth_kind="fm", freq=523, decay=1.0, library_query="fanfare tada"),
    SoundIntent("boing", ["띠용", "뾰잉", "뽀잉", "스프링", "boing"],
                synth_kind="fm", freq=420, decay=0.5, library_query="boing spring"),
    SoundIntent("heartbeat", ["두근두근", "두근", "심장", "heartbeat"],
                synth_kind="sub_impact", freq=55, decay=0.5, library_query="heartbeat"),
    SoundIntent("crash", ["와장창", "우당탕", "박살", "crash"],
                synth_kind="modal", material="glass", freq=1000, decay=0.7, library_query="crash smash debris"),
    SoundIntent("gasp", ["헉", "헐", "숨막", "gasp"],
                synth_kind="whoosh", freq=500, decay=0.4, library_query="gasp inhale"),
    SoundIntent("awkward", ["정적", "썰렁", "싸늘", "귀뚜라미", "cricket"],
                synth_kind="tone", freq=2600, decay=1.5, library_query="cricket awkward silence"),
    # 일상/방송 자주 쓰는 소리(룰 폴백 보강)
    SoundIntent("drip", ["물방울", "방울", "물 떨어", "drip"],
                synth_kind="pluck", freq=900, decay=0.3, library_query="water drip drop"),
    SoundIntent("splash", ["첨벙", "물 튀", "물놀이", "splash"],
                synth_kind="noise", freq=400, decay=0.6, library_query="water splash"),
    SoundIntent("applause", ["박수", "손뼉", "갈채", "applause", "clap"],
                synth_kind="noise", freq=2000, decay=1.5, library_query="applause clap"),
    SoundIntent("bird", ["새소리", "지저귐", "새가", "bird", "tweet", "chirp"],
                synth_kind="tone", freq=3200, decay=0.6, library_query="bird chirp tweet"),
    SoundIntent("chime", ["종소리", "차임", "풍경소리", "chime"],
                synth_kind="modal", material="metal", freq=600, decay=2.2, library_query="bell chime"),
    SoundIntent("firework", ["폭죽", "불꽃놀이", "firework"],
                synth_kind="noise", freq=300, decay=1.4, library_query="firework"),
    SoundIntent("footsteps", ["발걸음", "발소리", "걸음소리", "footsteps", "footstep"],
                synth_kind="noise", freq=200, decay=0.4, library_query="footsteps"),
    SoundIntent("water", ["물소리", "시냇물", "흐르는 물", "water stream"],
                synth_kind="rain", freq=400, decay=3.0, library_query="water stream"),
]


# ──────────────────────────────────────────────────────────────────────────
# 표 2) 수식어: 키워드 → 효과(피치/게인/길이/변형 체인)
#   transforms 항목은 dsp가 이해하는 op dict 그대로다.
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class Modifier:
    keywords: list[str]
    pitch: float = 0.0          # 누적될 반음
    gain_db: float = 0.0        # 누적될 게인
    length: float = 1.0         # 곱해질 길이 배율
    transforms: list[dict] = field(default_factory=list)


MODIFIERS: list[Modifier] = [
    Modifier(["공포", "호러", "무서운", "소름", "오싹", "horror", "scary", "creepy"],
             pitch=-3, transforms=[{"op": "reverb", "kind": "hall", "wet": 0.35}]),
    Modifier(["어두운", "음산", "암울", "dark", "ominous"],
             pitch=-2, transforms=[{"op": "filter", "kind": "lowpass", "cutoff_hz": 4000}]),
    Modifier(["밝은", "경쾌", "맑은", "명랑", "bright", "cheerful"], pitch=3),
    Modifier(["묵직", "저음", "낮은", "굵은", "deep", "heavy", "low"],
             pitch=-2, transforms=[{"op": "filter", "kind": "lowpass", "cutoff_hz": 1200}]),
    Modifier(["날카로운", "얇은", "높은", "예리", "sharp", "thin", "high"],
             transforms=[{"op": "filter", "kind": "highpass", "cutoff_hz": 800},
                         {"op": "eq", "freq_hz": 4000, "gain_db": 4, "q": 1.0}]),
    Modifier(["금속", "메탈릭", "쇳소리", "metallic", "metal"],
             transforms=[{"op": "eq", "freq_hz": 3500, "gain_db": 6, "q": 2.0},
                         {"op": "distortion", "drive_db": 6}]),
    Modifier(["공간감", "울림", "잔향", "홀", "에코", "reverb", "hall", "echo"],
             transforms=[{"op": "reverb", "kind": "hall", "wet": 0.4}]),
    Modifier(["좁은", "룸", "실내", "room"],
             transforms=[{"op": "reverb", "kind": "room", "wet": 0.25}]),
    Modifier(["멀리", "작게", "희미", "약하게", "은은", "far", "distant", "faint"],
             gain_db=-8, transforms=[{"op": "filter", "kind": "lowpass", "cutoff_hz": 3000}]),
    Modifier(["가까이", "크게", "강하게", "세게", "강한", "loud", "close", "strong"], gain_db=4),
    Modifier(["매우", "아주", "엄청", "극도", "very", "super"], gain_db=2),
    Modifier(["살짝", "약간", "조금", "slight", "subtle"], gain_db=-3),
    Modifier(["디지털", "전자", "사이버", "글리치", "digital", "glitch", "cyber"],
             transforms=[{"op": "bitcrush", "bit_depth": 8}]),
    Modifier(["빈티지", "레트로", "8비트", "8bit", "retro", "vintage"],
             transforms=[{"op": "bitcrush", "bit_depth": 6}]),
    Modifier(["거친", "왜곡", "디스토션", "distorted", "gritty"],
             transforms=[{"op": "distortion", "drive_db": 12}]),
    Modifier(["떨리는", "코러스", "풍성", "chorus", "shimmer"],
             transforms=[{"op": "chorus", "rate_hz": 1.5, "depth": 0.3, "mix": 0.5}]),
    Modifier(["딜레이", "메아리", "반복", "delay"],
             transforms=[{"op": "delay", "seconds": 0.25, "feedback": 0.35, "mix": 0.4}]),
    Modifier(["길게", "길고", "늘여", "long"], length=1.6),
    Modifier(["짧은", "짧게", "타이트", "short", "tight"], length=0.6),
    # ── 방송 무드(예능/드라마 연출) ──
    Modifier(["코믹", "예능", "만화", "장난", "comic", "cartoon"],
             pitch=2, transforms=[{"op": "chorus", "rate_hz": 2.0, "depth": 0.3, "mix": 0.4}]),
    Modifier(["서스펜스", "긴장감", "두근", "suspense", "tense"],
             pitch=-1, transforms=[{"op": "reverb", "kind": "hall", "wet": 0.3},
                                   {"op": "filter", "kind": "lowpass", "cutoff_hz": 6000}]),
    Modifier(["웅장", "장엄", "장대", "epic", "grand"],
             transforms=[{"op": "reverb", "kind": "hall", "wet": 0.45},
                         {"op": "eq", "freq_hz": 90, "gain_db": 4, "q": 1.0}]),
    Modifier(["감성", "잔잔", "애잔", "슬픈", "emotional", "sad"],
             transforms=[{"op": "reverb", "kind": "hall", "wet": 0.35},
                         {"op": "filter", "kind": "lowpass", "cutoff_hz": 6000}]),
    Modifier(["신비", "몽환", "환상", "mystical", "dreamy"],
             transforms=[{"op": "reverb", "kind": "hall", "wet": 0.4},
                         {"op": "delay", "seconds": 0.3, "feedback": 0.3, "mix": 0.3}]),
    Modifier(["따뜻", "포근", "warm"],
             transforms=[{"op": "eq", "freq_hz": 200, "gain_db": 3, "q": 1.0},
                         {"op": "filter", "kind": "lowpass", "cutoff_hz": 8000}]),
    Modifier(["차가운", "서늘", "냉랭", "cold"],
             transforms=[{"op": "filter", "kind": "highpass", "cutoff_hz": 400},
                         {"op": "eq", "freq_hz": 5000, "gain_db": 3, "q": 1.0}]),
    Modifier(["귀여운", "깜찍", "앙증", "cute"],
             pitch=4, transforms=[{"op": "chorus", "rate_hz": 2.5, "depth": 0.3, "mix": 0.4}]),
    Modifier(["반전", "충격적", "극적", "dramatic"],
             pitch=-1, transforms=[{"op": "reverb", "kind": "hall", "wet": 0.4}]),
]

# 변형 체인 정렬 순서(소닉하게 합리적인 순서로 고정 → 결과 안정성)
_CANON_ORDER = ["pitch", "stretch", "filter", "eq", "distortion", "bitcrush",
                "chorus", "delay", "reverb", "fade"]


@dataclass
class Interpretation:
    """해석 결과 + 근거(GUI에서 '왜 이렇게 만들었는지' 표시용)."""

    recipe: Recipe
    matched_types: list[str]
    matched_modifiers: list[str]
    source: str = "rules"  # "rules" | "llm"


class Interpreter:
    def __init__(self, library: Optional[Library] = None, llm=None, expand: bool = False,
                 length_sec: Optional[float] = None) -> None:
        # library: 라이브러리 매칭 시 사용.
        # llm: (system,user)->str 콜러블(예: Ollama). 주입 시 LLM 우선, 실패하면 룰로 폴백.
        # expand: True면 LLM이 모호한 프롬프트를 상세 브리프로 먼저 확장한 뒤 레시피화.
        # length_sec: 지정 시 단일 합성 베이스 길이를 그 초로 채운다(최종 길이 맞춤은 렌더에서).
        self.library = library
        self.llm = llm
        self.expand = expand
        self.length_sec = length_sec

    # ── 매칭 ───────────────────────────────────────────────────────────────
    def _match_types(self, text: str) -> list[SoundIntent]:
        """소리 종류 매칭. 한글 부분일치 오탐('삐'⊂'삐걱')을 두 단계로 거른다.

        Why: ① 어떤 종류의 매칭 키워드가 '다른 종류의 더 긴 매칭 키워드'에 통째로
        포함되면 오탐(팬텀)으로 보고 제거한다. ② 남은 종류는 매칭된 키워드의 길이가
        길수록(=구체적일수록) 앞에 오도록 정렬해, primary가 가장 구체적인 소리가 되게 한다.
        """
        matches: list[tuple[SoundIntent, list[str]]] = []
        for t in SOUND_TYPES:
            kws = [k for k in t.keywords if k in text]
            if kws:
                matches.append((t, kws))
        if not matches:
            return []

        all_kws = [k for _, kws in matches for k in kws]

        def is_phantom(kws: list[str]) -> bool:
            # 모든 매칭 키워드가 '자신보다 긴 다른 매칭 키워드'에 포함되면 팬텀
            return all(any(k != o and k in o for o in all_kws) for k in kws)

        filtered = [(t, kws) for t, kws in matches if not is_phantom(kws)] or matches
        filtered.sort(key=lambda tk: max(len(k) for k in tk[1]), reverse=True)
        return [t for t, _ in filtered]

    def _match_modifiers(self, text: str) -> list[Modifier]:
        return [m for m in MODIFIERS if any(k in text for k in m.keywords)]

    # ── 베이스 레이어 실현(라이브러리 우선, 실패 시 synth 폴백) ────────────────
    def _realize_base(self, intent: SoundIntent, length: float) -> dict:
        if intent.library_query and self.library is not None:
            grounded = self._ground_query(intent.library_query)
            if grounded is not None:
                layer = {"source": {"query": grounded, "pick": "best"}, "transforms": []}
                if abs(length - 1.0) > 1e-3:
                    layer["transforms"].append({"op": "stretch", "rate": length})
                return layer
        # synth 폴백(또는 애초에 synth형): 길이는 decay에 직접 반영
        kind = intent.synth_kind or "noise"
        synth_spec = {"kind": kind, "freq": intent.freq, "decay": max(0.05, intent.decay * length)}
        if intent.material:
            synth_spec["material"] = intent.material
        return {"synth": synth_spec, "transforms": []}

    def _ground_query(self, query: str) -> Optional[str]:
        """검색어를 실제 라이브러리에 존재하는 태그로 보정. 매칭 없으면 None."""
        if self.library is None:
            return None
        if not self.library.samples:
            self.library.scan()
        # 검색어 토큰 중 라이브러리 태그에 실제 존재하는 것만 남긴다.
        tags = self.library.all_tags()
        kept = [w for w in query.split() if w in tags]
        if not kept:
            # 부분일치라도 있으면 원 쿼리 유지(resolver의 부분일치에 맡김)
            return query if self.library.search(query) is not None else None
        return " ".join(kept)

    # ── 수식어 누적 → 변형 체인 조립 ──────────────────────────────────────────
    @staticmethod
    def _assemble_transforms(mods: list[Modifier]) -> tuple[list[dict], float, float]:
        total_pitch = sum(m.pitch for m in mods)
        total_gain = sum(m.gain_db for m in mods)
        length = 1.0
        for m in mods:
            length *= m.length

        extras: list[dict] = []
        seen_ops: set[str] = set()
        for m in mods:
            for tf in m.transforms:
                op = tf["op"]
                # eq는 밴드별로 여러 개 허용, 나머지는 op당 하나(첫 매칭 우선)
                if op != "eq" and op in seen_ops:
                    continue
                seen_ops.add(op)
                extras.append(tf)

        chain: list[dict] = []
        if abs(total_pitch) > 1e-3:
            chain.append({"op": "pitch", "semitones": max(-24, min(24, total_pitch))})
        chain.extend(extras)
        chain.sort(key=lambda t: _CANON_ORDER.index(t["op"]) if t["op"] in _CANON_ORDER else 99)
        return chain, total_gain, length

    # ── 메인 진입점 ──────────────────────────────────────────────────────────
    def interpret(self, prompt: str) -> Interpretation:
        """LLM이 주입돼 있으면 LLM 우선, 실패 시 룰 엔진으로 자동 폴백."""
        if not prompt.strip():
            raise ValueError("프롬프트가 비어 있습니다.")
        if self.llm is not None:
            try:
                return self._interpret_llm(prompt)
            except Exception as e:  # noqa: BLE001 - LLM 실패는 치명적이지 않다(룰로 폴백)
                print(f"[interpreter] LLM 해석 실패 → 룰 폴백: {e}")
        return self._interpret_rules(prompt)

    def _interpret_rules(self, prompt: str) -> Interpretation:
        text = prompt.strip().lower()
        types = self._match_types(text)
        mods = self._match_modifiers(text)
        chain, gain_db, length = self._assemble_transforms(mods)

        # 매칭된 소리 종류가 없으면: 거친 노이즈로 떨어지지 않게 '부드러운 톤'을 기본으로.
        # 무드(밝음/어두움)에 따라 음높이를 잡아 의도에 가깝게 만든다.
        if not types:
            bright = any(k in text for k in ("밝", "경쾌", "맑", "높", "bright", "포인트", "자막"))
            dark = any(k in text for k in ("어두", "저음", "낮", "묵직", "공포", "무거", "dark"))
            f = 1320.0 if bright else 174.0 if dark else 523.0
            types = [SoundIntent("default", [], synth_kind="tone", freq=f, decay=0.6)]

        # 복합 연출: 라이저가 있으면 라이저를 베이스(spine)로 삼는다. SOUND_TYPES
        # 정렬 순서와 무관하게 '빌드업→타격' 구조가 자연스럽게 나오도록 하기 위함.
        riser_intent = next((t for t in types if t.name == "riser"), None)
        impact_intent = next((t for t in types if t.name == "impact"), None)
        primary = riser_intent if riser_intent is not None else types[0]

        layers: list[dict] = []
        # 내부 템플릿(알고리즘): 특정 큐가 primary면 다층 레시피를 자동 사용해
        # 단일 합성보다 훨씬 그럴듯한 결과를 낸다(UI에 프리셋 목록을 노출하지 않음).
        use_combo = riser_intent is not None and impact_intent is not None
        tmpl = None if use_combo else template_layers(primary.name)
        if tmpl is not None:
            layers = tmpl
            first = layers[0]
            first["transforms"] = first.get("transforms", []) + chain  # 무드 반영
            first["gain_db"] = max(-60, min(24, float(first.get("gain_db", 0.0)) + gain_db))
        else:
            base = self._realize_base(primary, length)
            # 공통 수식어 체인을 베이스에 합치고(베이스 자체 stretch 뒤에) 게인 적용
            base["transforms"] = base.get("transforms", []) + chain
            base["start"] = 0.0
            base["gain_db"] = max(-60, min(24, gain_db))
            base["pan"] = 0.0
            layers.append(base)

            # 라이저 + 임팩트가 함께 언급되면 임팩트를 끝에 깔아 빌드업→타격 연출
            if use_combo:
                impact = self._realize_base(impact_intent, 1.0)
                impact["transforms"] = impact.get("transforms", [])
                impact["start"] = max(0.2, primary.decay * length - 0.2)
                impact["gain_db"] = 0.0
                impact["pan"] = 0.0
                layers.append(impact)

        # 명시적 길이(초) 지정 시: 단일 합성 베이스의 duration을 채워 자연스럽게 채운다.
        if self.length_sec is not None and len(layers) == 1 and "synth" in layers[0]:
            layers[0]["synth"]["duration"] = self.length_sec

        name = self._make_name(prompt)
        seed = zlib.crc32(prompt.encode("utf-8")) & 0x7FFFFFFF
        recipe = Recipe.model_validate(
            {"name": name, "seed": seed, "layers": layers,
             "master": {"loudness_lufs": -16, "limiter": True,
                        "format": {"bit": 24, "rate": 48000}, "prefix": "[SFX]"}}
        )
        return Interpretation(
            recipe=recipe,
            matched_types=[t.name for t in types],
            matched_modifiers=[m.keywords[0] for m in mods],
            source="rules",
        )

    # ── LLM 경로 ───────────────────────────────────────────────────────────
    def _interpret_llm(self, prompt: str) -> Interpretation:
        user = prompt
        if self.expand:
            user = self._expand_prompt(prompt)  # 모호한 한 줄 → 상세 브리프
        system = self._build_system_prompt()
        raw = self.llm(system, user)
        try:
            recipe = self._parse_recipe(raw, prompt)
        except Exception:
            # 1회 교정 재시도: 오류를 알려주고 JSON만 다시 요청
            raw2 = self.llm(system, user + "\n\n(이전 출력이 스키마에 맞지 않았습니다. 유효한 JSON만 출력하세요.)")
            recipe = self._parse_recipe(raw2, prompt)

        # 표시용 근거: 결과 레이어에서 종류 추출
        kinds = []
        for ly in recipe.layers:
            if ly.synth is not None:
                kinds.append(ly.synth.kind)
            elif ly.source is not None:
                kinds.append(f"lib:{ly.source.query}")
        return Interpretation(recipe=recipe, matched_types=kinds, matched_modifiers=[], source="llm")

    def _expand_prompt(self, prompt: str) -> str:
        """LLM으로 짧은 요청을 구체적 사운드 디자인 브리프로 확장(실패 시 원문 유지).

        complete()가 JSON을 강제하므로 {"brief": "..."} 형태로 받아 파싱한다.
        """
        sys = ("너는 방송 사운드 디자이너다. 사용자의 짧은 효과음 요청을 구체적인 "
               "사운드 디자인 브리프로 한국어 2~3문장으로 확장하라. 재질·동작·길이·공간·무드·레이어를 "
               "포함하라. 반드시 {\"brief\": \"...\"} JSON 한 개만 출력한다.")
        try:
            import json as _json

            raw = self.llm(sys, prompt)
            s, e = raw.find("{"), raw.rfind("}")
            brief = _json.loads(raw[s : e + 1]).get("brief", "").strip()
            return f"{prompt}\n\n[상세 브리프] {brief}" if brief else prompt
        except Exception:  # noqa: BLE001 - 확장 실패는 무시하고 원문 사용
            return prompt

    def _parse_recipe(self, raw: str, prompt: str) -> Recipe:
        """LLM 출력에서 JSON을 추출·검증해 Recipe로 만든다."""
        text = raw.strip()
        # 혹시 모를 코드펜스 제거
        if "```" in text:
            text = text.split("```")[1] if text.count("```") >= 2 else text
            text = text.replace("json", "", 1).strip() if text.lstrip().lower().startswith("json") else text
        # 첫 '{' ~ 마지막 '}' 구간만 취함
        s, e = text.find("{"), text.rfind("}")
        if s == -1 or e == -1:
            raise ValueError("JSON 객체를 찾지 못했습니다.")
        import json as _json

        data = _json.loads(text[s : e + 1])
        # 필수 필드 보강(모델이 빠뜨려도 검증 통과하도록)
        data.setdefault("name", self._make_name(prompt))
        data.setdefault("seed", zlib.crc32(prompt.encode("utf-8")) & 0x7FFFFFFF)
        data.setdefault("master", {"loudness_lufs": -16, "limiter": True,
                                   "format": {"bit": 24, "rate": 48000}, "prefix": "[SFX]"})
        return Recipe.model_validate(data)

    def _build_system_prompt(self) -> str:
        """레시피 스키마 + 사용 가능한 합성/변형 + (있으면) 라이브러리 태그를 안내."""
        synth_kinds = ("sub_impact, tone, noise, whoosh, riser, modal(material=metal|wood|glass), "
                       "pluck, fm(ratio,index), wind, rain, fire")
        ops = ("pitch(semitones), stretch(rate), reverse, gain(db), fade(in,out), "
               "filter(kind=highpass|lowpass, cutoff_hz), eq(freq_hz,gain_db,q), "
               "reverb(kind=room|hall|plate, wet), distortion(drive_db), "
               "chorus(rate_hz,depth,mix), delay(seconds,feedback,mix), bitcrush(bit_depth), normalize(peak_db), "
               "granular(grain_ms,density,pitch_jitter,stretch,spray_ms), spectral(mode=freeze|stretch|blur, amount)")
        tags_line = ""
        if self.library is not None:
            if not self.library.samples:
                try:
                    self.library.scan()
                except Exception:  # noqa: BLE001
                    pass
            tags = sorted(self.library.all_tags())[:120]
            if tags:
                tags_line = f"\n사용 가능한 라이브러리 태그(이 중에서 source.query를 고를 것): {', '.join(tags)}"
        master = '{"loudness_lufs":-16,"limiter":true,"format":{"bit":24,"rate":48000},"prefix":"[SFX]"}'
        # few-shot: 한국어 설명 → 레시피. 모델이 음색/레이어/변형 사용법을 학습하도록.
        examples = "\n\n".join([
            "입력: 유리잔 깨지는 소리\n출력: "
            '{"name":"glass","seed":1,"layers":['
            '{"synth":{"kind":"noise","decay":0.12},"transforms":[{"op":"filter","kind":"highpass","cutoff_hz":4000},{"op":"fade","out":0.1}],"gain_db":-4},'
            '{"synth":{"kind":"modal","material":"glass","freq":2600,"decay":0.28},"start":0.01},'
            '{"synth":{"kind":"modal","material":"glass","freq":3900,"decay":0.2},"start":0.04,"gain_db":-5}],'
            f'"master":{master}}}',
            "입력: 낮고 묵직한 폭발 임팩트\n출력: "
            '{"name":"boom","seed":1,"layers":['
            '{"synth":{"kind":"sub_impact","freq":50,"decay":1.4},"transforms":[{"op":"distortion","drive_db":5},{"op":"reverb","kind":"hall","wet":0.3}]},'
            '{"synth":{"kind":"noise","decay":0.3},"transforms":[{"op":"filter","kind":"lowpass","cutoff_hz":1200}],"gain_db":-8}],'
            f'"master":{master}}}',
            "입력: 밝은 알림음 띵\n출력: "
            '{"name":"ding","seed":1,"layers":[{"synth":{"kind":"fm","freq":1320,"decay":0.5,"ratio":2.0,"index":3},"transforms":[{"op":"eq","freq_hz":4000,"gain_db":3,"q":1.0}]}],'
            f'"master":{master}}}',
            "입력: 3초 동안 부는 바람\n출력: "
            '{"name":"wind","seed":1,"layers":[{"synth":{"kind":"wind","freq":600,"decay":3.0},"transforms":[{"op":"fade","in":0.4,"out":0.8}]}],'
            f'"master":{master}}}',
            "입력: 레이저 발사\n출력: "
            '{"name":"laser","seed":1,"layers":[{"synth":{"kind":"fm","freq":1100,"decay":0.4,"ratio":3.0,"index":7},"transforms":[{"op":"pitch","semitones":-9},{"op":"delay","seconds":0.1,"feedback":0.25,"mix":0.25}]}],'
            f'"master":{master}}}',
        ])
        return (
            "너는 방송 효과음 사운드 디자이너다. 사용자의 한국어 프롬프트를 받아 효과음 '레시피' JSON만 출력한다.\n"
            "절대 설명/마크다운 없이 JSON 객체 하나만 출력한다.\n\n"
            "레시피 구조: {name, seed, layers[], master}. 각 layer는 source 또는 synth 중 하나만 가진다.\n"
            f"- synth.kind ∈ {{{synth_kinds}}}, freq, decay\n"
            "- source: {query, pick(best|random)} — 보유 라이브러리에서 음원 검색\n"
            f"- transforms[]: 순서 있는 변형 체인. 사용 가능한 op: {ops}\n"
            "- layer: start(초), gain_db(-60~24), pan(-1~1)\n"
            "- 여러 소리가 겹친 효과음은 layer를 여러 개 쓴다(예: 충돌=노이즈 크랙+공진 파편).\n"
            "- 길이는 synth.decay로, 공간감은 reverb로, 밝기는 eq/filter로, 무게는 sub_impact로 표현한다.\n"
            f"- master는 항상: {master}\n"
            f"{tags_line}\n\n"
            f"예시:\n{examples}"
        )

    @staticmethod
    def _make_name(prompt: str) -> str:
        # 프롬프트 앞부분을 파일명 친화적으로 변환(최대 30자)
        clean = "_".join(prompt.strip().split())[:30]
        return clean or "sfx"
