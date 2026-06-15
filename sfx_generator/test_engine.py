"""엔진 통합 테스트. 외부 의존 없이 자체 생성한 샘플로 전 경로를 검증한다."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from core import dsp, synth
from core.recipe import Layer, Recipe
from core.render import render_recipe, render_variations

PASS, FAIL = "✅", "❌"
_results: list[tuple[bool, str]] = []


def check(cond: bool, msg: str) -> None:
    _results.append((bool(cond), msg))
    print(f"{PASS if cond else FAIL} {msg}")


# ── 1. 스키마 검증 ────────────────────────────────────────────────────────
def test_schema() -> None:
    # 정상 레시피
    ok = Recipe.from_json(
        '{"name":"t","layers":[{"synth":{"kind":"tone","freq":440,"decay":0.5}}]}'
    )
    check(ok.master.format.bit == 24 and ok.master.format.rate == 48000, "기본 포맷 24bit/48kHz")

    # source/synth 동시 지정 → 거부되어야 함
    try:
        Layer(source={"query": "x"}, synth={"kind": "tone"})
        check(False, "source+synth 동시 지정 거부")
    except Exception:
        check(True, "source+synth 동시 지정 거부")

    # 범위 밖 피치 → 거부
    try:
        Recipe.from_json(
            '{"name":"t","layers":[{"synth":{"kind":"tone"},"transforms":[{"op":"pitch","semitones":99}]}]}'
        )
        check(False, "범위 밖 파라미터 거부")
    except Exception:
        check(True, "범위 밖 파라미터 거부")


# ── 2. 모든 변형 op 개별 동작 ───────────────────────────────────────────────
def test_all_transforms() -> None:
    sr = 48000
    rng = np.random.default_rng(0)
    base = synth.tone(sr, 440, 1.0, rng)
    ops = [
        {"op": "pitch", "semitones": -3},
        {"op": "stretch", "rate": 1.4},
        {"op": "reverse"},
        {"op": "gain", "db": -6},
        {"op": "fade", "in": 0.05, "out": 0.2},
        {"op": "filter", "kind": "lowpass", "cutoff_hz": 2000},
        {"op": "eq", "freq_hz": 1000, "gain_db": 6, "q": 1.0},
        {"op": "reverb", "kind": "hall", "wet": 0.4},
        {"op": "distortion", "drive_db": 12},
        {"op": "chorus", "rate_hz": 1.5, "depth": 0.3, "mix": 0.5},
        {"op": "delay", "seconds": 0.25, "feedback": 0.3, "mix": 0.4},
        {"op": "bitcrush", "bit_depth": 8},
        {"op": "normalize", "peak_db": -1.0},
    ]
    from core.recipe import Layer as L

    layer = L(synth={"kind": "tone", "freq": 440, "decay": 1.0}, transforms=ops)
    out = dsp.apply_chain(base, sr, layer.transforms)
    finite = np.all(np.isfinite(out))
    inrange = float(np.max(np.abs(out))) <= 1.5  # 정규화 전이라 약간 여유
    check(out.size > 0 and finite, "전체 변형 체인 NaN/Inf 없음")
    check(inrange, f"전체 변형 체인 출력 범위 정상(peak={np.max(np.abs(out)):.2f})")


# ── 3. 합성 전용 레시피(공포 문 삐걱 + 저음 쿵, creak도 합성으로 대체) ──────────
def test_synth_recipe(out_dir: Path) -> None:
    recipe = Recipe.from_json(
        """
    {
      "name": "horror_riser_impact", "seed": 42,
      "layers": [
        {"synth": {"kind": "riser", "freq": 200, "decay": 2.0},
         "transforms": [{"op": "filter", "kind": "highpass", "cutoff_hz": 150},
                        {"op": "reverb", "kind": "hall", "wet": 0.35}],
         "start": 0.0, "gain_db": -3, "pan": -0.2},
        {"synth": {"kind": "sub_impact", "freq": 55, "decay": 1.2},
         "transforms": [{"op": "distortion", "drive_db": 6}],
         "start": 1.8, "gain_db": -2, "pan": 0.0}
      ],
      "master": {"loudness_lufs": -16, "limiter": true,
                 "format": {"bit": 24, "rate": 48000}, "prefix": "[SFX]"}
    }
    """
    )
    path = render_recipe(recipe, out_dir)
    info = sf.info(str(path))
    check(path.exists() and path.name.startswith("[SFX]"), f"합성 레시피 출력 생성: {path.name}")
    check(info.samplerate == 48000 and info.channels == 2, "출력 48kHz 스테레오")
    check("24" in info.subtype, f"출력 24bit ({info.subtype})")


# ── 4. 라이브러리 음원 레시피(가짜 샘플 + 단순 resolver) ──────────────────────
def test_library_recipe(out_dir: Path, lib_dir: Path) -> None:
    # 44.1kHz 가짜 'creak' 샘플 생성 → 리샘플링 경로까지 검증
    rng = np.random.default_rng(1)
    creak = (rng.standard_normal(44100) * np.linspace(1, 0, 44100)).astype(np.float32) * 0.3
    sf.write(str(lib_dir / "wood_creak_01.wav"), creak, 44100, subtype="PCM_16")

    def resolver(query: str, pick: str):
        # 단순 파일명 부분일치 매칭(향후 library.py가 의미검색으로 대체)
        words = query.lower().split()
        for p in sorted(lib_dir.glob("*.wav")):
            if any(w in p.stem.lower() for w in words):
                return p
        return None

    recipe = Recipe.from_json(
        """
    {
      "name": "door_creak_fx", "seed": 7,
      "layers": [
        {"source": {"query": "creak", "pick": "best"},
         "transforms": [{"op": "pitch", "semitones": -3}, {"op": "stretch", "rate": 1.4},
                        {"op": "reverb", "kind": "room", "wet": 0.3}],
         "start": 0.0, "gain_db": 0, "pan": -0.1}
      ],
      "master": {"format": {"bit": 24, "rate": 48000}, "prefix": "[SFX]"}
    }
    """
    )
    path = render_recipe(recipe, out_dir, resolver=resolver)
    info = sf.info(str(path))
    check(path.exists(), f"라이브러리 레시피 출력 생성: {path.name}")
    check(info.samplerate == 48000, "44.1k→48k 리샘플링 정상")


# ── 5. 변주 일괄 생성 ──────────────────────────────────────────────────────
def test_variations(out_dir: Path) -> None:
    recipe = Recipe.from_json(
        '{"name":"ui_beep","seed":100,"layers":[{"synth":{"kind":"tone","freq":880,"decay":0.4}}]}'
    )
    paths = render_variations(recipe, 3, out_dir)
    check(len(paths) == 3 and all(p.exists() for p in paths), "변주 3개 생성")
    # 서로 다른 seed → 내용이 달라야 함(여기선 tone이라 동일 가능 → 길이/존재만 확인)
    names = {p.name for p in paths}
    check(len(names) == 3, "변주 파일명 고유")


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td) / "out"
        lib_dir = Path(td) / "lib"
        out_dir.mkdir()
        lib_dir.mkdir()
        test_schema()
        test_all_transforms()
        test_synth_recipe(out_dir)
        test_library_recipe(out_dir, lib_dir)
        test_variations(out_dir)

    passed = sum(1 for ok, _ in _results if ok)
    total = len(_results)
    print(f"\n{'='*50}\n결과: {passed}/{total} 통과")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
