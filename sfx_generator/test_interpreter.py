"""interpreter.py 통합 테스트."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from core.interpreter import Interpreter
from core.library import Library
from core.render import render_recipe

PASS, FAIL = "✅", "❌"
_results = []


def check(cond, msg):
    _results.append(bool(cond))
    print(f"{PASS if cond else FAIL} {msg}")


def ops_of(layer):
    return [t.op for t in layer.transforms]


def main() -> int:
    itp = Interpreter()

    # 1) 소리 종류 감지
    r = itp.interpret("묵직한 저음 쿵 임팩트").recipe
    check(r.layers[0].synth and r.layers[0].synth.kind == "sub_impact", "감지: 쿵 → sub_impact")

    r = itp.interpret("경쾌한 UI 알림음 삐").recipe
    check(r.layers[0].synth and r.layers[0].synth.kind == "tone", "감지: 알림음 → tone")

    # 2) 수식어 → 변형
    res = itp.interpret("공포스러운 어두운 임팩트")
    ops = ops_of(res.recipe.layers[0])
    check("pitch" in ops and "reverb" in ops, "공포 → pitch down + reverb")
    check("공포" in res.matched_modifiers, "근거(matched_modifiers)에 '공포' 포함")

    r = itp.interpret("디지털 글리치 비프").recipe
    check("bitcrush" in ops_of(r.layers[0]), "디지털 → bitcrush")

    r = itp.interpret("멀리서 들리는 희미한 종소리").recipe
    check(r.layers[0].gain_db < 0, "멀리/희미 → 게인 감소")

    # 3) 피치 누적 + 범위 클램프
    r = itp.interpret("저음 어두운 공포 묵직한 쿵").recipe  # pitch들이 합쳐져도 -24 이하로 안 내려가야
    pitch_tf = [t for t in r.layers[0].transforms if t.op == "pitch"]
    check(pitch_tf and pitch_tf[0].semitones >= -24, "피치 누적 후 범위 클램프")

    # 4) 라이저 + 임팩트 → 2개 레이어, 임팩트가 뒤에 배치
    res = itp.interpret("점점 고조되는 긴장감 라이저 끝에 쿵")
    rec = res.recipe
    check(len(rec.layers) == 2, "라이저+임팩트 → 2개 레이어")
    check(rec.layers[1].start > rec.layers[0].start, "임팩트가 라이저 뒤에 배치")

    # 5) 재현성: 동일 프롬프트 → 동일 seed/이름
    a = itp.interpret("바람 휙 지나가는 소리").recipe
    b = itp.interpret("바람 휙 지나가는 소리").recipe
    check(a.seed == b.seed and a.name == b.name, "동일 프롬프트 → 재현성(seed/name)")

    # 6) 빈 프롬프트 거부
    try:
        itp.interpret("   ")
        check(False, "빈 프롬프트 거부")
    except ValueError:
        check(True, "빈 프롬프트 거부")

    with tempfile.TemporaryDirectory() as td:
        root, out = Path(td) / "lib", Path(td) / "out"
        root.mkdir()
        out.mkdir()
        # 라이브러리에 creak 음원 배치 → 그라운딩되어 source 레이어가 나와야 함
        (root / "foley").mkdir()
        rng = np.random.default_rng(0)
        sf.write(str(root / "foley" / "wood_door_creak.wav"),
                 (rng.standard_normal(48000) * 0.2).astype(np.float32), 48000)
        lib = Library(root)
        lib.scan()
        itp2 = Interpreter(library=lib)

        # 7) 라이브러리 그라운딩: creak 매칭 → source 레이어
        rec = itp2.interpret("삐걱대는 문 소리").recipe
        check(rec.layers[0].source is not None, "라이브러리 그라운딩 → source 레이어")

        # 8) 라이브러리에 없는 종류는 synth 폴백
        rec2 = itp2.interpret("저음 쿵 임팩트").recipe
        check(rec2.layers[0].synth is not None, "미보유 종류 → synth 폴백")

        # 9) 프롬프트 → 실제 렌더(엔드투엔드)
        res = itp2.interpret("공포 영화 문 삐걱 소리에 저음 쿵")
        path = render_recipe(res.recipe, out, resolver=lib.as_resolver())
        info = sf.info(str(path))
        check(path.exists() and info.samplerate == 48000 and info.channels == 2,
              f"프롬프트→WAV 엔드투엔드: {path.name}")

    passed = sum(_results)
    print(f"\n{'='*50}\n결과: {passed}/{len(_results)} 통과")
    return 0 if passed == len(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
