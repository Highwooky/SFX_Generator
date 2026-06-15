"""LLM 해석 경로 + 폴백 테스트. 실제 Ollama 없이 mock 콜러블로 검증한다."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from core.interpreter import Interpreter
from core.library import Library
from core.llm import _auto_pick
from core.render import render_recipe

PASS, FAIL = "✅", "❌"
_results = []


def check(cond, msg):
    _results.append(bool(cond))
    print(f"{PASS if cond else FAIL} {msg}")


VALID = (
    '{"name":"x","seed":1,"layers":[{"synth":{"kind":"whoosh","freq":300,"decay":1.0},'
    '"transforms":[{"op":"reverb","kind":"hall","wet":0.4}]}],'
    '"master":{"format":{"bit":24,"rate":48000},"prefix":"[SFX]"}}'
)


def good_llm(system, user):
    return VALID


def fenced_llm(system, user):
    return "```json\n" + VALID + "\n```"


def garbage_llm(system, user):
    return "죄송하지만 도와드릴 수 없습니다."  # JSON 없음 → 파싱 실패


def raising_llm(system, user):
    raise RuntimeError("connection refused")


def minimal_llm(system, user):
    # name/seed/master 누락 → 보강 로직으로 통과해야 함
    return '{"layers":[{"synth":{"kind":"tone","freq":440,"decay":0.5}}]}'


def main() -> int:
    # 1) 정상 JSON → LLM 경로
    res = Interpreter(llm=good_llm).interpret("아무 프롬프트")
    check(res.source == "llm" and res.recipe.layers[0].synth.kind == "whoosh", "정상 JSON → LLM 경로 사용")

    # 2) 코드펜스 감싼 출력도 파싱
    res = Interpreter(llm=fenced_llm).interpret("프롬프트")
    check(res.source == "llm", "```json 펜스 제거 후 파싱")

    # 3) 쓰레기 출력 → 룰로 폴백
    res = Interpreter(llm=garbage_llm).interpret("공포 임팩트 쿵")
    check(res.source == "rules" and res.recipe is not None, "파싱 실패 → 룰 폴백")

    # 4) LLM 예외 → 룰로 폴백
    res = Interpreter(llm=raising_llm).interpret("밝은 알림음")
    check(res.source == "rules", "LLM 예외 → 룰 폴백")

    # 5) 필수 필드 누락 → 보강 후 통과
    res = Interpreter(llm=minimal_llm).interpret("톤 소리")
    check(res.source == "llm" and res.recipe.name and res.recipe.master.prefix == "[SFX]",
          "name/seed/master 누락 보강")

    # 6) 시스템 프롬프트에 스키마/태그 포함
    itp = Interpreter()
    sysp = itp._build_system_prompt()
    check("레시피" in sysp and "sub_impact" in sysp, "시스템 프롬프트에 스키마 안내 포함")

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "lib"
        (root / "foley").mkdir(parents=True)
        sf.write(str(root / "foley" / "metal_clang.wav"),
                 (np.random.default_rng(0).standard_normal(48000) * 0.2).astype("float32"), 48000)
        lib = Library(root)
        lib.scan()
        sysp2 = Interpreter(library=lib)._build_system_prompt()
        check("metal" in sysp2 or "clang" in sysp2, "라이브러리 태그가 시스템 프롬프트에 주입됨")

        # 7) LLM 결과를 실제 렌더(엔드투엔드)
        out = Path(td) / "out"
        out.mkdir()
        res = Interpreter(llm=good_llm).interpret("프롬프트")
        path = render_recipe(res.recipe, out)
        check(path.exists(), f"LLM 레시피 렌더: {path.name}")

    # 8) 모델 자동 선택 우선순위
    check(_auto_pick(["llama3", "qwen2.5:7b", "gpt-oss:20b"]) == "qwen2.5:7b", "자동선택: qwen 우선")
    check(_auto_pick(["gpt-oss:20b", "llama3"]) == "gpt-oss:20b", "자동선택: 다음 gpt-oss")
    check(_auto_pick(["llama3"]) == "llama3", "자동선택: 그 외 첫 번째")

    passed = sum(_results)
    print(f"\n{'='*50}\n결과: {passed}/{len(_results)} 통과")
    return 0 if passed == len(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
