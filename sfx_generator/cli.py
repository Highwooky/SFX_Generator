"""CLI 엔트리포인트: 레시피 JSON 또는 프롬프트를 받아 WAV로 렌더한다.

사용 예:
    python -m sfx_generator.cli recipe.json --out ./out
    python -m sfx_generator.cli --prompt "공포 영화 문 삐걱 소리에 저음 쿵" --out ./out --library ./samples
    python -m sfx_generator.cli --prompt "경쾌한 UI 알림음" --out ./out --variations 5

설계 의도(Why):
- GUI와 무관하게 엔진을 단독 실행/배치할 수 있는 진입점. 향후 macOS Quick Action에서
  이 CLI를 호출하면 Finder 우클릭 워크플로와도 연동된다.
- 레시피 파일 모드와 프롬프트 모드를 한 진입점에서 처리한다(둘 중 하나는 필수).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from .core.interpreter import Interpreter
from .core.library import Library
from .core.recipe import Recipe
from .core.render import render_recipe, render_variations


def build_library(library_dir: Optional[Path]) -> Optional[Library]:
    """런타임 지정 폴더로 라이브러리를 구성(없으면 None → synth 폴백 경로)."""
    if library_dir is None:
        return None
    lib = Library(library_dir, cache_dir=Path(library_dir) / ".sfx_cache")
    count = lib.scan()
    print(f"📂 라이브러리 색인: {count}개 음원 ({library_dir})")
    return lib


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="SFX Forge — 방송 효과음 생성기")
    parser.add_argument("recipe", type=Path, nargs="?", help="레시피 JSON 파일 경로")
    parser.add_argument("--prompt", type=str, default=None, help="자연어 프롬프트(룰 해석)")
    parser.add_argument("--out", type=Path, default=Path("./out"), help="출력 디렉토리")
    parser.add_argument("--variations", type=int, default=1, help="생성할 변주 개수")
    parser.add_argument("--library", type=Path, default=None, help="라이브러리 음원 폴더(선택)")
    parser.add_argument("--llm", nargs="?", const="auto", default=None,
                        help="Ollama 사용(모델명 지정 가능, 생략 시 자동선택). 실패 시 룰 폴백")
    parser.add_argument("--ollama-host", default="http://localhost:11434", help="Ollama 호스트")
    parser.add_argument("--preset", type=str, default=None, help="프리셋 이름으로 생성")
    parser.add_argument("--list-presets", action="store_true", help="사용 가능한 프리셋 목록 출력")
    parser.add_argument("--expand", action="store_true", help="Ollama로 프롬프트를 상세 브리프로 확장 후 생성")
    parser.add_argument("--length", type=float, default=None, help="효과 길이(초). 지정 시 정확히 그 길이로 출력")
    parser.add_argument("--lufs", type=float, default=None, help="라우드니스 타깃(LUFS). 예: -16 일반, -23 EBU, -24 ATSC")
    parser.add_argument("--reference", type=Path, default=None, help="레퍼런스 음원을 분석해 비슷한 소리 생성")
    parser.add_argument("--process", type=Path, default=None, help="원본 음원을 소스로 받아 변형/보강(변주)")
    parser.add_argument("--style", type=str, default=None, help="원본 가공 스타일(예: 공간감(홀), 어둡게, 리버스 …)")
    args = parser.parse_args(argv)

    # 프리셋 목록 출력 후 종료
    if args.list_presets:
        from .core.presets import list_presets

        print("사용 가능한 프리셋:")
        for name, desc in list_presets():
            print(f"  • {name:14} — {desc}")
        return 0

    # 입력은 레시피 파일 / 프롬프트 / 프리셋 중 정확히 하나
    chosen = [bool(args.recipe), bool(args.prompt), bool(args.preset), bool(args.reference), bool(args.process)]
    if sum(chosen) != 1:
        print("❌ recipe.json / --prompt / --preset / --reference / --process 중 정확히 하나를 지정하세요.", file=sys.stderr)
        return 1

    try:
        library = build_library(args.library)
    except Exception as e:  # noqa: BLE001
        print(f"❌ 라이브러리 구성 실패: {e}", file=sys.stderr)
        return 1
    resolver = library.as_resolver() if library is not None else None

    # ── 레시피 확보(프리셋 / 프롬프트 / 파일) ───────────────────────────────
    if args.process:
        try:
            from .core.process import build_recipe, make_source_resolver

            if not args.process.exists():
                print(f"❌ 원본 파일을 찾을 수 없습니다: {args.process}", file=sys.stderr)
                return 1
            style = args.style or "공간감(홀)"
            recipe = build_recipe(style, name=f"{args.process.stem}_{style}")
            resolver = make_source_resolver(args.process, resolver)
            print(f"🎚 원본 가공: {args.process.name} · 스타일 {style}")
        except Exception as e:  # noqa: BLE001
            print(f"❌ 원본 가공 실패: {e}", file=sys.stderr)
            return 1
    elif args.reference:
        try:
            from .core.reference import (
                REF_MATCH_QUERY,
                analyze_reference,
                feature_tags,
                recipe_from_features,
                recipe_from_library_match,
            )

            feat = analyze_reference(args.reference)
            match = library.best_for_tags(feature_tags(feat)) if library is not None else None
            if match is not None:
                recipe = recipe_from_library_match(match)
                base = resolver
                resolver = (lambda q, p, _m=match, _b=base: _m if q == REF_MATCH_QUERY else (_b(q, p) if _b else None))
                print(f"🎯 레퍼런스+라이브러리 매칭: {match.name}")
            else:
                recipe, summary = recipe_from_features(feat)
                print(f"🎯 레퍼런스 분석: {summary}")
        except Exception as e:  # noqa: BLE001
            print(f"❌ 레퍼런스 분석 실패: {e}", file=sys.stderr)
            return 1
    elif args.preset:
        try:
            from .core.presets import get_recipe

            recipe = get_recipe(args.preset)
            print(f"🎚️ 프리셋: {args.preset}")
        except Exception as e:  # noqa: BLE001
            print(f"❌ {e}", file=sys.stderr)
            return 1
    elif args.prompt:
        # 카테고리/모음 요청이면 대표 자막 효과음 '팩'을 만들어 바로 렌더
        from .core.presets import concept_pack, detect_concept

        concept = detect_concept(args.prompt)
        if concept is not None:
            n = max(4, args.variations)
            pack = concept_pack(concept, n=n)
            print(f"🎬 {concept} 자막 효과음 팩 — {len(pack)}개")
            made = []
            for label, rec in pack:
                if args.lufs is not None:
                    rec.master.loudness_lufs = args.lufs
                rec.name = f"{concept}_{label}"
                try:
                    p = render_recipe(rec, args.out, resolver, length_sec=args.length)
                    made.append(p)
                    print(f"  ✅ {p.name}")
                except Exception as e:  # noqa: BLE001
                    print(f"  ❌ {label}: {e}", file=sys.stderr)
            return 0 if made else 1

        llm = None
        if args.llm is not None:
            try:
                from .core.llm import make_ollama_llm

                model = None if args.llm == "auto" else args.llm
                llm = make_ollama_llm(args.ollama_host, model)
                print(f"🧠 Ollama 사용: {getattr(llm, '__self__', None) and llm.__self__.model or model or 'auto'}")
            except Exception as e:  # noqa: BLE001
                print(f"⚠️ Ollama 사용 불가({e}) → 룰 엔진으로 진행")
                llm = None
        try:
            result = Interpreter(library=library, llm=llm, expand=args.expand, length_sec=args.length).interpret(args.prompt)
        except Exception as e:  # noqa: BLE001
            print(f"❌ 프롬프트 해석 실패: {e}", file=sys.stderr)
            return 1
        recipe = result.recipe
        print(f"🔎 해석({result.source}) 소리: {', '.join(result.matched_types) or '없음'}")
        if result.matched_modifiers:
            print(f"🔧 적용 수식어: {', '.join(result.matched_modifiers)}")
    else:
        if not args.recipe.exists():
            print(f"❌ 레시피 파일을 찾을 수 없습니다: {args.recipe}", file=sys.stderr)
            return 1
        try:
            recipe = Recipe.from_json(args.recipe.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            print(f"❌ 레시피 검증 실패:\n{e}", file=sys.stderr)
            return 1

    # ── 렌더 ────────────────────────────────────────────────────────────────
    try:
        if args.lufs is not None:
            recipe.master.loudness_lufs = args.lufs  # 방송 타깃 오버라이드
        if args.variations > 1:
            paths = render_variations(recipe, args.variations, args.out, resolver, length_sec=args.length)
        else:
            paths = [render_recipe(recipe, args.out, resolver, length_sec=args.length)]
    except Exception as e:  # noqa: BLE001 - 렌더 실패 원인을 명확히 출력
        print(f"❌ 렌더 실패: {e}", file=sys.stderr)
        return 1

    for p in paths:
        print(f"✅ {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
