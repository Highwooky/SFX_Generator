"""library.py 통합 테스트."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from core.library import Library
from core.recipe import Recipe
from core.render import render_recipe

PASS, FAIL = "✅", "❌"
_results = []


def check(cond, msg):
    _results.append(bool(cond))
    print(f"{PASS if cond else FAIL} {msg}")


def make_fake_library(root: Path):
    """폴더 구조가 곧 분류가 되도록 하위 폴더에 가짜 음원 배치."""
    layout = {
        "impact/metal_hit_01.wav": 44100,
        "impact/wood_thud_02.wav": 48000,
        "whoosh/air_swoosh.wav": 48000,
        "foley/door_creak_long.wav": 44100,
        "foley/glass_break.wav": 48000,
    }
    rng = np.random.default_rng(0)
    for rel, sr in layout.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        sig = (rng.standard_normal(sr) * 0.2).astype(np.float32)
        sf.write(str(p), sig, sr, subtype="PCM_16")


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "lib"
        cache = Path(td) / "cache"
        out = Path(td) / "out"
        root.mkdir()
        out.mkdir()
        make_fake_library(root)

        lib = Library(root, cache_dir=cache)
        n = lib.scan()
        check(n == 5, f"5개 음원 색인 (실제 {n})")

        # 폴더 구조 → 태그 추출 확인
        check("impact" in lib.all_tags() and "creak" in lib.all_tags(), "폴더/파일명 태그 추출")

        # 검색 정확도
        check(lib.search("creak").name == "door_creak_long.wav", "검색: creak → door_creak")
        check(lib.search("metal impact").name == "metal_hit_01.wav", "검색: metal impact 매칭")
        check(lib.search("whoosh air").name == "air_swoosh.wav", "검색: whoosh air 매칭")
        check(lib.search("존재하지않는소리") is None, "검색: 미매칭 → None")

        # 캐시 재사용(시그니처 동일 → 재스캔 안 함)
        cache_files = list(cache.glob("*.json"))
        check(len(cache_files) == 1, "캐시 파일 생성")
        lib2 = Library(root, cache_dir=cache)
        check(lib2.scan() == 5, "캐시에서 색인 복원")

        # resolver를 엔진에 연동 → 실제 렌더
        recipe = Recipe.from_json(
            """
        {"name":"creak_test","seed":1,
         "layers":[{"source":{"query":"creak","pick":"best"},
                    "transforms":[{"op":"pitch","semitones":-2},{"op":"reverb","kind":"room","wet":0.3}],
                    "start":0.0,"gain_db":0,"pan":0}],
         "master":{"format":{"bit":24,"rate":48000},"prefix":"[SFX]"}}
        """
        )
        path = render_recipe(recipe, out, resolver=lib.as_resolver())
        check(path.exists(), f"라이브러리 resolver로 렌더 성공: {path.name}")

    passed = sum(_results)
    print(f"\n{'='*50}\n결과: {passed}/{len(_results)} 통과")
    return 0 if passed == len(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
