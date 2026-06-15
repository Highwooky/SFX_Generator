"""확장 기능 통합 테스트: 합성 6종 · granular/spectral · 자동태깅 · 프리셋 · 해석기 · LLM확장."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from core.interpreter import Interpreter
from core.library import Library
from core.presets import get_recipe, list_presets
from core.recipe import Recipe
from core.render import render_recipe

PASS, FAIL = "✅", "❌"
_results = []


def check(cond, msg):
    _results.append(bool(cond))
    print(f"{PASS if cond else FAIL} {msg}")


def main() -> int:
    itp = Interpreter()

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "out"
        out.mkdir()

        # 1) 새 합성 6종 렌더
        for kind, extra in [("modal", {"material": "glass"}), ("pluck", {}), ("fm", {}),
                            ("wind", {}), ("rain", {}), ("fire", {})]:
            rec = Recipe.model_validate({
                "name": f"t_{kind}", "seed": 1,
                "layers": [{"synth": {"kind": kind, "freq": 300, "decay": 1.0, **extra}}],
                "master": {"format": {"bit": 24, "rate": 48000}, "prefix": "[SFX]"},
            })
            p = render_recipe(rec, out)
            check(p.exists(), f"합성 렌더: {kind}")

        # 2) granular / spectral 변형 렌더
        for tf in [{"op": "granular", "grain_ms": 60, "density": 3, "pitch_jitter": 4, "stretch": 6, "spray_ms": 30},
                   {"op": "spectral", "mode": "freeze", "amount": 4}]:
            rec = Recipe.model_validate({
                "name": f"t_{tf['op']}", "seed": 1,
                "layers": [{"synth": {"kind": "tone", "freq": 330, "decay": 0.6}, "transforms": [tf]}],
                "master": {"format": {"bit": 24, "rate": 48000}, "prefix": "[SFX]"},
            })
            p = render_recipe(rec, out)
            info = sf.info(str(p))
            check(p.exists() and info.channels == 2, f"변형 렌더: {tf['op']}")

        # 3) 자동 태깅: 짧고 밝은 노이즈 vs 길고 낮은 사인
        lib_root = Path(td) / "lib"
        lib_root.mkdir()
        rng = np.random.default_rng(0)
        sf.write(str(lib_root / "track_a.wav"), (rng.standard_normal(int(48000 * 0.2)) * 0.3).astype("float32"), 48000)
        tsine = np.sin(2 * np.pi * 90 * np.arange(48000 * 3) / 48000).astype("float32") * 0.5
        sf.write(str(lib_root / "track_b.wav"), tsine, 48000)
        lib = Library(lib_root, analyze=True)
        lib.scan()
        tags = {s.path.name: s.tags for s in lib.samples}
        check("short" in tags["track_a.wav"], "자동태깅: 짧은 노이즈 → short")
        check("long" in tags["track_b.wav"] and "dark" in tags["track_b.wav"], "자동태깅: 긴 저음 사인 → long/dark")
        check("tonal" in tags["track_b.wav"], "자동태깅: 사인 → tonal")

        # 4) 프리셋 전체 렌더
        names = [n for n, _ in list_presets()]
        check(len(names) >= 10, f"프리셋 {len(names)}종 등록")
        ok_all = True
        for n in names:
            try:
                p = render_recipe(get_recipe(n), out)
                d, _ = sf.read(str(p))
                if not (p.exists() and np.all(np.isfinite(d))):
                    ok_all = False
            except Exception:  # noqa: BLE001
                ok_all = False
                print("   프리셋 실패:", n)
        check(ok_all, "모든 프리셋 정상 렌더")

    # 5) 해석기 새 키워드 매핑
    gl = itp.interpret("유리 깨지는 소리").recipe.layers
    check(any(l.synth and l.synth.kind == "modal" for l in gl), "키워드: 유리 → modal(템플릿 다층)")
    check(itp.interpret("레이저 발사음").recipe.layers[0].synth.kind == "fm", "키워드: 레이저 → fm")
    check(itp.interpret("모닥불 타는 소리").recipe.layers[0].synth.kind == "fire", "키워드: 모닥불 → fire")
    check(itp.interpret("기타 현 튕기는 소리").recipe.layers[0].synth.kind == "pluck", "키워드: 현 → pluck")

    # 6) LLM 프롬프트 확장(mock): expand=True → 확장 호출 후 레시피 생성
    VALID = ('{"name":"x","seed":1,"layers":[{"synth":{"kind":"wind","freq":500,"decay":2.0}}],'
             '"master":{"format":{"bit":24,"rate":48000},"prefix":"[SFX]"}}')

    def mock_llm(system, user):
        if "brief" in system:           # 확장 요청
            return '{"brief": "성당 공간의 낮고 음산한 바람, 8초, 서서히 고조"}'
        assert "[상세 브리프]" in user   # 확장된 프롬프트가 레시피 생성에 전달됐는지
        return VALID

    res = Interpreter(llm=mock_llm, expand=True).interpret("바람소리")
    check(res.source == "llm" and res.recipe.layers[0].synth.kind == "wind", "LLM 프롬프트 확장 경로")

    # 7) 길이(초) 지정: 정확히 그 길이로 출력 + 단일 베이스 duration 채움
    r2 = Interpreter(length_sec=2.0).interpret("바람소리")
    check(abs(r2.recipe.layers[0].synth.duration - 2.0) < 1e-6, "길이 지정: 단일 베이스 duration=2.0")
    with tempfile.TemporaryDirectory() as td2:
        p = render_recipe(r2.recipe, Path(td2), length_sec=2.0)
        info = sf.info(str(p))
        check(abs(info.frames - 2 * info.samplerate) <= 1, "길이 지정: 출력 길이 정확히 2초")

    # 8) 내부 템플릿 자동 적용(프리셋 목록 비노출): 두둥/와장창 → 다층 레시피
    check(len(Interpreter().interpret("두둥 반전 등장").recipe.layers) >= 2, "내부 템플릿: 두둥 → 다층")
    check(len(Interpreter().interpret("와장창 깨지는 소리").recipe.layers) >= 3, "내부 템플릿: 와장창 → 다층")

    # 9) 레퍼런스 → 레시피 매핑 + 실제 분석
    from core.reference import analyze_reference, recipe_from_features
    rt, _ = recipe_from_features({"duration": 2.0, "centroid": 300, "flatness": 0.02, "dominant_freq": 120, "percussive": False, "attack_time": 0.5})
    check(rt.layers[0].synth.kind == "tone", "레퍼런스: 저음 지속 → tone")
    ri, _ = recipe_from_features({"duration": 0.3, "centroid": 200, "flatness": 0.05, "dominant_freq": 80, "percussive": True, "attack_time": 0.01})
    check(ri.layers[0].synth.kind == "sub_impact", "레퍼런스: 저음 타격 → sub_impact")
    rn, _ = recipe_from_features({"duration": 2.0, "centroid": 500, "flatness": 0.6, "dominant_freq": 500, "percussive": False, "attack_time": 0.5})
    check(rn.layers[0].synth.kind in ("wind", "noise"), "레퍼런스: 노이즈 지속 → wind/noise")
    with tempfile.TemporaryDirectory() as td3:
        f = Path(td3) / "r.wav"
        sf.write(str(f), (np.sin(2 * np.pi * 120 * np.arange(48000) / 48000) * 0.5).astype("float32"), 48000)
        feat = analyze_reference(f)
        check(feat["duration"] > 0.9 and not feat["percussive"], "레퍼런스: 실제 음원 분석 동작")
        # 정밀: 비조화 밝은 링 → modal + 감쇠 실측(길이보다 짧게)
        tt = np.arange(48000 * 2) / 48000
        ring = sum(np.sin(2 * np.pi * 2600 * r * tt) for r in (1.0, 2.3, 4.2)) * np.exp(-tt / 0.3)
        fr = Path(td3) / "ring.wav"
        sf.write(str(fr), (ring / (np.max(np.abs(ring)) or 1) * 0.8).astype("float32"), 48000)
        feat2 = analyze_reference(fr)
        rr, _ = recipe_from_features(feat2)
        check(rr.layers[0].synth.kind == "modal", "레퍼런스: 비조화 밝은 링 → modal")
        check(feat2["decay_time"] < 1.5, "레퍼런스: 감쇠 시간 실측(<길이)")

    # 10) 엔벨로프 폴로잉 · 다중 피치 · 라이브러리 매칭
    from core.dsp import apply_transform
    from core.recipe import EnvelopeTransform
    from core.reference import (
        REF_MATCH_QUERY, feature_tags, recipe_from_library_match,
    )

    sig = np.ones(2000, dtype=np.float32)
    shaped = apply_transform(sig, 48000, EnvelopeTransform(op="envelope", points=[(0.0, 1.0), (1.0, 0.0)]))
    check(shaped[0] > 0.9 and shaped[-1] < 0.05, "엔벨로프 변형: 음량 곡선 적용")

    with tempfile.TemporaryDirectory() as tdc:
        ct = np.arange(48000) / 48000
        chord = np.sin(2 * np.pi * 440 * ct) + np.sin(2 * np.pi * 660 * ct)  # 화음(지속)
        fc = Path(tdc) / "chord.wav"
        sf.write(str(fc), (chord / np.max(np.abs(chord)) * 0.7).astype("float32"), 48000)
        rc, _ = recipe_from_features(analyze_reference(fc))
        check(len(rc.layers) >= 2, "다중 피치: 화음 → 레이어 2개 이상")

    with tempfile.TemporaryDirectory() as tdl:
        root = Path(tdl) / "lib"
        root.mkdir()
        rng2 = np.random.default_rng(1)
        sf.write(str(root / "bright_hit.wav"), (rng2.standard_normal(int(48000 * 0.2)) * 0.4).astype("float32"), 48000)
        sf.write(str(root / "low_drone.wav"), (np.sin(2 * np.pi * 80 * np.arange(48000 * 3) / 48000) * 0.4).astype("float32"), 48000)
        lib = Library(root, analyze=True)
        lib.scan()
        ref = Path(tdl) / "ref.wav"
        sf.write(str(ref), (rng2.standard_normal(int(48000 * 0.2)) * 0.4).astype("float32"), 48000)
        match = lib.best_for_tags(feature_tags(analyze_reference(ref)))
        check(match is not None, "라이브러리 매칭: 특징으로 보유 샘플 선택")
        if match is not None:
            rec = recipe_from_library_match(match)
            resolver = (lambda q, p, _m=match: _m if q == REF_MATCH_QUERY else None)
            pp = render_recipe(rec, Path(tdl) / "o", resolver)
            check(pp.exists(), "라이브러리 매칭: 실제 샘플 렌더")
        # 진행 콜백: 마지막 보고가 (total,total)
        prog = []
        Library(root, analyze=True).scan(force=True, progress=lambda d, t: prog.append((d, t)))
        check(bool(prog) and prog[-1][0] == prog[-1][1] and prog[-1][1] == 2, "스캔 진행 콜백 보고")
        # 캐시 즉시 로드: 스캔(시그니처 재계산) 없이 색인 복원
        Library(root, cache_dir=root / ".sfx_cache", analyze=True).scan()
        check(Library(root, cache_dir=root / ".sfx_cache").load_cached() == 2, "캐시 즉시 로드(스캔 없이)")

    # 11) 원본 가공(변주·보강): 실제 원본을 소스로 받아 변형
    from core.process import SOURCE_QUERY, build_recipe, make_source_resolver, variation_recipes
    with tempfile.TemporaryDirectory() as tdp:
        src = Path(tdp) / "orig.wav"
        ot = np.arange(48000) / 48000
        sf.write(str(src), (np.sin(2 * np.pi * 300 * ot) * np.exp(-ot / 0.3) * 0.7).astype("float32"), 48000)
        resolver = make_source_resolver(src)
        rec = build_recipe("공간감(홀)", name="p")
        check(rec.layers[0].source.query == SOURCE_QUERY, "원본 가공: 소스 센티넬 참조")
        pth = render_recipe(rec, Path(tdp) / "o", resolver)
        check(pth.exists(), "원본 가공: 스타일 렌더(실제 원본)")
        vs = variation_recipes(3, base_name="orig")
        check(len(vs) == 3, "원본 가공: 변주 3개 생성")
        check(len(build_recipe("묵직하게+서브").layers) >= 2, "원본 가공: 보강 레이어 추가")

    # 12) 컨셉 추론 → 자막 효과음 팩
    from core.presets import concept_pack, detect_concept
    check(detect_concept("예능에서 쓸만한 자막 효과음") == "예능", "컨셉: 예능 감지")
    check(detect_concept("교양 다큐 인서트 모음") == "교양", "컨셉: 교양 감지")
    check(detect_concept("밝은 자막 효과") is None, "컨셉: 단일 요청은 팩 아님")
    pk = concept_pack("예능", n=5)
    check(len(pk) == 5 and all(isinstance(r, Recipe) for _, r in pk), "컨셉: 예능 팩 5개 유효")
    check(len(concept_pack("교양", n=6)) == 6, "컨셉: 교양 팩 6개")
    with tempfile.TemporaryDirectory() as tdk:
        pp = render_recipe(pk[0][1], Path(tdk) / "o", None)
        check(pp.exists(), "컨셉: 팩 레시피 렌더 가능")

    # 13) AI 생성 클라이언트(로컬 서버 호출, 표준 라이브러리)
    from core import aigen
    check(aigen.health("http://127.0.0.1:59999", timeout=0.3) is None, "AI 클라이언트: 미연결 시 None")
    with tempfile.TemporaryDirectory() as tda:
        ap = aigen.save_wav(b"RIFF....WAVE", Path(tda) / "a.wav")
        check(ap.exists(), "AI 클라이언트: WAV 저장")

    # 14) 조절 노브 → 변형 적용 + 렌더
    from core.adjust import apply_knobs
    kbase = Recipe.model_validate({"name": "k", "seed": 0,
            "layers": [{"synth": {"kind": "tone", "freq": 440, "decay": 0.6}}],
            "master": {"loudness_lufs": -16, "limiter": True, "format": {"bit": 24, "rate": 48000}, "prefix": "[SFX]"}})
    adj = apply_knobs(kbase, brightness=6, pitch=-3, space=0.3, weight=4, grit=5, attack=0.05)
    ops = [t.op for t in adj.layers[0].transforms]
    check(all(o in ops for o in ["pitch", "eq", "distortion", "reverb", "fade"]), "노브: 변형 체인 구성")
    adj_dark = apply_knobs(kbase, brightness=-8)
    check(any(t.op == "filter" for t in adj_dark.layers[0].transforms), "노브: 어두움→로우패스")
    with tempfile.TemporaryDirectory() as tdn:
        check(render_recipe(adj, Path(tdn) / "o", None).exists(), "노브: 조절 결과 렌더")

    # 15) 스템 자동 분할 + 라이브러리 순위 검색
    from core.stems import split_stem
    with tempfile.TemporaryDirectory() as tds:
        sr = 48000
        sig = np.zeros(int(sr * 3), dtype=np.float32)
        for start in (0.2, 1.2, 2.2):  # 3개 버스트 + 사이 무음
            s = int(start * sr)
            b = (np.random.default_rng(0).standard_normal(int(0.3 * sr)) * 0.5
                 * np.hanning(int(0.3 * sr))).astype(np.float32)
            sig[s:s + len(b)] += b
        stem = Path(tds) / "stem.wav"
        sf.write(str(stem), sig, sr)
        clips = split_stem(stem, Path(tds) / "out")
        check(len(clips) == 3 and all(p.exists() for p in clips), "스템 분할: 3개 클립 추출")

    with tempfile.TemporaryDirectory() as tdr:
        root2 = Path(tdr) / "lib"
        root2.mkdir()
        sf.write(str(root2 / "bright_hit.wav"), (np.random.default_rng(1).standard_normal(int(48000 * 0.2)) * 0.4).astype("float32"), 48000)
        sf.write(str(root2 / "low_tone.wav"), (np.sin(2 * np.pi * 90 * np.arange(48000) / 48000) * 0.4).astype("float32"), 48000)
        lib2 = Library(root2, analyze=True)
        lib2.scan()
        check(len(lib2.search_ranked("", limit=10)) == 2, "순위 검색: 빈 검색어→전체")
        check(isinstance(lib2.search_ranked("bright", limit=10), list), "순위 검색: 검색어 동작")

    passed = sum(_results)
    print(f"\n{'='*50}\n결과: {passed}/{len(_results)} 통과")
    return 0 if passed == len(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
