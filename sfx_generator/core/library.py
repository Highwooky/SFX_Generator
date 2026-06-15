"""보유 음원 라이브러리 인덱싱 + 검색.

설계 의도(Why):
- 라이브러리 루트는 '실행 시점에' 주입한다(하드코딩 금지). GUI에서 사용자가 고른
  폴더를 그대로 Library(root)로 넘기면 된다.
- 기본 검색은 파일명·폴더명에서 뽑은 태그 매칭이라 외부 의존성이 전혀 없다(에어갭 OK).
- 의미검색(임베딩)은 선택. embedder를 주입했을 때만 활성화되어, 모델이 없어도
  앱은 항상 동작한다(점진적 고도화).
- 색인은 폴더 시그니처(경로/수정시각/크기) 기반으로 캐시해, 바뀐 게 없으면 재스캔하지 않는다.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import soundfile as sf

# 지원 오디오 확장자. 필요 시 여기만 늘리면 된다.
AUDIO_EXTS = {".wav", ".aiff", ".aif", ".flac", ".mp3", ".ogg", ".m4a"}

# 태그로 의미 없는 토큰(번호/짧은 조각) 제거용
_SPLIT = re.compile(r"[\s_\-.,()\[\]]+")
_NUM_ONLY = re.compile(r"^\d+$")


@dataclass
class IndexedSample:
    path: Path
    tags: set[str]


@dataclass
class Library:
    """런타임에 지정된 음원 폴더를 색인하고 검색을 제공한다."""

    root: Path
    cache_dir: Optional[Path] = None
    # embedder: (list[str]) -> list[vector]. 주입 시 의미검색 활성화(선택).
    embedder: Optional[Callable[[list[str]], list]] = None
    # analyze: 색인 시 오디오 특징을 분석해 길이/밝기/타격성 태그를 자동 부여
    analyze: bool = True
    samples: list[IndexedSample] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        if not self.root.exists():
            raise FileNotFoundError(f"라이브러리 폴더가 없습니다: {self.root}")

    # ── 오디오 특징 자동 태깅 ────────────────────────────────────────────────
    @staticmethod
    def _analyze(path: Path) -> set[str]:
        """오디오를 가볍게 분석해 서술 태그를 만든다(파일명이 부실해도 검색 적중↑).

        Why: 'track07.wav' 같은 무의미한 파일명도 '밝은/짧은/타격' 같은 특징 태그가
        붙으면 프롬프트 검색에 잡힌다. 앞 1.5초만 읽어 비용을 억제한다.
        """
        try:
            data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        except Exception:  # noqa: BLE001 - 손상/미지원 파일은 특징 태그 없이 통과
            return set()
        mono = data.mean(axis=1)
        if mono.size == 0:
            return set()
        seg = mono[: int(sr * 1.5)] if sr else mono
        dur = len(mono) / sr if sr else 0.0
        tags: set[str] = set()

        # 길이
        tags.add("short" if dur < 0.5 else "long" if dur > 2.5 else "medium")

        # 밝기: 스펙트럼 무게중심(centroid)
        spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) + 1e-9
        freqs = np.fft.rfftfreq(len(seg), 1.0 / (sr or 48000))
        centroid = float((freqs * spec).sum() / spec.sum())
        tags.add("bright" if centroid > 3500 else "dark" if centroid < 800 else "mid")

        # 노이즈성 vs 토널: 스펙트럴 평탄도(flatness)
        gmean = np.exp(np.mean(np.log(spec)))
        amean = np.mean(spec)
        flatness = float(gmean / amean)
        tags.add("noisy" if flatness > 0.35 else "tonal" if flatness < 0.08 else "")
        tags.discard("")

        # 타격성 vs 지속성: 에너지가 앞쪽에 집중되면 타격(percussive)
        half = len(seg) // 2 or 1
        early = float(np.sum(seg[:half] ** 2))
        late = float(np.sum(seg[half:] ** 2)) + 1e-9
        tags.add("percussive" if early > late * 3 else "sustained")

        return tags

    # ── 태그 추출 ──────────────────────────────────────────────────────────
    @staticmethod
    def _extract_tags(path: Path, root: Path) -> set[str]:
        """파일명 + 상위 폴더명에서 검색용 토큰을 뽑는다.

        Why: 'impact/metal_hit_01.wav' → {impact, metal, hit}. 폴더 구조가 곧
        분류 정보이므로 상위 경로까지 토큰화하면 별도 메타데이터 없이도 검색이 된다.
        """
        parts = list(path.relative_to(root).parts)
        parts[-1] = Path(parts[-1]).stem  # 확장자 제거
        tokens: set[str] = set()
        for part in parts:
            for tok in _SPLIT.split(part.lower()):
                if len(tok) >= 2 and not _NUM_ONLY.match(tok):
                    tokens.add(tok)
        return tokens

    # ── 폴더 시그니처(캐시 무효화 판단) ──────────────────────────────────────
    def _signature(self) -> str:
        items = []
        for p in sorted(self.root.rglob("*")):
            if p.suffix.lower() in AUDIO_EXTS and p.is_file():
                st = p.stat()
                items.append(f"{p.relative_to(self.root)}|{int(st.st_mtime)}|{st.st_size}")
        return hashlib.sha1("\n".join(items).encode("utf-8")).hexdigest()

    def _cache_path(self) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        key = hashlib.sha1(str(self.root.resolve()).encode("utf-8")).hexdigest()[:16]
        return Path(self.cache_dir) / f"index_{key}.json"

    # ── 스캔/색인 ──────────────────────────────────────────────────────────
    def load_cached(self) -> Optional[int]:
        """시그니처 재계산 없이 캐시된 색인을 '즉시' 로드(앱 시작 가속). 없으면 None.

        전체 폴더 rglob/stat를 건너뛰므로 큰 라이브러리도 즉시 준비된다.
        폴더 변경분은 scan(force=True)나 '새로고침'으로 반영한다.
        """
        cache = self._cache_path()
        if cache is None or not cache.exists():
            return None
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            self.samples = [
                IndexedSample(self.root / s["rel"], set(s["tags"]))
                for s in data["samples"]
            ]
            return len(self.samples)
        except (json.JSONDecodeError, KeyError, OSError):
            return None

    def scan(self, force: bool = False, progress=None) -> int:
        """폴더를 색인한다. 캐시가 유효하면 재사용. 색인된 샘플 수를 반환.

        progress: 선택 콜백 progress(done:int, total:int). 분석 진행 상황 표시용.
        """
        sig = self._signature()
        cache = self._cache_path()

        if not force and cache is not None and cache.exists():
            try:
                data = json.loads(cache.read_text(encoding="utf-8"))
                if data.get("signature") == sig:
                    self.samples = [
                        IndexedSample(self.root / s["rel"], set(s["tags"]))
                        for s in data["samples"]
                    ]
                    if progress is not None:  # 캐시 적중: 즉시 완료 보고
                        progress(len(self.samples), len(self.samples))
                    return len(self.samples)
            except (json.JSONDecodeError, KeyError, OSError):
                pass  # 캐시 손상 시 조용히 재스캔

        # 대상 파일 먼저 수집(전체 개수를 알아야 진행률 계산)
        files = [p for p in sorted(self.root.rglob("*"))
                 if p.suffix.lower() in AUDIO_EXTS and p.is_file()]
        total = len(files)
        self.samples = []
        for i, p in enumerate(files, 1):
            tags = self._extract_tags(p, self.root)
            if self.analyze:
                tags |= self._analyze(p)  # 오디오 특징 태그 병합(느린 단계)
            self.samples.append(IndexedSample(p, tags))
            if progress is not None:
                progress(i, total)

        if cache is not None:
            payload = {
                "signature": sig,
                "samples": [
                    {"rel": str(s.path.relative_to(self.root)), "tags": sorted(s.tags)}
                    for s in self.samples
                ],
            }
            cache.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

        return len(self.samples)

    # ── 검색 ──────────────────────────────────────────────────────────────
    def _score(self, query_tokens: set[str], sample: IndexedSample) -> float:
        """태그 겹침 + 부분일치로 점수 산출. 외부 의존성 없는 결정적 스코어."""
        if not query_tokens:
            return 0.0
        exact = len(query_tokens & sample.tags)
        # 부분일치(예: 'creak' 검색어가 'creaking' 태그에 포함)도 약하게 가점
        partial = sum(
            0.5
            for q in query_tokens
            for t in sample.tags
            if q != t and (q in t or t in q)
        )
        return exact + partial

    def search(self, query: str, pick: str = "best") -> Optional[Path]:
        """검색어로 가장 잘 맞는 샘플 경로를 반환(없으면 None)."""
        if not self.samples:
            self.scan()
        q_tokens = {
            tok for tok in _SPLIT.split(query.lower()) if len(tok) >= 2 and not _NUM_ONLY.match(tok)
        }
        scored = [(self._score(q_tokens, s), s) for s in self.samples]
        scored = [(sc, s) for sc, s in scored if sc > 0]
        if not scored:
            return None
        scored.sort(key=lambda x: x[0], reverse=True)

        if pick == "random":
            # 최고점과 동점인 후보들 중에서 무작위 선택(다양성 확보)
            top = scored[0][0]
            best = [s.path for sc, s in scored if sc == top]
            return random.choice(best)
        return scored[0][1].path

    def search_ranked(self, query: str, limit: int = 40) -> list:
        """검색어에 맞는 샘플들을 점수 순으로 반환. 빈 검색어면 전체를 이름순으로.

        반환: [(Path, score, tags(set)), ...]  — 라이브러리 브라우저용.
        """
        q_tokens = {
            tok for tok in _SPLIT.split(query.lower()) if len(tok) >= 2 and not _NUM_ONLY.match(tok)
        }
        if not q_tokens:  # 검색어 없음 → 전체를 이름순으로
            items = sorted(self.samples, key=lambda s: s.path.name.lower())
            return [(s.path, 0.0, s.tags) for s in items[:limit]]
        scored = [(self._score(q_tokens, s), s) for s in self.samples]
        scored = [(sc, s) for sc, s in scored if sc > 0]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [(s.path, sc, s.tags) for sc, s in scored[:limit]]

    def as_resolver(self) -> Callable[[str, str], Optional[Path]]:
        """render.py가 기대하는 resolver 콜러블로 변환."""
        return lambda query, pick: self.search(query, pick)

    def best_for_tags(self, tags: list[str]) -> Optional[Path]:
        """레퍼런스 특징 태그(short/bright/percussive 등)와 가장 많이 겹치는 샘플 경로.

        레퍼런스 매칭에서 '합성 대신 실제 보유 샘플'을 쓰고 싶을 때 사용한다.
        """
        if not self.samples:
            self.scan()
        if not self.samples:
            return None
        tset = set(tags)
        best, best_score = None, 0
        for s in self.samples:
            score = len(tset & set(s.tags))
            if score > best_score:
                best, best_score = s, score
        return best.path if (best is not None and best_score > 0) else None

    def all_tags(self) -> set[str]:
        """색인된 전체 태그 집합. 룰 해석기가 검색어를 라이브러리에 맞춰 보정할 때 쓴다."""
        tags: set[str] = set()
        for s in self.samples:
            tags |= s.tags
        return tags
