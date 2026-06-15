"""백그라운드 렌더 워커.

설계 의도(Why):
- pedalboard/time_stretch 렌더는 수백 ms~수 초가 걸릴 수 있어 메인 스레드에서 돌리면
  UI가 멈춘다. QThread로 분리해 반응성을 유지한다.
- 시그널명은 QThread.finished와 충돌하지 않도록 finished_ok로 둔다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from PySide6 import QtCore

from ..core.library import Library
from ..core.recipe import Recipe
from ..core.render import render_recipe, render_variations


class ScanWorker(QtCore.QThread):
    """라이브러리 색인을 백그라운드에서 수행하며 진행률을 보고한다."""

    progress = QtCore.Signal(int, int)  # (done, total)
    finished_ok = QtCore.Signal(int)    # 색인된 샘플 수
    failed = QtCore.Signal(str)

    def __init__(self, library: Library, force: bool = False, parent=None) -> None:
        super().__init__(parent)
        self._library = library
        self._force = force

    def run(self) -> None:
        try:
            count = self._library.scan(force=self._force, progress=lambda d, t: self.progress.emit(d, t))
            self.finished_ok.emit(count)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class RenderWorker(QtCore.QThread):
    finished_ok = QtCore.Signal(list)  # list[str] 출력 경로
    failed = QtCore.Signal(str)

    def __init__(
        self,
        recipe: Recipe,
        out_dir: Path,
        resolver: Optional[Callable],
        variations: int,
        length_sec: Optional[float] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._recipe = recipe
        self._out_dir = out_dir
        self._resolver = resolver
        self._variations = max(1, variations)
        self._length_sec = length_sec

    def run(self) -> None:
        try:
            if self._variations > 1:
                paths = render_variations(self._recipe, self._variations, self._out_dir,
                                          self._resolver, length_sec=self._length_sec)
            else:
                paths = [render_recipe(self._recipe, self._out_dir, self._resolver,
                                       length_sec=self._length_sec)]
            self.finished_ok.emit([str(p) for p in paths])
        except Exception as e:  # noqa: BLE001 - 실패 사유를 UI로 그대로 전달
            self.failed.emit(str(e))


class AiGenWorker(QtCore.QThread):
    """로컬 AI 서버(예: Stable Audio Open)로 생성 → 원본 WAV 경로를 반환."""

    finished_ok = QtCore.Signal(str)  # 생성된 원본 wav 경로
    failed = QtCore.Signal(str)

    def __init__(self, prompt: str, seconds: float, host: str, seed: int, out_dir: Path, parent=None) -> None:
        super().__init__(parent)
        self._prompt = prompt
        self._seconds = seconds
        self._host = host
        self._seed = seed
        self._out_dir = out_dir

    def run(self) -> None:
        try:
            from ..core import aigen

            data = aigen.generate(self._prompt, self._seconds, self._host, self._seed)
            tmp = Path(self._out_dir) / ".ai_tmp"
            p = tmp / f"ai_{abs(hash(self._prompt)) % 100000}_{self._seed}.wav"
            aigen.save_wav(data, p)
            self.finished_ok.emit(str(p))
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class ProcessWorker(QtCore.QThread):
    """원본 가공: 여러 가공 레시피를 순차 렌더(실제 원본 오디오에 변형/보강)."""

    finished_ok = QtCore.Signal(list)
    failed = QtCore.Signal(str)

    def __init__(self, recipes, out_dir: Path, resolver, length_sec: Optional[float] = None, parent=None) -> None:
        super().__init__(parent)
        self._recipes = recipes  # list[Recipe]
        self._out_dir = out_dir
        self._resolver = resolver
        self._length_sec = length_sec

    def run(self) -> None:
        try:
            paths = [str(render_recipe(r, self._out_dir, self._resolver, length_sec=self._length_sec))
                     for r in self._recipes]
            self.finished_ok.emit(paths)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))
