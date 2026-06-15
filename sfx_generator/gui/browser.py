"""라이브러리 브라우저: 보유 샘플을 검색·미리듣기·다중 선택해서 합치거나 소스로 보낸다.

설계 의도(Why):
- 합성이 빗나가는 가장 큰 이유는 '맞는 샘플이 없을 때'. 실제 샘플을 순위로 보여주고
  들어보고 고르게 하면 정확도가 합성과 비교가 안 된다. 여러 개를 겹쳐(콜라주) 새 효과음도 만든다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from PySide6 import QtCore, QtWidgets

try:
    import sounddevice as sd
except Exception:  # noqa: BLE001
    sd = None


class LibraryBrowser(QtWidgets.QDialog):
    def __init__(self, library, parent=None) -> None:
        super().__init__(parent)
        self.library = library
        self.action: str | None = None          # "merge" | "source"
        self._selected: list[Path] = []
        self.setWindowTitle("라이브러리 브라우저 — 검색·미리듣기·합치기")
        self.resize(640, 520)

        v = QtWidgets.QVBoxLayout(self)
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("검색어 입력(예: glass, 타격, whoosh) — 비우면 전체")
        self.search.textChanged.connect(self._refresh)
        v.addWidget(self.search)

        self.list = QtWidgets.QListWidget()
        self.list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.list.itemClicked.connect(self._on_clicked)
        self.list.itemDoubleClicked.connect(lambda _it: self._merge())
        v.addWidget(self.list, 1)

        hint = QtWidgets.QLabel("클릭=미리듣기 · 여러 개 선택 후 '선택 합치기' · 더블클릭=바로 합치기")
        hint.setStyleSheet("color:#9ca3af;")
        v.addWidget(hint)

        btns = QtWidgets.QHBoxLayout()
        self.play_btn = QtWidgets.QPushButton("▶ 미리듣기")
        self.play_btn.clicked.connect(self._play_selected)
        self.stop_btn = QtWidgets.QPushButton("■ 정지")
        self.stop_btn.clicked.connect(self._stop)
        self.source_btn = QtWidgets.QPushButton("→ 원본 가공으로 보내기")
        self.source_btn.clicked.connect(self._to_source)
        self.merge_btn = QtWidgets.QPushButton("✚ 선택 합치기(레이어)")
        self.merge_btn.clicked.connect(self._merge)
        btns.addWidget(self.play_btn)
        btns.addWidget(self.stop_btn)
        btns.addStretch(1)
        btns.addWidget(self.source_btn)
        btns.addWidget(self.merge_btn)
        v.addLayout(btns)

        self._refresh()

    def _refresh(self) -> None:
        self.list.clear()
        for path, score, tags in self.library.search_ranked(self.search.text(), limit=80):
            tagstr = ", ".join(sorted(tags)[:6])
            item = QtWidgets.QListWidgetItem(f"{path.name}    [{tagstr}]")
            item.setData(QtCore.Qt.UserRole, str(path))
            self.list.addItem(item)
        if self.list.count() == 0:
            self.list.addItem("(결과 없음)")

    def _paths(self, only_selected: bool = True) -> list[Path]:
        items = self.list.selectedItems() if only_selected else [self.list.item(0)]
        out = []
        for it in items:
            p = it.data(QtCore.Qt.UserRole)
            if p:
                out.append(Path(p))
        return out

    def _on_clicked(self, item: QtWidgets.QListWidgetItem) -> None:
        p = item.data(QtCore.Qt.UserRole)
        if p:
            self._play(Path(p))

    def _play_selected(self) -> None:
        ps = self._paths()
        if ps:
            self._play(ps[0])

    def _play(self, path: Path) -> None:
        if sd is None:
            return
        try:
            data, sr = sf.read(str(path), dtype="float32", always_2d=True)
            sd.stop()
            sd.play(data, sr)
        except Exception:  # noqa: BLE001
            pass

    def _stop(self) -> None:
        if sd is not None:
            try:
                sd.stop()
            except Exception:  # noqa: BLE001
                pass

    def _to_source(self) -> None:
        ps = self._paths()
        if not ps:
            self.parent().statusBar().showMessage("샘플을 먼저 선택하세요") if self.parent() else None
            return
        self._stop()
        self._selected = ps[:1]
        self.action = "source"
        self.accept()

    def _merge(self) -> None:
        ps = self._paths()
        if not ps:
            return
        self._stop()
        self._selected = ps
        self.action = "merge"
        self.accept()

    def selected_paths(self) -> list[Path]:
        return self._selected
