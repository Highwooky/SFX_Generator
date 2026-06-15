"""파형 미리보기 위젯.

설계 의도(Why):
- 외부 차트 라이브러리 없이 QPainter로 직접 그린다(의존성 최소화, 에어갭 친화).
- 위젯 폭에 맞춰 매 paint마다 피크를 다시 계산해 창 크기 변화에 자연스럽게 반응한다.
"""

from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets


class WaveformWidget(QtWidgets.QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._mono: np.ndarray | None = None
        self.setMinimumHeight(140)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)

    def set_samples(self, samples: np.ndarray) -> None:
        """스테레오/모노 어느 쪽이 와도 표시용 모노로 정규화해 보관."""
        if samples is None or samples.size == 0:
            self._mono = None
        else:
            self._mono = samples.mean(axis=1) if samples.ndim == 2 else samples.astype(np.float32)
        self.update()

    def clear(self) -> None:
        self._mono = None
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        w, h = self.width(), self.height()
        mid = h / 2

        # 배경 + 중앙선
        painter.fillRect(self.rect(), QtGui.QColor("#16181d"))
        painter.setPen(QtGui.QPen(QtGui.QColor("#32363f"), 1))
        painter.drawLine(0, int(mid), w, int(mid))

        if self._mono is None or self._mono.size == 0 or w <= 0:
            painter.setPen(QtGui.QColor("#6b7280"))
            painter.drawText(self.rect(), QtCore.Qt.AlignCenter, "파형 없음 — 효과음을 생성하세요")
            return

        # 폭(px) 단위로 버킷을 나눠 각 열의 min/max를 세로선으로 그린다.
        n = self._mono.size
        samples_per_px = max(1, n // w)
        painter.setPen(QtGui.QPen(QtGui.QColor("#4ade80"), 1))
        peak = max(float(np.max(np.abs(self._mono))), 1e-6)
        for x in range(w):
            start = x * samples_per_px
            end = min(start + samples_per_px, n)
            if start >= end:
                break
            chunk = self._mono[start:end]
            lo = float(chunk.min()) / peak
            hi = float(chunk.max()) / peak
            y1 = mid - hi * mid * 0.95
            y2 = mid - lo * mid * 0.95
            painter.drawLine(x, int(y1), x, int(y2))
