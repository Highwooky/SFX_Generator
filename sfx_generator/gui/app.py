"""GUI 엔트리포인트. 실행: python -m sfx_generator.gui"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6 import QtGui, QtWidgets

from .. import __app_name__
from .main_window import MainWindow

# 어두운 테마(오디오 작업 환경에 맞춤). 시스템 테마를 강제하지 않고 최소 스타일만.
_STYLE = """
QWidget { background:#1f2329; color:#e5e7eb; font-size:13px; }
QGroupBox { border:1px solid #32363f; border-radius:6px; margin-top:8px; padding-top:8px; }
QGroupBox::title { subcontrol-origin: margin; left:10px; color:#9ca3af; }
QPushButton { background:#2d333b; border:1px solid #3b424c; border-radius:5px; padding:6px 12px; }
QPushButton:hover { background:#3b424c; }
QPushButton:disabled { color:#6b7280; }
QLineEdit, QPlainTextEdit, QListWidget, QSpinBox, QComboBox {
    background:#16181d; border:1px solid #32363f; border-radius:5px; padding:4px;
}
"""


def _icon_path() -> Path | None:
    """개발/번들 양쪽에서 아이콘 PNG 경로를 찾는다(번들 시 _MEIPASS/assets)."""
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[2]))
    p = base / "assets" / "AppIcon_1024.png"
    return p if p.exists() else None


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(__app_name__)
    app.setStyleSheet(_STYLE)
    icon = _icon_path()
    if icon is not None:
        app.setWindowIcon(QtGui.QIcon(str(icon)))
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
