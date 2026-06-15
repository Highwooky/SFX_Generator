"""PyInstaller가 번들하는 GUI 진입 스크립트."""

import sys

from sfx_generator.gui.app import main

if __name__ == "__main__":
    sys.exit(main())
