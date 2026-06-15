# -*- mode: python ; coding: utf-8 -*-
"""SFX Forge PyInstaller spec — macOS .app 번들.

설계 의도(Why):
- pedalboard/soundfile/sounddevice는 컴파일된 확장과 동적 라이브러리(libsndfile,
  PortAudio)를 포함하므로 collect_all로 바이너리·데이터를 빠짐없이 수집한다.
- PySide6는 PyInstaller 내장 훅이 처리하므로 별도 hiddenimports를 두지 않는다.
- 빌드는 GitHub Actions의 macos-14(Apple Silicon) 러너에서 수행 → arm64 .app 산출.
"""

from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = [], [], []
for pkg in ("pedalboard", "soundfile", "sounddevice", "pyloudnorm", "scipy"):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception as e:  # noqa: BLE001
        # 선택 패키지(예: 백엔드 없는 머신의 sounddevice)는 수집 실패해도 빌드 계속
        print(f"[spec] '{pkg}' 수집 건너뜀: {e}")

# 런타임 창 아이콘용 PNG를 번들에 포함(.icns는 BUNDLE이 dock 아이콘으로 사용)
datas += [("assets/AppIcon_1024.png", "assets")]

a = Analysis(
    ["run_gui.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "PyQt5", "PyQt6"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="SFX_Generator",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # GUI 앱이므로 콘솔 창 없음
    target_arch="arm64",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="SFX_Generator",
)

app = BUNDLE(
    coll,
    name="SFX_Generator.app",
    icon="assets/AppIcon.icns",
    bundle_identifier="com.jtbc.sfxgenerator",
    info_plist={
        "NSHighResolutionCapable": True,
        "LSMinimumSystemVersion": "12.0",
        "CFBundleShortVersionString": "1.4.0",
    },
)
