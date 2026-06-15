#!/usr/bin/env python3
r"""[Windows에서 실행] 에어갭 Mac으로 옮길 'transfer' 꾸러미를 만든다.

Windows(인터넷)에서 macOS(Apple Silicon)용 파이썬 wheel과 Stable Audio Open 모델을
내려받아 한 폴더(transfer/)에 모은다. 이 폴더를 통째로 USB 등으로 Mac에 옮기면,
Mac에서는 인터넷 없이 install_offline_mac.command 로 설치/실행할 수 있다.

준비:
  1) Windows에 Python 설치(가능하면 Mac과 같은 버전: 기본 3.11).  https://www.python.org/downloads/
  2) Hugging Face 가입 후 https://huggingface.co/stabilityai/stable-audio-open-1.0 에서 라이선스 '동의'.
  3) 토큰 발급(https://huggingface.co/settings/tokens) → 아래 실행 시 입력.

실행(Windows 명령 프롬프트):
  python prepare_on_windows.py --py 3.11
  (Mac의 파이썬 버전과 --py 를 맞추세요.  Mac에서  python3 --version  으로 확인)

결과: transfer/  (wheels/, model/, sao_server.py, requirements.txt, install_offline_mac.command)
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = "stabilityai/stable-audio-open-1.0"
# Apple Silicon용 여러 macOS 버전 태그(되는 wheel을 폭넓게 확보)
MAC_PLATFORMS = ["macosx_11_0_arm64", "macosx_12_0_arm64", "macosx_13_0_arm64", "macosx_14_0_arm64"]


def run(cmd: list[str]) -> None:
    print("  $", " ".join(cmd))
    subprocess.check_call(cmd)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--py", default="3.11", help="대상(Mac) 파이썬 버전. 예: 3.11")
    ap.add_argument("--out", default=str(HERE / "transfer"), help="전송 꾸러미 폴더")
    args = ap.parse_args()
    out = Path(args.out)
    wheels = out / "wheels"
    model = out / "model"
    out.mkdir(parents=True, exist_ok=True)
    wheels.mkdir(parents=True, exist_ok=True)
    model.mkdir(parents=True, exist_ok=True)
    abi = "cp" + args.py.replace(".", "")  # 3.11 → cp311

    print("① 다운로드 도구 설치(huggingface_hub)…")
    run([sys.executable, "-m", "pip", "install", "-U", "huggingface_hub"])

    print(f"② macOS(arm64) wheel 다운로드 → {wheels}  (대상 Python {args.py})")
    req = (HERE / "requirements.txt")
    cmd = [sys.executable, "-m", "pip", "download", "-r", str(req), "-d", str(wheels),
           "--only-binary=:all:", "--python-version", args.py,
           "--implementation", "cp", "--abi", abi]
    for plat in MAC_PLATFORMS:
        cmd += ["--platform", plat]
    try:
        run(cmd)
    except subprocess.CalledProcessError:
        print("  ⚠ 일부 패키지의 macOS wheel을 한 번에 못 받았습니다.")
        print("    requirements.txt 의 버전을 조금 바꾸거나, 실패한 패키지를 개별로 받아보세요.")
        print("    (이 메시지를 그대로 전달해 주시면 명령을 맞춰 드립니다)")

    print(f"③ 모델 다운로드 → {model}  (Hugging Face 로그인/동의 필요)")
    token = os.environ.get("HF_TOKEN") or input("   HF 토큰을 붙여넣고 엔터(미입력 시 건너뜀): ").strip()
    if token:
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(REPO, local_dir=str(model), token=token)
            print("   ✔ 모델 준비 완료")
        except Exception as e:  # noqa: BLE001
            print(f"   ⚠ 모델 다운로드 실패: {e}")
            print("     라이선스 동의 여부와 토큰을 확인하세요.")
    else:
        print("   (토큰 미입력 — 모델은 나중에 받아 transfer/model 에 넣어도 됩니다)")

    print("④ 실행 스크립트 동봉…")
    for f in ["sao_server.py", "requirements.txt", "install_offline_mac.command"]:
        src = HERE / f
        if src.exists():
            shutil.copy2(src, out / f)

    print("\n✅ 완료!  이 폴더를 통째로 Mac으로 옮기세요:")
    print(f"   {out}")
    print("   Mac에서 install_offline_mac.command 를 더블클릭하면 끝입니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
