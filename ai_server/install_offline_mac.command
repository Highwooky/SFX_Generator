#!/bin/bash
# [에어갭 Mac에서 실행] Windows에서 만든 transfer 꾸러미로 오프라인 설치/실행.
# prepare_on_windows.py 결과 폴더(transfer/) 안에 이 파일이 들어있습니다. 더블클릭하세요.

set -e
cd "$(dirname "$0")"
echo "════════════════════════════════════════════"
echo " SFX_Generator · AI 서버 오프라인 설치 (Mac)"
echo "════════════════════════════════════════════"

# 0) Mac 파이썬 버전 = Windows에서 받을 때 쓴 --py 와 같아야 합니다.
PYBIN="python3"
command -v "$PYBIN" >/dev/null 2>&1 || { echo "❌ python3 가 없습니다. python.org 설치본(.pkg)을 옮겨 설치하세요."; read -r _; exit 1; }
echo "▶ Python: $("$PYBIN" --version)"

# 1) 가상환경
if [ ! -d "venv" ]; then
  echo "▶ 가상환경 생성…"; "$PYBIN" -m venv venv
fi
# shellcheck disable=SC1091
source venv/bin/activate

# 2) 오프라인 설치(전송된 wheels 폴더에서만, 인터넷 사용 안 함)
if [ ! -d "wheels" ]; then echo "❌ wheels 폴더가 없습니다. Windows 단계가 끝났는지 확인하세요."; read -r _; exit 1; fi
echo "▶ 의존성 오프라인 설치(wheels/)…"
pip install --no-index --find-links=./wheels -r requirements.txt

# 3) 모델 경로(전송된 model 폴더). 없으면 mock 으로 실행.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
if [ -d "model" ] && [ -n "$(ls -A model 2>/dev/null)" ]; then
  echo "▶ 실제 생성 서버 시작 (로컬 모델) — http://127.0.0.1:8765  (종료 Ctrl+C)"
  exec python sao_server.py --device mps --model "$PWD/model"
else
  echo "▶ (모델 폴더 비어있음) mock 서버로 시작 — 앱 연결만 확인 가능."
  echo "   실제 음질은 Windows에서 model 폴더를 받아 넣은 뒤 다시 실행하세요."
  exec python sao_server.py --mock
fi
