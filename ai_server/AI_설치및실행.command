#!/bin/bash
# SFX_Generator — AI 생성 서버 원클릭 설치/실행 (macOS, Apple Silicon)
# 더블클릭하면: 가상환경 생성 → 의존성 설치 → (최초 1회) 모델 다운로드 → 서버 실행.
# 두 번째부터는 바로 서버만 켜집니다. 종료는 이 창에서 Ctrl+C.

set -e
cd "$(dirname "$0")"
echo "════════════════════════════════════════════"
echo " SFX_Generator · AI 생성 서버 (Stable Audio Open)"
echo "════════════════════════════════════════════"

# 1) python3 확인
if ! command -v python3 >/dev/null 2>&1; then
  echo "❌ python3 가 필요합니다.  먼저 설치하세요:  brew install python"
  echo "   (Homebrew가 없으면 https://brew.sh 참고)"
  read -r -p "엔터를 누르면 종료합니다…" _; exit 1
fi

# 2) 가상환경(최초 1회)
if [ ! -d "venv" ]; then
  echo "▶ 가상환경 생성 중…"
  python3 -m venv venv
fi
# shellcheck disable=SC1091
source venv/bin/activate
python -m pip install --upgrade pip >/dev/null 2>&1 || true

# 3) 의존성(이미 설치되어 있으면 빠르게 통과)
if ! python -c "import torch, diffusers, soundfile" >/dev/null 2>&1; then
  echo "▶ 의존성 설치 중… (최초 1회, 수 분 소요 — 인터넷 필요)"
  pip install -r requirements.txt
else
  echo "✔ 의존성 이미 설치됨"
fi

# 4) 모델 사전 다운로드(최초 1회). 게이트 모델이라 동의+토큰이 필요할 수 있음.
echo "▶ 모델 확인/다운로드 중… (최초 1회, 수 GB — 인터넷 필요)"
python - << 'PYEOF' || true
try:
    from huggingface_hub import snapshot_download
    snapshot_download("stabilityai/stable-audio-open-1.0")
    print("✔ 모델 준비 완료")
except Exception as e:
    print("⚠ 모델 자동 다운로드 실패:", e)
    print("  → 다음을 한 번 실행해 로그인/동의 후 다시 시도하세요:")
    print("     source venv/bin/activate && huggingface-cli login")
    print("  (Hugging Face에서 stable-audio-open-1.0 라이선스에 먼저 동의해야 합니다)")
    print("  모델 없이 연결만 확인하려면 --mock 으로 실행됩니다.")
PYEOF

# 5) 서버 실행 (모델 있으면 실제, 없으면 mock으로 폴백)
if python -c "from huggingface_hub import snapshot_download; snapshot_download('stabilityai/stable-audio-open-1.0', local_files_only=True)" >/dev/null 2>&1; then
  echo "▶ 실제 생성 서버 시작 — http://127.0.0.1:8765  (종료: Ctrl+C)"
  exec python sao_server.py --device mps
else
  echo "▶ (모델 미확인) mock 서버로 시작 — 앱 연결만 확인 가능. http://127.0.0.1:8765"
  exec python sao_server.py --mock
fi
