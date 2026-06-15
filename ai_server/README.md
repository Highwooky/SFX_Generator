# AI 생성 서버 (Stable Audio Open)

앱과 **분리 실행**하는 로컬 오디오 생성 서버. 앱은 이 서버를 HTTP로 호출해 WAV를 받고,
합성/변주 파이프라인으로 연결한다. 일단 설치되면 **네트워크 불필요(에어갭 OK)**.

---

## A. 인터넷 되는 Mac이 있는 경우 (가장 쉬움)
`AI_설치및실행.command` **더블클릭** → venv·의존성·모델·서버까지 자동(최초 1회만 인터넷).

---

## B. 인터넷이 Windows에만 있고, Mac은 에어갭인 경우  ← 권장 절차

파이썬 패키지는 OS마다 파일이 다르므로 **Windows에서 'macOS용' 파일을 받아 옮긴다.**

### 1) Windows(인터넷)에서
1. Python 설치(가능하면 Mac과 같은 버전, 기본 3.11): https://www.python.org/downloads/
   - Mac의 버전은 Mac에서 `python3 --version` 으로 확인해 맞추세요.
2. Hugging Face 가입 → https://huggingface.co/stabilityai/stable-audio-open-1.0 에서 **라이선스 '동의'**
   → 토큰 발급: https://huggingface.co/settings/tokens
3. 이 `ai_server` 폴더에서:
   ```
   python prepare_on_windows.py --py 3.11
   ```
   - macOS(Apple Silicon)용 wheel + 모델을 `transfer/` 폴더에 모읍니다.
   - 토큰을 물으면 붙여넣기.
4. 만들어진 **`transfer/` 폴더를 통째로** USB 등으로 Mac에 복사.

### 2) Mac(에어갭)에서
- `transfer/` 안의 **`install_offline_mac.command` 더블클릭**.
  - venv 생성 → wheels로 **오프라인 설치** → 로컬 모델로 서버 실행.
- 그 다음부터는 앱의 **🤖 AI 생성**이 켜집니다(앱이 서버를 자동 감지/기동).

> Mac에 python3가 없거나 너무 구버전이면, Windows에서 python.org의 **macOS 설치본(.pkg)**도 함께
> 받아 옮겨 설치하세요(오프라인 설치 가능).

---

## 서버 직접 실행(참고)
```bash
source venv/bin/activate
python sao_server.py --device mps --model ./model   # 로컬 모델
python sao_server.py --mock                          # 모델 없이 연결만 확인
```
- 기본 주소: `http://127.0.0.1:8765`

## 라이선스 주의
Stable Audio Open 가중치/출력의 **상업 사용 조건**을 반드시 확인하세요(방송 사용 시 법무 검토 권장).
