# SFX_Generator — 방송 효과음 생성기

프롬프트 한 줄로 **저작권 걱정 없는 방송 효과음**을 만드는 완전 오프라인 macOS(Apple Silicon) 데스크톱 도구.
입력을 사운드 **레시피(JSON)**로 변환하고, DSP 엔진이 **보유 라이브러리 변형·결합 + 절차적 합성**으로 실행해
24bit/48kHz WAV로 출력합니다. AI 생성모델을 쓰지 않아 저작권에서 자유롭고, 외부 API·네트워크 없이 동작합니다.

> 제작: JTBC Mediatech – Production J – Postproduction – Yu byungwook

---

## 주요 기능

**생성 방식**
- **프롬프트 해석** — 한글 자연어 → 레시피. 룰 엔진 기본, 로컬 **Ollama**(qwen·gpt-oss) 연결 시 더 정확. 실패 시 룰로 자동 폴백.
- **LLM 프롬프트 확장** — 모호한 한 줄을 상세 사운드 디자인 브리프로 펼친 뒤 생성(`--expand` / GUI 옵션).
- **내부 템플릿(자동 알고리즘)** — 두둥·짜잔·와장창·두근두근·정적 등 방송 자막 큐는 내부 다층 레시피로 자동 처리(목록 비노출).
- **원본 가공(변주·효과)** — 가지고 있는 음원을 **실제 소스로 받아** 변형/보강 레이어를 입혀 변주나 어울리는 효과를 만듭니다(공간감·어둡게·밝게·리버스·더블·텍스처·서브 보강 등). 파일 드래그&드롭 지원.
- **AI 생성 연동(선택)** — 로컬 생성 서버(**Stable Audio Open**)로 만든 소리를 받아, 곧바로 24/48·라우드니스·[SFX]로 마스터링하고 변주까지 잇습니다. 모델은 앱과 분리 실행(에어갭 OK) — `ai_server/` 참고.
- **컨셉 추론(자막 효과음 팩)** — '예능/교양에서 쓸만한 자막 효과음'처럼 카테고리로 요청하면, 그 분위기에 맞는 **대표 큐를 한 세트로** 생성합니다(예능=두둥·짜잔·두구두구·띠용·반짝·와장창·땡·정적 / 교양=전환·포인트·차임·인서트·은은한 반짝·낮은 강조).

**합성 엔진(라이브러리 없이도 동작)**
- `sub_impact` · `tone` · `noise` · `whoosh` · `riser`
- `modal`(금속/나무/유리 타격) · `pluck`(현/튕김) · `fm`(전자/레이저) · `wind` · `rain` · `fire`

**DSP 변형**
- pitch · stretch · reverse · gain · fade · filter · eq · reverb · distortion · chorus · delay · bitcrush · normalize
- `granular`(짧은 샘플 → 텍스처/구름) · `spectral`(freeze/stretch/blur → 드론/패드) · `envelope`(엔벨로프 폴로잉)

**출력·워크플로**
- **길이(초) 지정** — 정확한 길이로 출력(GUI "길이 지정" 체크 + 스핀박스 / CLI `--length 2`).
- **라우드니스 타깃** — -16 일반 / -23 EBU R128 / -24 ATSC (GUI 콤보 / CLI `--lufs`).
- **변주 일괄 생성** — 한 번에 여러 변주(seed 차이)를 만들어 결과 목록에서 골라 쓰기.
- **재현성** — seed 기반, 같은 설정이면 동일 결과. 레시피 JSON 직접 편집 가능.
- **라이브러리** — 폴더 **경로 유지**·색인 **진행률**·**🔄 새로고침**·디스크 캐시로 재로딩 가속·길이/밝기/타격성 **자동 태깅**.
- **라이브러리 브라우저(🔎 찾아보기)** — 검색 → 순위 표시 → **미리듣기** → 여러 샘플 **합치기(레이어 콜라주)** 또는 원본 가공으로 보내기. **진짜 샘플**로 만들어 가장 정확.
- **스템 분할(✂️)** — 효과 스템(SFX만 있는 트랙)을 무음 기준으로 **개별 효과음으로 잘라 라이브러리에 자동 추가**. '내가 쓰던 소리'를 그대로 재료화.
- **조절 노브** — 생성된 소리를 밝기·피치·어택·공간감·무게·거칠기 슬라이더로 **실시간 다듬기**(상상에 수렴).

---

## 설치 (사용자)

1. repo의 **Releases** 페이지에서 `SFX_Generator-macos-arm64.dmg`(권장) 또는 `.zip`을 받습니다.
   - ⚠️ Actions의 *Artifacts*에서 받으면 zip이 이중 포장돼 압축 해제가 안 될 수 있습니다. 반드시 **Releases**에서 받으세요.
2. `.dmg`는 더블클릭 후 앱을 드래그, `.zip`은 더블클릭으로 풀립니다.
3. 미서명 앱이라 첫 실행 시 "개발자 확인 불가"가 뜨면 **우클릭 → 열기**. 그래도 안 되면:
   ```bash
   xattr -dr com.apple.quarantine "/Applications/SFX_Generator.app"
   ```

---

## 사용법

### GUI
```bash
python run_gui.py        # 또는  python -m sfx_generator.gui
```
1. (선택) 라이브러리 폴더·출력 폴더 지정
2. 프롬프트 입력 → **생성**  /  또는 **원본 가공**: 음원을 끌어다 놓고(또는 📂 원본 선택) 스타일 고른 뒤 🎚 가공 생성
3. (선택) **길이 지정**·**라우드니스**·**변주 개수** 설정
4. 파형 미리듣기·재생 → 결과 목록에서 파일 확인

### CLI
```bash
# 프롬프트
python -m sfx_generator.cli --prompt "웅장하고 묵직한 폭발" --out ./out
# 길이 2초 + EBU R128 라우드니스 + 변주 5개
python -m sfx_generator.cli --prompt "거센 바람" --length 2 --lufs -23 --variations 5 --out ./out
# 로컬 Ollama + 프롬프트 확장
python -m sfx_generator.cli --prompt "밝은 자막 효과" --llm auto --expand --out ./out
# 원본 가공(변주·효과)
python -m sfx_generator.cli --process orig.wav --style "공간감(홀)" --variations 3 --out ./out
# 컨셉 팩(카테고리 추론)
python -m sfx_generator.cli --prompt "예능에서 쓸만한 자막 효과음" --variations 6 --out ./out
# 레시피 파일 / 프리셋
python -m sfx_generator.cli recipe.json --out ./out
python -m sfx_generator.cli --list-presets
```

---

## AI 생성 연동 (Stable Audio Open)
무거운 모델은 앱에 넣지 않고 **로컬 서버로 분리 실행**합니다(Ollama와 같은 방식). 앱은 그 서버를 호출해
소리를 받고, 합성/변주 파이프라인으로 연결합니다.

**가장 쉬운 방법 — 원클릭:** `ai_server/AI_설치및실행.command` 를 **더블클릭**하면 venv·의존성·모델·서버까지 자동입니다(최초 1회만 인터넷 필요). 또는 앱에서 **⚙️ AI 서버 설치/시작** 버튼.

수동 실행을 원하면:
```bash
cd ai_server
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python sao_server.py --device mps        # 실제 생성
# 또는 모델 없이 연결만 확인:  python sao_server.py --mock
```

> 최초 1회는 인터넷이 필요합니다(의존성+모델 다운로드, HF 라이선스 동의·로그인). 이후엔 오프라인 동작.
> 완전 에어갭이고 **인터넷이 Windows에만** 있다면: `ai_server/prepare_on_windows.py` 로 Windows에서 macOS용 wheel+모델을
> 받아 옮긴 뒤, Mac에서 `install_offline_mac.command` 로 오프라인 설치합니다. 자세한 절차는 `ai_server/README.md`.
서버가 떠 있으면 앱에서 **🤖 AI 생성**이 활성화됩니다. 프롬프트로 생성 → 자동 마스터링 →(옵션) 변주.
⚠️ Stable Audio Open 가중치/출력의 **상업 사용 조건**을 확인하세요(방송 사용 시 법무 검토 권장). 자세한 건 `ai_server/README.md`.

## 라이브러리 확장
보유 음원이 적으면 합법적 무료 소스나 자체 녹음으로 키우세요 — [SOURCES.md](SOURCES.md) 참고.
폴더 구조로 분류하면 폴더명이 태그가 되고, 자동 태깅이 길이/밝기/타격성을 더해 검색이 잘 됩니다.

---

## 개발 / 빌드
- 개발: Windows 등에서 코딩 → GitHub push
- 빌드: **GitHub Actions(macos-14, Apple Silicon)** 가 테스트 → PyInstaller로 `.app` → `.zip`+`.dmg` 생성
- 배포: `v*` 태그를 푸시하면 Releases에 `.dmg`/`.zip` 자동 첨부
  ```bash
  git tag v1.2.0 && git push origin v1.2.0
  ```

## 테스트
```bash
python sfx_generator/test_engine.py
python sfx_generator/test_library.py
python sfx_generator/test_interpreter.py
python sfx_generator/test_llm.py
python sfx_generator/test_extensions.py
QT_QPA_PLATFORM=offscreen python sfx_generator/test_gui_smoke.py
```

## 라이선스
MIT — 자세한 내용은 [LICENSE](LICENSE).
