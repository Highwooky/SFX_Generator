"""로컬 AI 오디오 생성 서버(예: Stable Audio Open) 호출 클라이언트.

설계 의도(Why):
- 무거운 생성 모델(torch, 수 GB 가중치)을 가벼운 앱에 내장하지 않는다. Ollama처럼
  모델은 '로컬 서버'로 따로 띄우고, 앱은 표준 라이브러리(urllib)로만 호출한다.
- 그래야 앱은 가볍고 에어갭/패키징이 안전하며, 서버만 교체하면 어떤 모델이든 붙는다.
- 서버 스펙: GET {host}/health → {"ok":true,...}, POST {host}/generate {prompt,seconds,seed,steps}
  → {"wav_base64": "..."} (44.1kHz 등 서버가 만든 WAV 바이트의 base64).
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_AI_HOST = "http://127.0.0.1:8765"


def health(host: str = DEFAULT_AI_HOST, timeout: float = 1.0) -> dict | None:
    """서버 상태 확인. 살아있으면 정보 딕셔너리, 아니면 None."""
    try:
        with urllib.request.urlopen(f"{host.rstrip('/')}/health", timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, json.JSONDecodeError, ValueError):
        return None


def generate(prompt: str, seconds: float = 8.0, host: str = DEFAULT_AI_HOST,
             seed: int = 0, steps: int = 50, timeout: float = 600.0) -> bytes:
    """프롬프트로 오디오를 생성해 WAV 바이트를 반환. 실패 시 예외."""
    payload = json.dumps({
        "prompt": prompt, "seconds": float(seconds),
        "seed": int(seed), "steps": int(steps),
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{host.rstrip('/')}/generate", data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if "wav_base64" not in data:
        raise RuntimeError(data.get("error", "서버 응답에 wav_base64가 없습니다."))
    return base64.b64decode(data["wav_base64"])


def save_wav(data: bytes, path: Path) -> Path:
    """WAV 바이트를 파일로 저장."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path
