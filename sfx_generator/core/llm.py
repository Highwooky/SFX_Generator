"""로컬 Ollama 연동 클라이언트.

설계 의도(Why):
- 별도 SDK 없이 stdlib urllib만으로 호출한다(의존성 0, 에어갭에서 로컬 Ollama만 있으면 동작).
- format="json"으로 모델이 JSON만 뱉도록 강제해 레시피 파싱 안정성을 높인다.
- 모델 미지정 시 설치된 모델 중 우선순위(qwen → gpt-oss → 그 외)로 자동 선택한다.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Callable, Optional

DEFAULT_HOST = "http://localhost:11434"


class OllamaError(Exception):
    pass


def _post(url: str, payload: dict, timeout: float) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise OllamaError(f"Ollama 연결 실패({url}): {e}") from e


def list_models(host: str = DEFAULT_HOST, timeout: float = 5.0) -> list[str]:
    """설치된 모델 이름 목록. 실패 시 OllamaError."""
    try:
        with urllib.request.urlopen(f"{host.rstrip('/')}/api/tags", timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise OllamaError(f"Ollama 모델 목록 조회 실패: {e}") from e
    return [m["name"] for m in data.get("models", [])]


def _auto_pick(models: list[str]) -> str:
    """모델 자동 선택: qwen 계열 우선, 다음 gpt-oss, 그 외 첫 번째."""
    for pref in ("qwen", "gpt-oss"):
        for m in models:
            if pref in m.lower():
                return m
    if not models:
        raise OllamaError("설치된 Ollama 모델이 없습니다.")
    return models[0]


class OllamaClient:
    def __init__(
        self, host: str = DEFAULT_HOST, model: Optional[str] = None, timeout: float = 120.0
    ) -> None:
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.model = model or _auto_pick(list_models(host, timeout=5.0))

    def complete(self, system: str, user: str) -> str:
        """system/user 메시지로 chat 호출. JSON 강제. assistant content 문자열 반환."""
        payload = {
            "model": self.model,
            "stream": False,
            "format": "json",  # 모델이 JSON만 출력하도록 강제
            "options": {"temperature": 0.4},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        data = _post(f"{self.host}/api/chat", payload, self.timeout)
        content = data.get("message", {}).get("content", "")
        if not content:
            raise OllamaError("Ollama 응답이 비어 있습니다.")
        return content


def make_ollama_llm(
    host: str = DEFAULT_HOST, model: Optional[str] = None
) -> Callable[[str, str], str]:
    """Interpreter(llm=...)에 주입할 콜러블 생성. (system,user)->str"""
    client = OllamaClient(host=host, model=model)
    return client.complete
