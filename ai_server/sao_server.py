#!/usr/bin/env python3
"""Stable Audio Open 로컬 생성 서버 (앱과 분리 실행).

앱(SFX_Generator)과 별개의 가상환경에서 실행한다. 앱은 이 서버를 HTTP로 호출해
WAV를 받아 합성/변주 파이프라인에 연결한다(에어갭에서도 동작 — 네트워크 불필요).

엔드포인트:
  GET  /health   → {"ok":true,"mode":"real|mock","model":...,"device":...}
  POST /generate {"prompt","seconds","seed","steps"} → {"wav_base64": "..."}

실행:
  # 1) 통합 테스트용(모델 없이, 즉시): 앱 연결만 확인
  python sao_server.py --mock
  # 2) 실제 생성(Apple Silicon):
  python sao_server.py --device mps --steps 60
      (최초 1회 모델 가중치 필요 — 인터넷 되는 PC에서 받아 옮기거나 HF 로그인 후 캐시)

의존성(real 모드): torch, diffusers>=0.27, soundfile  (mock 모드는 numpy, soundfile만)
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import soundfile as sf

# 전역(서버 시작 시 1회 로드)
_PIPE = None
_CFG = {"mode": "mock", "model": None, "device": "cpu", "steps": 50, "sr": 44100}


def _load_model(model: str, device: str) -> None:
    """Stable Audio Open 파이프라인 로드(real 모드). 실패 시 예외."""
    global _PIPE
    import torch  # noqa: PLC0415
    from diffusers import StableAudioPipeline  # noqa: PLC0415

    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
    pipe = StableAudioPipeline.from_pretrained(model, torch_dtype=dtype)
    pipe = pipe.to(device)
    _PIPE = pipe
    try:
        _CFG["sr"] = int(pipe.vae.sampling_rate)
    except Exception:  # noqa: BLE001
        _CFG["sr"] = 44100


def _gen_real(prompt: str, seconds: float, seed: int, steps: int) -> tuple[np.ndarray, int]:
    """실제 모델로 생성 → (오디오[samples,ch] float32, sr)."""
    import torch  # noqa: PLC0415

    gen = torch.Generator("cpu").manual_seed(int(seed))
    out = _PIPE(
        prompt,
        negative_prompt="Low quality, distorted, clipping.",
        num_inference_steps=int(steps),
        audio_end_in_s=float(seconds),
        num_waveforms_per_prompt=1,
        generator=gen,
    ).audios
    audio = out[0].T.float().cpu().numpy()  # (samples, channels)
    return audio.astype(np.float32), _CFG["sr"]


def _gen_mock(prompt: str, seconds: float, seed: int, steps: int) -> tuple[np.ndarray, int]:
    """모델 없이 프롬프트 해시로 placeholder 합성(통합 테스트용)."""
    sr = 44100
    n = int(sr * max(0.3, min(seconds, 20)))
    rng = np.random.default_rng((abs(hash(prompt)) + seed) % (2**32))
    t = np.arange(n) / sr
    base_hz = 180 + (abs(hash(prompt)) % 600)
    tone = np.sin(2 * np.pi * base_hz * t) * np.exp(-t / (seconds * 0.5))
    noise = rng.standard_normal(n) * 0.2 * np.exp(-t / (seconds * 0.3))
    sig = (tone + noise)
    sig = sig / (np.max(np.abs(sig)) or 1.0) * 0.8
    stereo = np.stack([sig, sig], axis=1).astype(np.float32)
    return stereo, sr


def _wav_b64(audio: np.ndarray, sr: int) -> str:
    buf = io.BytesIO()
    sf.write(buf, audio, sr, subtype="PCM_24", format="WAV")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):  # 조용히
        pass

    def _send(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.rstrip("/") == "/health":
            self._send(200, {"ok": True, "mode": _CFG["mode"], "model": _CFG["model"], "device": _CFG["device"]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/generate":
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length).decode("utf-8"))
            prompt = str(req.get("prompt", "")).strip()
            if not prompt:
                self._send(400, {"error": "prompt 비어있음"})
                return
            seconds = float(req.get("seconds", 8.0))
            seed = int(req.get("seed", 0))
            steps = int(req.get("steps", _CFG["steps"]))
            gen = _gen_real if _CFG["mode"] == "real" else _gen_mock
            audio, sr = gen(prompt, seconds, seed, steps)
            self._send(200, {"wav_base64": _wav_b64(audio, sr), "sr": sr})
        except Exception as e:  # noqa: BLE001
            self._send(500, {"error": str(e)})


def main() -> int:
    ap = argparse.ArgumentParser(description="Stable Audio Open 로컬 생성 서버")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--mock", action="store_true", help="모델 없이 placeholder 생성(통합 테스트용)")
    ap.add_argument("--model", default="stabilityai/stable-audio-open-1.0")
    ap.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    ap.add_argument("--steps", type=int, default=50)
    args = ap.parse_args()

    _CFG["steps"] = args.steps
    if args.mock:
        _CFG["mode"] = "mock"
        print(f"🤖 [mock] 생성 서버 — http://{args.host}:{args.port} (모델 미사용)")
    else:
        _CFG.update(mode="real", model=args.model, device=args.device)
        print(f"⏳ 모델 로딩 중: {args.model} ({args.device}) …")
        _load_model(args.model, args.device)
        print(f"🤖 [real] 생성 서버 — http://{args.host}:{args.port} · {args.model} · sr={_CFG['sr']}")

    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
