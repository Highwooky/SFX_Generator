"""Foley Forge 앱 아이콘 생성기.

설계 의도(Why):
- 외부 디자인 툴 없이 코드로 결정적으로 생성(재현 가능, 색/형태 조정 용이).
- 4배 슈퍼샘플링 후 축소해 가장자리/곡선을 매끄럽게(안티에일리어싱).
- macOS Big Sur+ 규약: 1024 캔버스에 약간의 여백을 둔 둥근 사각(스퀘어클).
- 모티프: 파형(=효과음) + 골드 스파크(=프롬프트로 '생성/단조'). 앱 다크테마와 통일.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SS = 4  # 슈퍼샘플 배율
SIZE = 1024 * SS
INSET = int(100 * SS)               # macOS 아이콘 여백
SQ = SIZE - INSET * 2               # 스퀘어클 한 변
RADIUS = int(SQ * 0.2237)           # Big Sur 코너 반경 비율


def _vertical_gradient(w: int, h: int, top: tuple, bottom: tuple) -> Image.Image:
    """위→아래 선형 그라데이션 이미지를 만든다."""
    t = np.linspace(0, 1, h)[:, None]
    top_a = np.array(top, dtype=np.float32)
    bot_a = np.array(bottom, dtype=np.float32)
    grad = (top_a[None, :] * (1 - t) + bot_a[None, :] * t).astype(np.uint8)
    return Image.fromarray(np.repeat(grad[:, None, :], w, axis=1), "RGB")


def _squircle_mask() -> Image.Image:
    mask = Image.new("L", (SIZE, SIZE), 0)
    d = ImageDraw.Draw(mask)
    d.rounded_rectangle([INSET, INSET, INSET + SQ, INSET + SQ], radius=RADIUS, fill=255)
    return mask


def _glow_layer(draw_fn) -> Image.Image:
    """투명 레이어에 그린 뒤 블러 → 부드러운 글로우."""
    layer = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw_fn(ImageDraw.Draw(layer))
    return layer.filter(ImageFilter.GaussianBlur(22 * SS))


def _draw_spark(draw: ImageDraw.ImageDraw, cx: int, cy: int, r: int, color) -> None:
    """4각 반짝임(별). 가로/세로로 뾰족한 마름모 두 개를 겹쳐 그린다."""
    waist = r * 0.16
    draw.polygon([(cx, cy - r), (cx + waist, cy), (cx, cy + r), (cx - waist, cy)], fill=color)
    draw.polygon([(cx - r, cy), (cx, cy - waist), (cx + r, cy), (cx, cy + waist)], fill=color)


def build_icon() -> Image.Image:
    base = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))

    # 1) 배경: 딥네이비 그라데이션을 스퀘어클로 마스킹
    grad = _vertical_gradient(SIZE, SIZE, (24, 42, 70), (8, 12, 22)).convert("RGBA")
    mask = _squircle_mask()
    base.paste(grad, (0, 0), mask)

    # 2) 중앙 뒤쪽 청록 글로우(입체감)
    def _bg_glow(d):
        d.ellipse([SIZE * 0.30, SIZE * 0.32, SIZE * 0.70, SIZE * 0.72], fill=(34, 211, 238, 130))
    glow = _glow_layer(_bg_glow)
    glow.putalpha(Image.composite(glow.getchannel("A"), Image.new("L", (SIZE, SIZE), 0), mask))
    base.alpha_composite(glow)

    # 3) 파형 막대: 좌우 대칭, 시안→그린 그라데이션. 중앙이 가장 높다.
    n = 7
    pattern = [0.34, 0.55, 0.78, 1.0, 0.78, 0.55, 0.34]
    bar_w = int(SQ * 0.072)
    gap = int(SQ * 0.045)
    total_w = n * bar_w + (n - 1) * gap
    x0 = (SIZE - total_w) // 2
    cy = SIZE // 2 + int(SQ * 0.02)
    max_h = int(SQ * 0.52)

    wave = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    wd = ImageDraw.Draw(wave)
    for i, p in enumerate(pattern):
        x = x0 + i * (bar_w + gap)
        h = int(max_h * p)
        # 막대별 색: 가장자리 시안 → 중앙 그린
        f = abs(i - n // 2) / (n // 2)
        col = (
            int(34 + (74 - 34) * (1 - f)),    # R
            int(211 + (222 - 211) * (1 - f)), # G
            int(238 + (128 - 238) * (1 - f)), # B
            255,
        )
        wd.rounded_rectangle([x, cy - h // 2, x + bar_w, cy + h // 2], radius=bar_w // 2, fill=col)
    base.alpha_composite(wave)

    # 4) 골드 스파크(생성의 상징) — 우상단, 글로우 포함
    sx, sy = int(SIZE * 0.66), int(SIZE * 0.33)
    sr = int(SQ * 0.11)
    spark_glow = _glow_layer(lambda d: _draw_spark(d, sx, sy, int(sr * 1.4), (251, 191, 36, 180)))
    base.alpha_composite(spark_glow)
    sd = ImageDraw.Draw(base)
    _draw_spark(sd, sx, sy, sr, (251, 191, 36, 255))
    _draw_spark(sd, sx, sy, int(sr * 0.5), (255, 245, 220, 255))  # 흰 코어

    # 5) 상단 살짝 밝은 시트(유리 느낌) — 가이드라인상 과하지 않게
    sheen = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    ImageDraw.Draw(sheen).rounded_rectangle(
        [INSET, INSET, INSET + SQ, INSET + int(SQ * 0.42)], radius=RADIUS, fill=(255, 255, 255, 18)
    )
    sheen.putalpha(Image.composite(sheen.getchannel("A"), Image.new("L", (SIZE, SIZE), 0), mask))
    base.alpha_composite(sheen)

    return base.resize((1024, 1024), Image.LANCZOS)


def save_all(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    icon = build_icon()
    png = out_dir / "AppIcon_1024.png"
    icon.save(png)

    # .icns 시도(Pillow) → 실패 시 .iconset PNG 세트 생성
    icns = out_dir / "AppIcon.icns"
    sizes = [16, 32, 64, 128, 256, 512, 1024]
    try:
        icon.save(icns, format="ICNS")
    except Exception as e:  # noqa: BLE001
        print(f"[icon] Pillow ICNS 저장 실패({e}) → .iconset 생성")
        iconset = out_dir / "AppIcon.iconset"
        iconset.mkdir(exist_ok=True)
        for s in [16, 32, 128, 256, 512]:
            icon.resize((s, s), Image.LANCZOS).save(iconset / f"icon_{s}x{s}.png")
            icon.resize((s * 2, s * 2), Image.LANCZOS).save(iconset / f"icon_{s}x{s}@2x.png")
    return png


if __name__ == "__main__":
    import sys

    p = save_all(Path(sys.argv[1] if len(sys.argv) > 1 else "./assets"))
    print(f"✅ {p}")
