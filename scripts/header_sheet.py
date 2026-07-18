"""把若干金标图的“对勾 + 转账成功”表头区域裁出来拼成一张对照图，用于验证经理的判据：
勾和“转账成功”是否在同一条线上（苹果=在，安卓=不在，没有勾=假图）。

按平台排序（先 iOS 后安卓），每行标注平台/分辨率。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from alipay_platform.cache_regions import load_upright_rgb  # noqa: E402


def _font(size: int):
    for p in ("C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/arial.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()


def main(argv: list[str] | None = None) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", type=Path, default=Path("gold/gold_seed_v1.json"))
    ap.add_argument("--image-root", type=Path, default=Path("../raw_images"))
    ap.add_argument("--output", type=Path, default=Path("runs/pool_20260701/header_sheet.png"))
    ap.add_argument("--per-class", type=int, default=5)
    ap.add_argument("--y0", type=float, default=0.045)
    ap.add_argument("--y1", type=float, default=0.20)
    ap.add_argument("--x0", type=float, default=0.10)
    ap.add_argument("--x1", type=float, default=0.90)
    ap.add_argument("--width", type=int, default=1000)
    args = ap.parse_args(argv)

    labels = json.loads(args.gold.read_text(encoding="utf-8"))["labels"]
    by = {"ios": [], "android": []}
    for row in labels:
        by[row["platform"]].append(row)
    selected = by["ios"][: args.per_class] + by["android"][: args.per_class]

    band_w = 190
    rows: list[np.ndarray] = []
    idx_font, small = _font(40), _font(22)
    for i, row in enumerate(selected, 1):
        src = args.image_root / row["file"]
        try:
            img = load_upright_rgb(src)
        except Exception:
            continue
        h, w = img.shape[:2]
        crop = img[int(h * args.y0) : int(h * args.y1), int(w * args.x0) : int(w * args.x1)]
        pil = Image.fromarray(crop)
        scale = args.width / pil.width
        pil = pil.resize((args.width, max(1, int(pil.height * scale))))
        canvas = Image.new("RGB", (args.width + band_w, pil.height), (20, 20, 20))
        canvas.paste(pil, (band_w, 0))
        d = ImageDraw.Draw(canvas)
        d.text((8, 6), str(i), fill=(255, 230, 0), font=idx_font)
        d.text((8, 52), row["platform"], fill=(120, 220, 255), font=small)
        d.text((8, 80), row.get("res", ""), fill=(180, 180, 180), font=small)
        rows.append(np.asarray(canvas))
        print(f'  行{i:>2} {row["platform"]:<8} {row.get("res",""):<10} {row["file"]}')

    maxw = max(r.shape[1] for r in rows)
    sep = np.full((5, maxw, 3), 255, np.uint8)
    stacked: list[np.ndarray] = []
    for r in rows:
        if r.shape[1] < maxw:
            r = np.concatenate([r, np.full((r.shape[0], maxw - r.shape[1], 3), 20, np.uint8)], axis=1)
        stacked.extend([r, sep])
    sheet = np.concatenate(stacked, axis=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sheet).save(args.output)
    print(f"header sheet: {args.output}  shape={sheet.shape}")


if __name__ == "__main__":
    main()
