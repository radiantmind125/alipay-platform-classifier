"""把若干张图的状态栏条拼成一张联系表（contact sheet），便于一次性肉眼核对/打标。

用于验证“仅凭状态栏条就能判定平台”，以及分辨率规则（iPhone 分辨率=iOS、短边∈{720,1080,1440}=安卓）
在真实数据上的精度。每行左侧标注序号、分辨率、分桶，右侧是该图状态栏。
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
from alipay_platform.metadata_seed import IPHONE_RESOLUTIONS  # noqa: E402

ANDROID_WIDTHS = {720, 1080, 1440}


def bucket(w: int, h: int) -> str:
    if (w, h) in IPHONE_RESOLUTIONS:
        return "ios"
    if min(w, h) in ANDROID_WIDTHS:
        return "android"
    return "ambiguous"


def _font(size: int):
    for p in ("C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/arial.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _pick_spread(items: list, n: int) -> list:
    if len(items) <= n:
        return items
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def main(argv: list[str] | None = None) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=Path, default=Path("runs/pool_20260701/inspect.jsonl"))
    ap.add_argument("--image-root", type=Path, default=Path("../raw_images"))
    ap.add_argument("--output", type=Path, default=Path("runs/pool_20260701/contact_sheet.png"))
    ap.add_argument("--per-bucket", type=int, default=6)
    ap.add_argument("--only", type=str, default="", help="只取某个分桶：ios|android|ambiguous")
    ap.add_argument("--fraction", type=float, default=0.06, help="状态栏条占图高比例")
    ap.add_argument("--width", type=int, default=1100)
    args = ap.parse_args(argv)

    recs = [json.loads(l) for l in args.jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
    ok = [r for r in recs if "error" not in r]
    by_bucket: dict[str, list] = {"ios": [], "android": [], "ambiguous": []}
    for r in ok:
        by_bucket[bucket(r["width"], r["height"])].append(r)

    names = [args.only] if args.only else ["ios", "android", "ambiguous"]
    sample: list = []
    for name in names:
        for r in _pick_spread(by_bucket[name], args.per_bucket):
            r = dict(r)
            r["bucket"] = name
            sample.append(r)

    band_w = 200
    rows: list[np.ndarray] = []
    mapping: list[dict] = []
    idx_font, small_font = _font(46), _font(22)
    for idx, r in enumerate(sample, 1):
        src = args.image_root / r["file"]
        try:
            img = load_upright_rgb(src)
        except Exception:
            continue
        h = img.shape[0]
        strip = img[: max(1, int(h * args.fraction))]
        pil = Image.fromarray(strip)
        scale = args.width / pil.width
        pil = pil.resize((args.width, max(1, int(pil.height * scale))))
        canvas = Image.new("RGB", (args.width + band_w, pil.height), (25, 25, 25))
        canvas.paste(pil, (band_w, 0))
        d = ImageDraw.Draw(canvas)
        d.text((10, 6), str(idx), fill=(255, 230, 0), font=idx_font)
        d.text((70, 12), f'{r["width"]}x{r["height"]}', fill=(200, 200, 200), font=small_font)
        d.text((70, 40), r["bucket"], fill=(120, 200, 255), font=small_font)
        rows.append(np.asarray(canvas))
        mapping.append({"row": idx, "file": r["file"], "res": f'{r["width"]}x{r["height"]}', "bucket": r["bucket"]})

    maxw = max(r.shape[1] for r in rows)
    sep = np.full((4, maxw, 3), 255, np.uint8)
    stacked: list[np.ndarray] = []
    for r in rows:
        if r.shape[1] < maxw:
            r = np.concatenate([r, np.full((r.shape[0], maxw - r.shape[1], 3), 25, np.uint8)], axis=1)
        stacked.extend([r, sep])
    sheet = np.concatenate(stacked, axis=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sheet).save(args.output)

    print(f"contact sheet: {args.output}  shape={sheet.shape}")
    for m in mapping:
        print(f'  行{m["row"]:>2}  {m["bucket"]:<10} {m["res"]:<10} {m["file"]}')


if __name__ == "__main__":
    main()
