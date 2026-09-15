"""把一批图拼成一张联系表(contact sheet), 一眼看清整批是什么货色。

用途: 归集出来的"拼音图"里到底有多少是**账单详情回单**, 有多少是锁屏/聊天/
主屏这类跟订单号无关的截图。抽样拼一张图, 比一张张点开快得多。

    python make_contact_sheet.py --src <目录> --out sheet.png --count 24
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build(src: Path, out: Path, count: int, cols: int, thumb_w: int, seed: int) -> int:
    files = [p for p in src.iterdir() if p.is_file() and p.suffix.lower() in EXTS]
    if not files:
        print(f"{src} 里没有图")
        return 0

    random.seed(seed)
    picked = random.sample(files, min(count, len(files)))

    # 手机截图都是竖的, 按 9:19.5 左右留格子
    thumb_h = int(thumb_w * 2.1)
    label_h = 18
    cell_h = thumb_h + label_h
    rows = (len(picked) + cols - 1) // cols

    sheet = Image.new("RGB", (cols * thumb_w, rows * cell_h), (235, 235, 235))
    draw = ImageDraw.Draw(sheet)

    for i, p in enumerate(picked):
        cx = (i % cols) * thumb_w
        cy = (i // cols) * cell_h
        try:
            im = Image.open(p)
            im.draft("RGB", (thumb_w * 2, thumb_h * 2))   # jpeg 快速降采样
            im = im.convert("RGB")
            im.thumbnail((thumb_w - 4, thumb_h - 4), Image.LANCZOS)
            sheet.paste(im, (cx + 2, cy + 2))
        except Exception as e:                            # 坏图不要整批崩掉
            draw.text((cx + 6, cy + 6), f"坏图\n{e.__class__.__name__}", fill=(200, 0, 0))

        draw.rectangle([cx, cy, cx + thumb_w - 1, cy + cell_h - 1], outline=(150, 150, 150))
        draw.text((cx + 4, cy + thumb_h + 3), f"{i:02d}", fill=(0, 0, 0))

    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(f"{len(picked)} 张 -> {out}  ({sheet.size[0]}x{sheet.size[1]})")
    for i, p in enumerate(picked):
        print(f"  {i:02d}  {p.name}")
    return len(picked)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--count", type=int, default=24)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--thumb-w", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    build(a.src, a.out, a.count, a.cols, a.thumb_w, a.seed)


if __name__ == "__main__":
    main()
