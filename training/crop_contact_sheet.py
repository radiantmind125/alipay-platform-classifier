r"""把每组裁块抽几张拼成一张大图 —— 用眼睛直接看每组切的是不是金额(只读)。

为什么要有这个
--------------
前面所有判定都是**推断链**: 源图版式 -> 定位器 -> 框尺寸 -> 裁块尺寸。链条越长越可能有错,
而且 `apiwan` `apiseed` 这两组因为源图丢了, 根本没进推断链, 只能靠 §214 的账推。

**眼睛是最短的验证路径。** 金额裁块一眼就是一串大数字; 红包促销卡裁块一眼就是
"到店支付红包 ¥0.6" 那种小字卡片。拼成一张图, 谁对谁错一秒就看出来。

看 `nature` 那一侧(原始未改动的裁块)最清楚 —— `ai` 那侧被重绘过, 反而糊。

用法
----
  python training/crop_contact_sheet.py --crops D:\localcrops --out D:\probe\crop_sheet.png
  python training/crop_contact_sheet.py --crops D:\localcrops --out D:\probe\sheet_ai.png --side ai
  python training/crop_contact_sheet.py --crops D:\localcrops --out D:\probe\sheet.png --tags apiwan apiseed --per-tag 8

**只读**: 只读裁块, 只写你指定的那一张 png。
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

_EXTS = {".jpg", ".jpeg", ".png"}


def tag_of(nm: str) -> str:
    stem = Path(nm).stem
    if not stem.startswith("crop_"):
        return "(命名不认识)"
    return stem[len("crop_"):].rpartition("_")[0] or "(命名不认识)"


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="每组裁块抽样拼图, 用眼睛验(只读)")
    ap.add_argument("--crops", type=Path, required=True, help="含 ai/ 与 nature/ 的目录")
    ap.add_argument("--out", type=Path, required=True, help="输出 png")
    ap.add_argument("--side", default="nature", choices=["nature", "ai"],
                    help="看哪一侧。默认 nature(原始裁块, 最清楚)")
    ap.add_argument("--per-tag", type=int, default=5, help="每组抽几张")
    ap.add_argument("--tags", nargs="*", default=None, help="只看这几组(默认全部)")
    ap.add_argument("--cell-h", type=int, default=72, help="每张缩到多高")
    ap.add_argument("--cell-w", type=int, default=300, help="每张最宽多少")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = args.crops / args.side
    if not d.is_dir():
        raise SystemExit(f"目录不存在: {d}")
    by_tag: dict[str, list[Path]] = defaultdict(list)
    for p in d.iterdir():
        if p.is_file() and p.suffix.lower() in _EXTS:
            by_tag[tag_of(p.name)].append(p)
    if args.tags:
        keep = set(args.tags)
        unknown = keep - set(by_tag)
        if unknown:
            raise SystemExit(f"这几组不存在: {', '.join(sorted(unknown))}")
        by_tag = {k: v for k, v in by_tag.items() if k in keep}
    if not by_tag:
        raise SystemExit("没找到裁块")

    order = sorted(by_tag, key=lambda t: -len(by_tag[t]))
    label_w = 210
    gap = 6
    row_h = args.cell_h + gap
    sheet_w = label_w + args.per_tag * (args.cell_w + gap)
    sheet_h = row_h * len(order) + gap
    sheet = Image.new("RGB", (sheet_w, sheet_h), (245, 245, 245))
    drw = ImageDraw.Draw(sheet)

    rng = random.Random(args.seed)
    for r, tag in enumerate(order):
        y = gap + r * row_h
        drw.rectangle([0, y - 3, sheet_w, y + args.cell_h + 2],
                      fill=(255, 255, 255) if r % 2 == 0 else (236, 236, 236))
        drw.text((8, y + args.cell_h // 2 - 6), f"{tag}  n={len(by_tag[tag])}", fill=(0, 0, 0))
        files = by_tag[tag]
        pick = files if len(files) <= args.per_tag else rng.sample(files, args.per_tag)
        for c, p in enumerate(pick):
            try:
                im = Image.open(p).convert("RGB")
            except Exception:
                continue
            w, h = im.size
            s = min(args.cell_h / max(1, h), args.cell_w / max(1, w))
            im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
            x = label_w + c * (args.cell_w + gap)
            sheet.paste(im, (x, y + (args.cell_h - im.size[1]) // 2))
            drw.rectangle([x - 1, y - 1, x + im.size[0], y + args.cell_h],
                          outline=(200, 200, 200))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out)
    print(f"{len(order)} 组, 每组抽 {args.per_tag} 张, 看的是 {args.side} 那一侧")
    print(f"-> {args.out}  ({sheet_w}x{sheet_h})")
    print()
    print("怎么看:")
    print("  一串大数字(金额)            -> 这组切对了")
    print("  小字卡片/优惠券/红包那种     -> **这组切到红包促销卡了, 该剔**")
    print("  白底深字 = 白图金额; 蓝底白字 = 蓝图金额, 两种都对")


if __name__ == "__main__":
    main()
