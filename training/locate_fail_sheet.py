r"""把"金额定位得到"和"定位不到"的图各抽几张拼成一张大图 —— 用眼睛看清楚差在哪(只读)。

为什么要有这个
--------------
`D:\download2\OtherImages` 的定位率只有 **31.8%**(635/2000), 而真图池 `genuine_20k` 是 **82.6%**。
同样是白图, 差了两倍多。这个差决定了对外能报的覆盖率:

    进件像 genuine_20k  -> 覆盖率约 0.826 x 0.905 = **75%**
    进件像 OtherImages  -> 覆盖率约 0.318 x 0.905 = **29%**

**光看数字分不出是哪种**, 因为两种情况的定位率长得一模一样:

1. 定位不到的那 68% **就是账单详情页**, 只是定位器抓不住 -> **定位器有大缺口, 修它比调模型值钱得多**
2. 定位不到的那 68% 根本不是账单详情页(账单列表/收款码/设置页...) -> **这个池子不代表进件**,
   覆盖率仍按 genuine_20k 的 82.6% 算

**只有把两组图摆在一起看才分得清。** 这就是这个脚本。

用法
----
  python training/locate_fail_sheet.py --src D:\download2\OtherImages --out D:\probe\locate_fail.png
  python training/locate_fail_sheet.py --src D:\probe\genuine_20k --out D:\probe\locate_fail_genuine.png --per 10

**只读**: 只读图片, 只写你指定的那一张 png。
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))

from locate_blue import is_blue_page, locate_amount_auto  # noqa: E402

_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="定位成功 vs 定位失败 的图各抽几张拼一起看(只读)")
    ap.add_argument("--src", type=Path, required=True, help="要看的图片目录")
    ap.add_argument("--out", type=Path, required=True, help="输出 png")
    ap.add_argument("--per", type=int, default=8, help="每组抽几张")
    ap.add_argument("--max-scan", type=int, default=400, help="最多试多少张来凑够两组")
    ap.add_argument("--cell-w", type=int, default=190, help="每张缩到多宽")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    names: list[str] = []
    with os.scandir(args.src) as it:
        for e in it:
            if e.is_file() and os.path.splitext(e.name)[1].lower() in _EXT:
                names.append(e.path)
    if not names:
        raise SystemExit(f"{args.src} 里没有图片")
    random.Random(args.seed).shuffle(names)

    ok: list[tuple[Image.Image, tuple]] = []      # (缩略图, 金额框换算到缩略图坐标)
    bad: list[Image.Image] = []
    tried = blue_n = 0
    for p in names:
        if (len(ok) >= args.per and len(bad) >= args.per) or tried >= args.max_scan:
            break
        tried += 1
        try:
            im = Image.open(p)
            im.draft("RGB", (im.size[0] // 2, im.size[1] // 2))   # 大图先半解, 快一倍
            a = np.asarray(im.convert("RGB"))
        except Exception:
            continue
        blue_n += int(is_blue_page(a))
        loc, _page = locate_amount_auto(a)
        h, w = a.shape[:2]
        s = args.cell_w / max(1, w)
        th = Image.fromarray(a).resize((args.cell_w, max(1, int(h * s))), Image.LANCZOS)
        if loc and len(ok) < args.per:
            box = (int(loc[0] * s), int(loc[1] * s), int(loc[2] * s), int(loc[3] * s))
            ok.append((th, box))
        elif not loc and len(bad) < args.per:
            bad.append(th)

    if not ok and not bad:
        raise SystemExit("一张都没读成")
    cell_h = max([t.size[1] for t, _ in ok] + [t.size[1] for t in bad] + [1])
    gap, lab = 8, 26
    cols = max(len(ok), len(bad), 1)
    sheet_w = gap + cols * (args.cell_w + gap)
    sheet_h = 2 * (cell_h + lab + gap) + gap
    sheet = Image.new("RGB", (sheet_w, sheet_h), (250, 250, 250))
    drw = ImageDraw.Draw(sheet)

    for r, (title, items) in enumerate([
            (f"LOCATED  定位得到  n={len(ok)}  (红框=金额区)", ok),
            (f"NOT LOCATED  定位不到  n={len(bad)}", bad)]):
        y0 = gap + r * (cell_h + lab + gap)
        drw.rectangle([0, y0, sheet_w, y0 + lab], fill=(225, 235, 225) if r == 0 else (245, 225, 225))
        drw.text((10, y0 + 7), title, fill=(0, 0, 0))
        for c, it in enumerate(items):
            th, box = it if r == 0 else (it, None)
            x = gap + c * (args.cell_w + gap)
            sheet.paste(th, (x, y0 + lab))
            if box:
                drw.rectangle([x + box[0], y0 + lab + box[1], x + box[2], y0 + lab + box[3]],
                              outline=(230, 30, 30), width=2)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out)
    print(f"试了 {tried} 张 | 定位得到 {len(ok)} 张, 定位不到 {len(bad)} 张 | 其中蓝图 {blue_n} 张")
    print(f"-> {args.out}  ({sheet_w}x{sheet_h})")
    print()
    print("怎么看下面那一排(定位不到的):")
    print("  **也是账单详情页, 只是没框住金额** -> 定位器有大缺口, **修定位器比调模型值钱得多**")
    print("  **根本不是账单详情页**(账单列表/收款码/设置页/聊天页...) -> 这个池子不代表进件,")
    print("     覆盖率仍按 genuine_20k 的 82.6% 算")


if __name__ == "__main__":
    main()
