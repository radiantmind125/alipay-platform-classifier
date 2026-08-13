r"""把真图池里分数最高的那几张拼成一张图, 让人**亲眼看一遍**(只读)。

为什么这一步值得做
------------------
自动拒的分数线是**第 k 高分**决定的, 而 k = 池子张数 / 预算。
97,324 张 + 1/5000 预算 -> k=19 -> **整条线由第 20 高的那张图定死**。
也就是说: **二十张图决定了整个系统的工作点。**

`watermark_scan` 能非循环地认出**带水印/带 AIGC 元数据**的假图(这轮认出 38 张),
但**去了水印又清了元数据的假图它抓不到**。剩下那些高分图到底是
"难的真图"还是"洗干净的假图", 目前没有任何自动判据能分。

**二十张图, 人看一眼就有答案。** 这个量级不值得再造工具去猜。

判读
----
- **看着就是假的**(金额字体不对/边缘发糊/拼接痕迹/排版错位) -> 加进排除名单, 分数线能往下放
- **看着是正常真图** -> 那就是真的难样本, 分数线只能停在这儿, 如实写进短板
- **看不准** -> 保持原样, 但报数时要说明"工作点由 N 张无法判定的图决定"

★ 千万别用模型分数当"它是假图"的证据来剔 —— 那是循环论证, 会把困难真图也一起剔掉,
  线下指标全线变好而线上误杀变差。判据只能是**画面本身**或水印/元数据。

用法
----
  python training/top_score_sheet.py --csv D:\probe\gen100k_blue\summary.csv D:\probe\gen100k_white\summary.csv ^
      --exclude D:\probe\exclude_big2.txt --require-located --top 30 --out D:\probe\top30.png

**只读**: 只读 CSV 和图片, 只写你指定的那张 png。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="真图分数最高的 N 张拼图, 供人工判读(只读)")
    ap.add_argument("--csv", type=Path, nargs="+", required=True, help="真图 summary.csv, 可多个")
    ap.add_argument("--col", default="tile_top3")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--exclude", type=Path, default=None, help="已确认假图的名单, 这些不再显示")
    ap.add_argument("--require-located", action="store_true",
                    help="只看金额定位成功的(线路B 只在定位成功时出信号, 定不到的本来就不参与定线)")
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--cell-w", type=int, default=200)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    drop: set[str] = set()
    if args.exclude and args.exclude.exists():
        drop = {ln.strip().lstrip("\ufeff")
                for ln in args.exclude.read_text(encoding="utf-8-sig").splitlines() if ln.strip()}
        print(f"(已排除名单 {len(drop)} 个)")

    rows: list[tuple[float, str, str]] = []
    for c in args.csv:
        n = 0
        for r in csv.DictReader(open(c, encoding="utf-8-sig")):
            if args.require_located and (r.get("roi_amount_located") or "1").strip() != "1":
                continue
            nm = (r.get("image_name") or "").strip()
            if nm in drop:
                continue
            v = (r.get(args.col) or "").strip()
            if not v:
                continue
            try:
                rows.append((float(v), r.get("image") or "", nm))
                n += 1
            except ValueError:
                pass
        print(f"  {c} -> {n} 行")
    if not rows:
        raise SystemExit("没读到分数 —— 先确认 --col 和 --require-located")
    rows.sort(key=lambda t: -t[0])
    top = rows[: args.top]
    print(f"\n共 {len(rows)} 张, 取分数最高的 {len(top)} 张 "
          f"(最高 {top[0][0]:.4f}, 第 {len(top)} 名 {top[-1][0]:.4f})")

    cols = args.cols
    rows_n = (len(top) + cols - 1) // cols
    lab, gap = 22, 8
    cells: list[tuple[Image.Image, str]] = []
    for s, path, nm in top:
        try:
            im = Image.open(path)
            im.draft("RGB", (im.size[0] // 3, im.size[1] // 3))
            a = np.asarray(im.convert("RGB"))
        except Exception:
            a = np.full((400, 200, 3), 220, np.uint8)
        h, w = a.shape[:2]
        sc = args.cell_w / max(1, w)
        cells.append((Image.fromarray(a).resize((args.cell_w, max(1, int(h * sc))), Image.LANCZOS),
                      f"{s:.4f} {nm[-18:]}"))
    ch = max(c.size[1] for c, _ in cells)
    W = gap + cols * (args.cell_w + gap)
    H = gap + rows_n * (ch + lab + gap)
    sheet = Image.new("RGB", (W, H), (248, 248, 248))
    drw = ImageDraw.Draw(sheet)
    for i, (im, label) in enumerate(cells):
        r, c = divmod(i, cols)
        x = gap + c * (args.cell_w + gap)
        y = gap + r * (ch + lab + gap)
        drw.rectangle([x - 1, y, x + args.cell_w, y + lab - 4], fill=(232, 232, 240))
        drw.text((x + 3, y + 4), label, fill=(0, 0, 0))
        sheet.paste(im, (x, y + lab))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out)
    print(f"-> {args.out}  ({W}x{H})")
    print()
    print("怎么判:")
    print("  **一眼就假**(金额字体不对/边缘糊/拼接痕/排版错位) -> 加进排除名单, 分数线能往下放")
    print("  **看着是正常真图**                                -> 是真的难样本, 线只能停这儿, 如实写进短板")
    print("  **看不准**                                        -> 保持原样, 但报数时要说明工作点由几张存疑的图决定")
    print()
    print("★ 别拿模型分数当'它是假图'的证据 —— 那是循环论证, 会连困难真图一起剔掉,")
    print("  线下指标全线变好而线上误杀变差。判据只能是画面本身或水印/元数据。")


if __name__ == "__main__":
    main()
