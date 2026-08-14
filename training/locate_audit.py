r"""把金额定位框的**几何形状**量一遍 —— 用来判断"定位成功"到底可不可信(只读)。

为什么要有这个
--------------
人工看 top30 的时候发现两张**根本不是收据**的图**通过了"定位成功"**:
第 20 名是豆浆机商品页, 第 22 名是近乎全灰的空图。定位器在它们身上"找到了金额",
线路B 于是给那个垃圾裁块打了 **0.99** —— **线上这两张会被自动拒掉。**

而第 20 名正好就是硬拒那条线本身(阈值 0.9883 = 第 20 高分)。

这不是标定问题, 是**正确性问题**: 系统在自信地拒绝它根本没资格判断的东西。

但**先量再修**。`summary.csv` 只存了 `roi_amount_located` 这个 0/1, 没存框的坐标,
所以现在既不知道这种误触发有多频繁, 也不知道该拿什么判据去卡。
这个脚本把框的几何量出来, **判据从数据里定, 不靠拍脑袋。**

定位分支与 `predict_tiled.py:198-205` **逐行一致**:
    蓝图 -> `locate_amount_blue`; 白图 -> `engine_b_tamper.locate_amount`
不一致的话量出来的分布就没有参考价值。

用法
----
  python training/locate_audit.py --csv D:\probe\gen100k_blue\summary.csv D:\probe\gen100k_white\summary.csv ^
      --out D:\probe\locate_geom.csv --sample 20000

  # 假图那边也量一遍, 用来确认判据不会把**真的金额框**误伤(那会直接掉召回)
  python training/locate_audit.py --csv D:\probe\_clean_ld9fix\wanblue_ld9fix_clean.csv ^
      --out D:\probe\locate_geom_fake.csv

**只读**: 只读图片, 只写你指定的那个 CSV。不用 GPU。
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="量金额定位框的几何形状(只读, 不用 GPU)")
    ap.add_argument("--csv", type=Path, nargs="+", required=True, help="含 image 列的 summary.csv")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--exclude", type=Path, default=None, help="要跳过的文件名清单")
    ap.add_argument("--sample", type=int, default=0, help="随机抽这么多张(0=全部)")
    ap.add_argument("--sample-seed", type=int, default=0)
    args = ap.parse_args()

    from engine_b_tamper import locate_amount
    from locate_blue import is_blue_page, locate_amount_blue

    drop: set[str] = set()
    if args.exclude and args.exclude.exists():
        drop = {ln.strip().lstrip("\ufeff")
                for ln in args.exclude.read_text(encoding="utf-8-sig").splitlines() if ln.strip()}
        print(f"(排除名单 {len(drop)} 个)")

    items: list[tuple[str, str]] = []
    for c in args.csv:
        n = 0
        with open(c, encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                nm = (r.get("image_name") or "").strip()
                p = (r.get("image") or "").strip()
                if p and nm not in drop:
                    items.append((nm or Path(p).name, p))
                    n += 1
        print(f"  {c} -> {n} 行")
    if not items:
        raise SystemExit("CSV 里没读到图片")
    if args.sample and args.sample < len(items):
        items = random.Random(args.sample_seed).sample(items, args.sample)
        print(f"  随机抽 {len(items)} 张")

    print(f"\n逐张定位 {len(items)} 张(纯 CPU, 不解码到模型)...", flush=True)
    rows: list[dict] = []
    for i, (nm, p) in enumerate(items, 1):
        rec = {"image_name": nm, "page": "", "located": 0, "x0": "", "y0": "", "x1": "", "y1": "",
               "glyphs": "", "W": "", "H": "", "w_rel": "", "h_rel": "", "aspect": "", "ycen_rel": ""}
        try:
            arr = np.asarray(Image.open(p).convert("RGB"))
            H, W = arr.shape[:2]
            blue = bool(is_blue_page(arr))
            rec["page"] = "blue" if blue else "white"
            rec["W"], rec["H"] = W, H
            loc = locate_amount_blue(arr) if blue else locate_amount(arr)
            if loc:
                x0, y0, x1, y1 = loc[0], loc[1], loc[2], loc[3]
                bw, bh = max(1, x1 - x0), max(1, y1 - y0)
                rec.update(located=1, x0=x0, y0=y0, x1=x1, y1=y1,
                           glyphs=len(loc[4]) if len(loc) > 4 and loc[4] is not None else "",
                           w_rel=round(bw / W, 4), h_rel=round(bh / H, 4),
                           aspect=round(bw / bh, 3), ycen_rel=round((y0 + y1) / 2 / H, 4))
        except Exception:  # noqa: BLE001
            pass
        rows.append(rec)
        if i % 5000 == 0:
            print(f"  已过 {i}/{len(items)}...", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"\n-> {args.out}({len(rows)} 行)")

    for page in ("blue", "white"):
        sub = [r for r in rows if r["page"] == page and r["located"]]
        tot = sum(1 for r in rows if r["page"] == page)
        if not sub:
            print(f"\n{page}: {tot} 张, 一张都没定位到")
            continue
        print(f"\n=== {page} === 共 {tot} 张, 定位成功 {len(sub)} ({len(sub)/max(1,tot)*100:.1f}%)")
        print(f"{'指标':10s}{'p1':>9s}{'p5':>9s}{'p50':>9s}{'p95':>9s}{'p99':>9s}")
        for key, label in (("aspect", "宽高比"), ("h_rel", "框高/页高"),
                           ("w_rel", "框宽/页宽"), ("ycen_rel", "框中心/页高"), ("glyphs", "字符数")):
            v = [r[key] for r in sub if r[key] != ""]
            if not v:
                print(f"{label:10s}{'(没有这一列)':>45s}")
                continue
            a = np.asarray(v, dtype=float)
            print(f"{label:10s}" + "".join(f"{np.percentile(a, q):9.3f}" for q in (1, 5, 50, 95, 99)))

    print()
    print("怎么用这张表:")
    print("  **正常金额框**应该很集中: 一行数字, 宽高比大致固定, 高度占页面比例也固定。")
    print("  分布**两端的那些**(p1 以下 / p99 以上)就是可疑的误触发 —— 挑几张出来看一眼就知道。")
    print("  定好判据后, **一定要拿假图那边再量一遍**: 判据若把真的金额框也卡掉, 直接掉召回。")
    print()
    print("★ 判据只能用**框本身的几何/内容**, 不能用模型分数 —— 用分数就是循环论证。")


if __name__ == "__main__":
    main()
