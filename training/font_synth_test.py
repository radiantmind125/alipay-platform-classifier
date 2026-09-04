r"""把真图的金额整体放大, 看 font_scan 到底抓不抓得住 —— 实测检出率, 不靠推算。

为什么要有这个脚本
------------------
经理说的"金额字体过大"这种假图, **我们手上一张都没有**。
之前报的检出率是**算出来的**: 把真图量到的高度乘以 1.10, 看会不会越过阈值。
那只验了算术, 没验管线 —— **定位器在改过的图上会不会跑偏, 一点都没测。**

这个脚本做真的: 拿真图, 把金额行整体放大若干倍再贴回去, 存成图,
然后**走完整条 measure() 管线**重新量一遍, 看报不报。

手法
----
`retype_bigger`: 取金额行连同上下留白, 整体放大 k 倍, 居中裁回原尺寸贴回原位。
模拟的是"在图片编辑器里把金额重打成更大的字号"。
放大会带来重采样痕迹, 但**这条判据量的是外接框尺寸, 不是清晰度**, 所以不受影响。

覆盖边界
--------
- 只造"整个金额一起变大"这一种。**只改其中一位数字**是另一回事, 这个脚本不覆盖。
- 放大倍数超过约 1.25 时金额可能顶到页面两边, 那种图现实中也不会有, 所以只跑到 1.20。
- 判据用的常态表**从同一批真图现算**, 和 font_scan 口径一致。

用法
----
  python training/font_synth_test.py D:\download2\OtherImages --n 400
  # 想留下造出来的图看看
  python training/font_synth_test.py D:\download2\OtherImages --n 400 --save D:\probe\synth
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from font_scan import _EXTS, _locate_amount, measure   # noqa: E402


def enlarge_amount(img: np.ndarray, factor: float):
    """把金额行整体放大 factor 倍再贴回原位。返回新图, 定位不到返回 None。"""
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    box = _locate_amount(gray, W, H)
    if box is None:
        return None
    bx0, by0, bx1, by1 = box

    # 连同上下留白一起取, 免得放大后数字被自己的边界切掉
    m = int((by1 - by0) * 0.55)
    y0, y1 = max(0, by0 - m), min(H, by1 + m)
    band = img[y0:y1, :]
    bh, bw = band.shape[:2]
    if bh < 8 or bw < 8:
        return None

    big = cv2.resize(band, (int(bw * factor), int(bh * factor)),
                     interpolation=cv2.INTER_CUBIC)
    nh, nw = big.shape[:2]
    oy, ox = (nh - bh) // 2, (nw - bw) // 2
    out = img.copy()
    out[y0:y1, :] = big[oy:oy + bh, ox:ox + bw]
    return out


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="金额放大的合成假图实测检出率")
    ap.add_argument("input", type=Path)
    ap.add_argument("--n", type=int, default=400, help="用多少张真图来造")
    ap.add_argument("--factors", type=str, default="1.03,1.05,1.08,1.10,1.15,1.20")
    ap.add_argument("--size-mult", type=float, default=1.03)
    ap.add_argument("--ratio-mult", type=float, default=1.05)
    ap.add_argument("--min-group", type=int, default=30,
                    help="定常态所需的最少张数; 这里只是自检, 可以比正式标定松")
    ap.add_argument("--save", type=Path, default=None, help="把造出来的图存到这里")
    ap.add_argument("--seed", type=int, default=20260904)
    args = ap.parse_args()

    factors = [float(x) for x in args.factors.split(",")]

    files = [p for p in args.input.rglob("*") if p.suffix.lower() in _EXTS]
    random.Random(args.seed).shuffle(files)
    print(f"图库 {len(files):,} 个文件, 取前面能量到的 {args.n} 张来造")

    # ---- 1. 先量真图, 现算常态表(口径与 font_scan 一致) ----
    base = []
    for p in files:
        if len(base) >= args.n:
            break
        try:
            r = measure(str(p))
        except Exception:
            r = None
        if r:
            r["path"] = str(p)
            base.append(r)
    print(f"量到 {len(base)} 张真图")
    if not base:
        print("一张都没量到"); return

    by_res = defaultdict(list)
    for r in base:
        by_res[(r["W"], r["H"])].append(r)
    groups = {k: v for k, v in by_res.items() if len(v) >= args.min_group}
    if not groups:
        print(f"没有分辨率够 {args.min_group} 张, 定不出常态"); return
    norm = {k: Counter(int(round(x["amount_h"])) for x in v).most_common(1)[0][0]
            for k, v in groups.items()}
    # 与 font_scan 一致: 比值常态按分辨率算, 不用全局中位
    norm_ratio = {k: float(np.median([r["amt_to_body"] for r in v]))
                  for k, v in groups.items()}
    usable = [r for r in base if (r["W"], r["H"]) in norm]
    print(f"常态表 {len(norm)} 组, 可用于自检的真图 {len(usable)} 张")
    for k in norm:
        print(f"    {k[0]}x{k[1]}  常态高 {norm[k]}  比值常态 {norm_ratio[k]:.4f}")

    def flagged(rec):
        n = norm[(rec["W"], rec["H"])]
        return (rec["amount_h"] > args.size_mult * n
                and rec["amt_to_body"] > args.ratio_mult * norm_ratio[(rec["W"], rec["H"])])

    print(f"\n对照组: 这 {len(usable)} 张真图本身报出 "
          f"{sum(1 for r in usable if flagged(r))} 张")

    if args.save:
        args.save.mkdir(parents=True, exist_ok=True)

    # ---- 2. 逐个倍数造图, 走完整条管线重新量 ----
    print(f"\n{'放大倍数':>9} {'造出来':>7} {'量得到':>7} {'报出':>7} {'检出率':>9}  {'量到的高度中位':>12}")
    tmp = args.save if args.save else Path(os.environ.get("TEMP", ".")) / "font_synth_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    for f in factors:
        made = meas = hit = 0
        heights = []
        for r in usable:
            img = cv2.imread(r["path"], cv2.IMREAD_COLOR)
            if img is None:
                continue
            big = enlarge_amount(img, f)
            if big is None:
                continue
            made += 1
            # 存成 PNG 再读回来, 保证走的是和线上一样的解码路径
            outp = tmp / f"x{f:.2f}_{Path(r['path']).stem}.png"
            cv2.imwrite(str(outp), big)
            try:
                r2 = measure(str(outp))
            except Exception:
                r2 = None
            if not args.save:
                try: outp.unlink()
                except Exception: pass
            if not r2:
                continue
            meas += 1
            heights.append(r2["amount_h"])
            if (r2["W"], r2["H"]) in norm and flagged(r2):
                hit += 1
        rate = f"{100.0 * hit / meas:.1f}%" if meas else "-"
        medh = f"{np.median(heights):.1f}" if heights else "-"
        print(f"{f:>9.2f} {made:>7} {meas:>7} {hit:>7} {rate:>9}  {medh:>12}")

    print("\n★ '量得到'比'造出来'少的部分是**定位器在改过的图上失效**了 —— "
          "这正是纯算术推算看不到的那一块。")
    if args.save:
        print(f"造出来的图留在 {args.save}, 可以打开看看像不像真的篡改。")


if __name__ == "__main__":
    main()
