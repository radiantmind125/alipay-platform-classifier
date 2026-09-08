r"""探路: 找**与数字内容无关**的字体不变量。

为什么要这样找
--------------
负号和小数点之所以好用, 是因为它们是**恒定字形** —— 每张图上都是同一个形状。
数字不是: "1" 比 "0" 窄得多, 所以"数字中位宽 / 中位高"会随金额里恰好有几个 1 而变。
17.6 万张实测: 这个比值分布很紧(p1~p99 只有 11.4%), 但大额富集只有 3.59 倍,
远不如负号(12.2~16.0 倍)和圆点(14.4 倍) —— 紧不等于能分辨。

所以这一轮只找**不挑数字内容**的量:

  1. stroke_ratio  笔画粗细 / 数字高
     笔画粗细用距离变换取: 前景内部到背景的最大距离约等于半个笔宽。
     字体一旦换了, 笔画粗细跟字高的比例就变了。**和画的是哪个数字无关。**

  2. pitch_ratio   相邻数字左边缘间距的中位 / 数字高
     金额多半用表格数字(tabular figures), 每个数字占一样宽的槽。
     换成比例字体后, 槽宽就不再恒定。**也和内容无关。**

  3. pitch_cv      间距的变异系数
     表格数字应当接近 0; 比例字体会明显偏大。**这条甚至不需要阈值标定, 0 就是理论值。**

  4. height_cv     数字高度的变异系数
     同一行同一字体, 数字高度应当几乎一样。

用法
----
  python training/fontinv_probe.py --src <图库> --n 800
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
GRAY_DARK = 140


def comps(bin_img, min_area=6):
    n, _, stats, _ = cv2.connectedComponentsWithStats(bin_img, connectivity=8)
    return [(int(stats[i][0]), int(stats[i][1]), int(stats[i][2]),
             int(stats[i][3]), int(stats[i][4]))
            for i in range(1, n) if stats[i][4] >= min_area]


def locate_amount(gray):
    """8%~55% 高度带里字最高的一行, 返回该行的连通域列表。"""
    H, W = gray.shape
    y0b, y1b = int(H * 0.08), int(H * 0.55)
    if y1b - y0b < 8:
        return None
    band = gray[y0b:y1b]
    dark = cv2.threshold(band, GRAY_DARK - 1, 255, cv2.THRESH_BINARY_INV)[1]
    cs = [(x, y + y0b, w, h, a) for x, y, w, h, a in comps(dark, 20)
          if 0.02 * H < h < 0.22 * H and w < 0.5 * W]
    if not cs or len(cs) > 20000:
        return None
    cs.sort(key=lambda c: (c[1], c[0]))
    rows, bounds = [], []
    for c in cs:
        x, y, w, h, a = c
        placed = False
        for i, (ry0, ry1) in enumerate(bounds):
            if min(y + h, ry1) - max(y, ry0) >= 0.5 * min(h, ry1 - ry0):
                rows[i].append(c)
                bounds[i] = (min(ry0, y), max(ry1, y + h))
                placed = True
                break
        if not placed:
            rows.append([c]); bounds.append((y, y + h))
    best, bestmed = None, -1
    for r in rows:
        if len(r) < 2:
            continue
        x0 = min(c[0] for c in r); x1 = max(c[0] + c[2] for c in r)
        if x1 - x0 < 0.1 * W:
            continue
        med = float(np.median([c[3] for c in r]))
        if med > bestmed:
            bestmed, best = med, r
    return best


def measure(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    H, W = img.shape[:2]
    if W * H >= 6_000_000 or max(W, H) / max(1, min(W, H)) < 1.7:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    row = locate_amount(gray)
    if not row:
        return None

    x0 = min(c[0] for c in row); x1 = max(c[0] + c[2] for c in row)
    y0 = min(c[1] for c in row); y1 = max(c[1] + c[3] for c in row)
    pad = max(2, int((y1 - y0) * 0.15)); padx = max(pad, int((x1 - x0) * 0.30))
    cx0, cy0 = max(0, x0 - padx), max(0, y0 - pad)
    cx1, cy1 = min(W, x1 + pad), min(H, y1 + pad)
    sub = gray[cy0:cy1, cx0:cx1]
    if sub.shape[0] < 8 or sub.shape[1] < 8:
        return None

    fg = cv2.threshold(sub, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    if np.count_nonzero(fg) * 255 > 127 * fg.size:
        fg = cv2.bitwise_not(fg)
    g = comps(fg, 6)
    if len(g) < 4:
        return None
    hs = sorted(c[3] for c in g); medH = hs[len(hs) // 2]
    if medH < 20:
        return None
    digits = [c for c in g if c[3] > 0.75 * medH and c[2] < 1.5 * c[3]]
    if len(digits) < 4:
        return None
    mh = float(np.median([c[3] for c in digits]))
    mw = float(np.median([c[2] for c in digits]))
    if mh < 60 or not (0.45 <= mw / mh <= 0.75):
        return None

    # 1. 笔画粗细: 对每个数字做距离变换, 取内部最大距离 (约为半个笔宽)
    strokes = []
    for x, y, w, h, a in digits:
        m = fg[y:y + h, x:x + w]
        if m.size == 0:
            continue
        pad_m = cv2.copyMakeBorder(m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        dt = cv2.distanceTransform(pad_m, cv2.DIST_L2, 5)
        if dt.max() > 0:
            strokes.append(float(dt.max()) * 2.0)
    if len(strokes) < 3:
        return None
    stroke = float(np.median(strokes))

    # 2/3. 字距: 按 x 排序取相邻左边缘之差
    # ★ 只取相邻**且中间没有别的东西**的一对 —— 小数点会撑开一个不规则的间隔,
    #   上一版把它算进去了, 所以 pitch_cv 中位是 0.19 而不是接近 0。
    ds = sorted(digits, key=lambda c: c[0])
    gaps = []
    for a, b in zip(ds, ds[1:]):
        d = b[0] - a[0]
        if 0.5 * mw < d < 1.8 * mw:      # 挡掉跨小数点那一跳
            gaps.append(d)
    if len(gaps) < 3:
        return None
    pitch = float(np.median(gaps))
    pitch_cv = float(np.std(gaps) / max(1e-9, np.mean(gaps)))

    hh = [c[3] for c in digits]
    height_cv = float(np.std(hh) / max(1e-9, np.mean(hh)))

    # 4/5. ★ 拿**负号这个恒定字形**的另外两个属性:
    #   竖直位置和它与第一个数字的间隔。两者都是字体度量, 与画的是哪个数字无关。
    top = float(np.median([c[1] for c in digits]))
    bars = [c for c in g if c[2] >= 1.5 * c[3] and c[3] <= 0.45 * medH]
    bar_vpos = bar_gap = float("nan")
    if bars:
        bar = sorted(bars, key=lambda c: (c[0], c[1], c[2]))[0]
        bar_vpos = ((bar[1] + bar[3] / 2.0) - top) / mh
        right = [c for c in ds if c[0] >= bar[0] + bar[2]]
        if right:
            bar_gap = (right[0][0] - (bar[0] + bar[2])) / mh

    # 6. 小数点竖直位置: 它坐在基线上, 应当很稳
    base = float(np.median([c[1] + c[3] for c in digits]))
    dots = [c for c in g if c not in digits and c not in bars
            and c[3] <= 0.30 * mh and c[2] <= 0.60 * mw]
    dot_vpos = float("nan")
    if len(dots) == 1:
        dot_vpos = ((dots[0][1] + dots[0][3]) - base) / mh

    return {
        "name": path.name, "mh": mh,
        "stroke_ratio": stroke / mh,
        "pitch_ratio": pitch / mh,
        "pitch_cv": pitch_cv,
        "height_cv": height_cv,
        "aspect": mw / mh,
        "bar_vpos": bar_vpos,
        "bar_gap": bar_gap,
        "dot_vpos": dot_vpos,
        "n": len(digits),
    }


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="找与数字内容无关的字体不变量")
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    files = []
    for p in sorted(args.src.rglob("*")):
        if p.suffix.lower() in _EXTS:
            files.append(p)
        if len(files) >= args.n:
            break
    print(f"扫 {len(files)} 张")

    rows = [d for d in (measure(p) for p in files) if d]
    print(f"量到 {len(rows)} 张 ({len(rows) / max(1, len(files)) * 100:.0f}%)\n")
    if not rows:
        return

    print("★ 分布越紧越可能当判据。对照: 负号宽比 p1~p99 相对宽度 11.3%")
    import math
    for k in ("stroke_ratio", "pitch_ratio", "pitch_cv", "height_cv", "aspect",
              "bar_vpos", "bar_gap", "dot_vpos"):
        v = sorted(r[k] for r in rows if not math.isnan(r.get(k, float("nan"))))
        if len(v) < 20:
            print(f'  {k:<14} 量到的太少({len(v)}), 跳过'); continue
        n = len(v)
        q = lambda p: v[min(n - 1, int(n * p))]
        med = v[n // 2]
        w = (q(.99) - q(.01)) / med * 100 if med else 0
        print(f"  {k:<14} 中位 {med:7.4f}   p1 {q(.01):7.4f}   p99 {q(.99):7.4f}"
              f"   相对宽度 {w:6.1f}%   量到 {n}")

    if args.out:
        import csv
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wtr.writeheader(); wtr.writerows(rows)
        print(f"\n写出 {args.out}")


if __name__ == "__main__":
    main()
