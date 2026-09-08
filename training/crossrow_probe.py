r"""探路: 金额数字 vs 时间行数字, **同一张图内部**比字形。

思路
----
账单详情页除了金额, "创建时间 2026-07-05 22:13:00" 那一行也是数字,
而且是**同一个渲染引擎、同一套字体**画出来的。

如果有人只改金额(最省事的做法), 金额那几个数字就和时间行那几个数字**不再是同一套字**了。
这是纯粹的**同图内部对比** —— 不查表、不要字库、不挑机型,
和负号那条是同一个套路, 只是参照物从"负号"换成了"这张图自己的另一处数字"。

★ 顺带解决 flux 说的那个弊端("必须得不断扩充字型"):
  在同一张图里比, 就不需要任何参照字形库。

要量什么
--------
金额字大、时间字小, 不能直接比像素。所以量**与尺度无关**的形状描述量:
    aspect   = 字宽 / 字高
    fill     = 前景面积 / 外接框面积          (笔画有多粗)
    stroke   = 前景面积 / 骨架长度            (笔画绝对粗细, 再除以字高做归一)
真图上这三个量在两处应当高度一致(同一套字);
只改金额的话, 金额侧会偏离时间侧。

注意: 字体在不同字号上并非严格等比(hinting、小字号加粗),
所以**不指望比值等于 1**, 只要求它在真图上**分布很紧**。这正是标定的方式。

用法
----
  python training/crossrow_probe.py --src <图库> --n 400 --dump <存图目录>
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
    """连通域 -> [(x, y, w, h, area)]"""
    n, _, stats, _ = cv2.connectedComponentsWithStats(bin_img, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a >= min_area:
            out.append((int(x), int(y), int(w), int(h), int(a)))
    return out


def rows_of(gray):
    """把深色连通域按纵向重叠聚成行, 返回 [(y0, y1, [comp...])], 按 y 排序。"""
    H, W = gray.shape
    dark = cv2.threshold(gray, GRAY_DARK - 1, 255, cv2.THRESH_BINARY_INV)[1]
    cs = [c for c in comps(dark, 20)
          if 0.008 * H < c[3] < 0.22 * H and c[2] < 0.5 * W]
    if not cs or len(cs) > 20000:
        return []
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
            rows.append([c])
            bounds.append((y, y + h))
    out = [(b[0], b[1], r) for b, r in zip(bounds, rows)]
    out.sort(key=lambda t: t[0])
    return out


def digits_of(row_comps):
    """从一行里挑出像数字的块: 高度接近中位、不是横条、不是小点。"""
    if len(row_comps) < 3:
        return []
    hs = sorted(c[3] for c in row_comps)
    mh = hs[len(hs) // 2]
    if mh < 8:
        return []
    d = [c for c in row_comps
         if c[3] > 0.75 * mh and c[2] < 1.5 * c[3] and c[2] > 0.25 * c[3]]
    return d


def shape_stats(gray, ds):
    """对一组数字块量与尺度无关的形状描述量。"""
    if len(ds) < 3:
        return None
    asp, fill = [], []
    for x, y, w, h, a in ds:
        if w <= 0 or h <= 0:
            continue
        sub = gray[y:y + h, x:x + w]
        if sub.size == 0:
            continue
        # 局部 Otsu, 前景取深色
        t = cv2.threshold(sub, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        ink = int(np.count_nonzero(t))
        if ink == 0 or ink == w * h:
            continue
        asp.append(w / h)
        fill.append(ink / (w * h))
    if len(asp) < 3:
        return None
    return {
        "n": len(asp),
        "h": float(np.median([c[3] for c in ds])),
        "aspect": float(np.median(asp)),
        "fill": float(np.median(fill)),
        "aspect_cv": float(np.std(asp) / max(1e-9, np.mean(asp))),
    }


def analyse(path: Path, dump: Path | None = None):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    H, W = img.shape[:2]
    # 翻拍/异形不看
    if W * H >= 6_000_000 or max(W, H) / max(1, min(W, H)) < 1.7:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rs = rows_of(gray)
    if not rs:
        return None

    # 金额行: 8%~55% 高度带里字最高的一行
    band = [r for r in rs if 0.08 * H <= r[0] <= 0.55 * H]
    if not band:
        return None
    amt = max(band, key=lambda r: np.median([c[3] for c in r[2]]))
    amt_d = digits_of(amt[2])
    if len(amt_d) < 4:
        return None

    # 时间行: 金额行**下方**、数字个数 >= 6、字比金额小的行里, 最靠上的那个
    cand = []
    for y0, y1, cs in rs:
        if y0 <= amt[1]:
            continue
        ds = digits_of(cs)
        if len(ds) < 6:
            continue
        mh = np.median([c[3] for c in ds])
        if mh >= 0.6 * np.median([c[3] for c in amt_d]):
            continue          # 和金额差不多大的不是正文
        cand.append((y0, ds))
    if not cand:
        return None
    cand.sort(key=lambda t: t[0])
    date_d = cand[0][1]

    a = shape_stats(gray, amt_d)
    b = shape_stats(gray, date_d)
    if not a or not b:
        return None

    if dump is not None:
        y0 = max(0, amt[0] - 10); y1 = min(H, cand[0][0] + 60)
        crop = img[y0:y1]
        sc = 700 / max(1, crop.shape[1])
        if sc < 1:
            crop = cv2.resize(crop, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(dump / f"{path.stem[:34]}_amt{a['h']:.0f}_date{b['h']:.0f}.png"), crop)

    return {
        "name": path.name,
        "amt_h": a["h"], "date_h": b["h"],
        "amt_aspect": a["aspect"], "date_aspect": b["aspect"],
        "amt_fill": a["fill"], "date_fill": b["fill"],
        "aspect_ratio": a["aspect"] / b["aspect"] if b["aspect"] else 0,
        "fill_ratio": a["fill"] / b["fill"] if b["fill"] else 0,
        "size_ratio": a["h"] / b["h"] if b["h"] else 0,
        "amt_n": a["n"], "date_n": b["n"],
    }


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="金额数字 vs 时间行数字, 同图内比字形")
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--dump", type=Path, default=None, help="存几张切图人工核对")
    ap.add_argument("--dump-n", type=int, default=8)
    args = ap.parse_args()

    if args.dump:
        args.dump.mkdir(parents=True, exist_ok=True)

    files = []
    for p in sorted(args.src.rglob("*")):
        if p.suffix.lower() in _EXTS:
            files.append(p)
        if len(files) >= args.n:
            break
    print(f"扫 {len(files)} 张")

    rows, dumped = [], 0
    for p in files:
        d = analyse(p, args.dump if dumped < args.dump_n else None)
        if d:
            rows.append(d)
            if args.dump and dumped < args.dump_n:
                dumped += 1

    print(f"两处都量到的 {len(rows)} 张 ({len(rows) / max(1, len(files)) * 100:.0f}%)")
    if not rows:
        print("★ 覆盖率为零, 这条路走不通")
        return

    def rep(k):
        v = sorted(r[k] for r in rows)
        n = len(v)
        q = lambda p: v[min(n - 1, int(n * p))]
        med = v[n // 2]
        spread = (q(0.99) - q(0.01)) / med if med else 0
        print(f"  {k:<14} 中位 {med:7.4f}   p1 {q(0.01):7.4f}   p99 {q(0.99):7.4f}"
              f"   p1~p99 相对宽度 {spread * 100:5.1f}%")

    print("\n★ 分布越紧, 越能当判据(负号那条的 p1~p99 大致在 20% 以内):")
    for k in ("aspect_ratio", "fill_ratio", "size_ratio"):
        rep(k)
    print("\n参考(两处各自的绝对值):")
    for k in ("amt_aspect", "date_aspect", "amt_fill", "date_fill", "amt_h", "date_h"):
        rep(k)


if __name__ == "__main__":
    main()
