r"""金额行"恒定形状"扫描 —— 负号宽比 + 小数点面积比, 两条都不查表。

为什么是这两个
--------------
经理 2026-09-03: "可以在**这个思路**上做扩展"。「这个思路」= `MinusCheck` 的方法本身:
量 `负号宽 / 数字中位宽`, **两个量都在同一张图内部**。

关键不在"图内", 而在于 **负号是一个恒定形状** ——
不管什么机型、什么金额、哪个用户, 负号永远是同一个字形。数字不是, 它取决于付了多少钱。
**所以这条不用查表、不挑机型、不随 app 改版失效。**

之前几轮做的字体检查都违背了这一点(拿分辨率常态表当参照), 于是只能用在六种苹果分辨率上。
这次找的是**页面上其它的恒定形状**, 试了四个, 只有小数点站住了。

★ 小数点和负号一样是恒定形状, 而且**每一笔金额都有**。

用面积不用外接框
----------------
小数点只有十来个像素宽, **边长的量化太粗**(一个像素就是 8%)。
**面积是上百个像素的计数**, 同样抖动一个像素, 影响小一个量级。
同一个小数点, 量二阶矩那条死了, 量面积这条活了。

机理
----
**支付宝金额字体的小数点异常地大**(面积/数字高² ≈ 0.030), 换字体基本都偏小:
Arial 0.63 倍, SegoeUI 0.61 倍, Leelawadee 0.61 倍, MSYaHei 0.69 倍, Calibri 0.83 倍。

★ 和负号那条是**正交**的: 等宽字体 Consolas 的负号宽度正好撞上真图,
负号那条只抓到 0.5%, **小数点这条抓 100%**。落在 0.94~1.03 那几个小数点逃掉的, 由负号那条接住。

顺带查一件事: ¥ 会不会把负号那条顶出误报
------------------------------------------
页面上有 `-¥100.00` 这种排版, **¥ 在负号右边**。¥ 比数字窄但够高,
会被当成一位数字算进去, **把数字中位宽 `mw` 拉低, 于是 `负号宽/mw` 被顶高** ——
也就是说 ¥ 可能在**给已上线的那条制造误报**。这个脚本一并统计, 用数据说话。

覆盖边界
--------
- 只管白底付款页(要有负号)。蓝底转账页、收款页跳过。
- 抓的是"金额被换字体重打"。**这种手法我们至今一张真样本都没有**,
  检出率是造图实测的, 不是真样本上的。
- 经理给的三张假图小数点是 0.0309, **正常, 这条抓不到它们** —— 它们的金额本来就没被重打过。

用法
----
  # 试跑
  python training/dot_scan.py D:\download2\OtherImages --limit 5000 --out D:\probe\dot_try.csv

  # 主跑
  python training/dot_scan.py D:\download2\OtherImages --limit 50000 --out D:\probe\dot.csv

  # 换阈值重算, 不用重扫(秒级)
  python training/dot_scan.py D:\download2\OtherImages --replay D:\probe\dot.csv

  # 把报出来的金额行拼成一张图, 人工看
  python training/dot_scan.py D:\download2\OtherImages --replay D:\probe\dot.csv --sheet D:\probe\dot_hits.png

★ 输出**分 png / jpg 两栏**。实测 png 里 97.9% 是苹果分辨率, jpg 里只有 0.6%,
  所以 png/jpg 基本就是 iPhone/安卓。**两边差一个量级就等于这条在安卓上不成立。**
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_TS = re.compile(r"_(\d{8})\d{6}")

# 与 MinusCheck.cs 的常量一一对应, 改这里就要同步改那边
GRAY_DARK = 140
BAR_HIGH = 0.78          # MinusCheck.Threshold, 已上线
BAR_LOW = 0.0            # MinusCheck.ThresholdLow, 默认关闭(实测代价单边压在安卓上)
DOT_LOW, DOT_HIGH = 0.0200, 0.0365
MIN_DIGIT_HEIGHT = 60


def _label(bw, min_area):
    n, _, st, _ = cv2.connectedComponentsWithStats(bw, 8)
    return [(int(st[i, 0]), int(st[i, 1]), int(st[i, 2]), int(st[i, 3]), int(st[i, 4]))
            for i in range(1, n) if st[i, 4] >= min_area]


def _locate_amount(gray, W, H):
    y0b, y1b = int(H * 0.08), int(H * 0.55)
    if y1b - y0b < 8:
        return None
    _, dark = cv2.threshold(gray[y0b:y1b], GRAY_DARK - 1, 255, cv2.THRESH_BINARY_INV)
    comps = [(x, y + y0b, w, h, a) for (x, y, w, h, a) in _label(dark, 20)
             if not (h < 0.02 * H or h > 0.22 * H or w > 0.5 * W)]
    if not comps or len(comps) > 20000:
        return None
    rows, bounds = [], []
    for c in sorted(comps, key=lambda k: (k[1], k[0])):
        x, y, w, h, _a = c
        for i in range(len(rows)):
            ry0, ry1 = bounds[i]
            if min(y + h, ry1) - max(y, ry0) >= 0.5 * min(h, ry1 - ry0):
                rows[i].append(c)
                bounds[i] = (min(ry0, y), max(ry1, y + h))
                break
        else:
            rows.append([c]); bounds.append((y, y + h))
    best, best_med = None, -1.0
    for r in rows:
        if len(r) < 2:
            continue
        if max(k[0] + k[2] for k in r) - min(k[0] for k in r) < 0.1 * W:
            continue
        med = sorted(k[3] for k in r)[len(r) // 2]
        if med > best_med:
            best_med, best = med, r
    if best is None:
        return None
    return (min(k[0] for k in best), min(k[1] for k in best),
            max(k[0] + k[2] for k in best), max(k[1] + k[3] for k in best))


def measure(path):
    """逐条对应 MinusCheck.cs 的判定链。返回 dict 或 None(判不了)。"""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    H, W = img.shape[:2]
    if W < 16 or H < 16:
        return None
    top = img[: max(1, H // 3)]
    b, g, r = (float(top[:, :, i].mean()) for i in range(3))
    if b > r + 25 and b > g + 15:          # 蓝底转账页
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    box = _locate_amount(gray, W, H)
    if box is None:
        return None
    bx0, by0, bx1, by1 = box
    pad = max(2, int((by1 - by0) * 0.15))
    padx = max(pad, int((bx1 - bx0) * 0.30))
    cx0, cy0 = max(0, bx0 - padx), max(0, by0 - pad)
    cx1, cy1 = min(W, bx1 + pad), min(H, by1 + pad)
    if cx1 - cx0 < 8 or cy1 - cy0 < 8:
        return None

    sub = gray[cy0:cy1, cx0:cx1]
    _, fg = cv2.threshold(sub, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    if 255 * int((fg > 0).sum()) > 127 * fg.size:
        fg = 255 - fg
    glyphs = _label(fg, 6)
    if len(glyphs) < 4:
        return None
    hs = sorted(k[3] for k in glyphs)
    med_h = hs[len(hs) // 2]
    if med_h < 20:
        return None

    digits = [k for k in glyphs if k[3] > 0.75 * med_h and k[2] < 1.5 * k[3]]
    bars = [k for k in glyphs if k[2] >= 1.5 * k[3] and k[3] <= 0.45 * med_h]
    if len(digits) < 4 or not bars:
        return None
    dh = np.array([k[3] for k in digits], float)
    dw = np.array([k[2] for k in digits], float)
    if dh.std() / dh.mean() > 0.08 or dw.std() / dw.mean() > 0.30:
        return None
    mw, mh = float(np.median(dw)), float(np.median(dh))
    if not (0.45 <= mw / mh <= 0.75):
        return None

    bar = sorted(bars, key=lambda k: (k[0], k[1], k[2]))[0]
    baseline = float(np.median([k[1] + k[3] for k in digits]))
    dots = [k for k in glyphs if k not in digits and k not in bars
            and k[3] <= 0.30 * mh and k[2] <= 0.60 * mw
            and abs((k[1] + k[3]) - baseline) <= 0.06 * mh]

    # ¥ 排查: 最左边那个"数字"如果明显比其余矮, 多半是 ¥ 被当成数字算进去了。
    # 它比数字窄, 会把 mw 拉低, 于是把 负号宽/mw 顶高 —— 给已上线那条制造误报。
    ds = sorted(digits, key=lambda k: k[0])
    yen = False
    mw_noyen = mw
    if len(ds) >= 4:
        rest_h = float(np.median([k[3] for k in ds[1:]]))
        if ds[0][3] < 0.95 * rest_h:
            yen = True
            mw_noyen = float(np.median([k[2] for k in ds[1:]]))

    return dict(
        name=os.path.basename(path), W=W, H=H,
        mh=mh, mw=mw, n_digit=len(digits),
        bar_ratio=bar[2] / mw,
        bar_ratio_noyen=bar[2] / mw_noyen,
        dot_area=float(dots[0][4]) if len(dots) == 1 else 0.0,
        dot_ratio=(dots[0][4] / (mh * mh)) if len(dots) == 1 else 0.0,
        n_dot=len(dots), yen=int(yen),
        fmt="png" if path.lower().endswith(".png") else "jpg",
        month=(_TS.search(os.path.basename(path)).group(1)[:6]
               if _TS.search(os.path.basename(path)) else ""),
    )


def _pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def _analyse(args, rows):
    ok = [r for r in rows if r["mh"] >= MIN_DIGIT_HEIGHT]
    print(f"\n量到 {len(rows):,} 张, 其中数字高 >= {MIN_DIGIT_HEIGHT} 的 {len(ok):,} 张"
          f" (线上真正判定的那一批)")
    if not ok:
        print("没有可判定的图"); return

    by = defaultdict(list)
    for r in ok:
        by[r["fmt"]].append(r)

    print(f"\n{'':<10} {'张数':>7} {'小数点量到':>10} {'小数点面积比 中位/p1/p99':>32} {'负号宽比 中位/p1/p99':>28}")
    for f in ("png", "jpg"):
        v = by.get(f)
        if not v:
            continue
        d = np.array([r["dot_ratio"] for r in v if r["n_dot"] == 1])
        bw = np.array([r["bar_ratio"] for r in v])
        print(f"  {f:<8} {len(v):>7,} {100.0*len(d)/len(v):>9.2f}% "
              f"{np.median(d):>10.4f} {_pct(d,1):>10.4f} {_pct(d,99):>10.4f} "
              f"{np.median(bw):>9.4f} {_pct(bw,1):>8.4f} {_pct(bw,99):>8.4f}")

    print(f"\n逐条分解 (报出的都是真图, 所以就是误报率)")
    print(f"{'组合':<26} {'png':>16} {'jpg':>16} {'jpg/png':>9}")
    combos = [
        ("只负号高侧(已上线)", lambda r: r["bar_ratio"] >= BAR_HIGH),
        ("只小数点(新增)", lambda r: r["n_dot"] == 1 and not (DOT_LOW <= r["dot_ratio"] <= DOT_HIGH)),
        ("两条并联", lambda r: r["bar_ratio"] >= BAR_HIGH
                    or (r["n_dot"] == 1 and not (DOT_LOW <= r["dot_ratio"] <= DOT_HIGH))),
    ]
    for lbl, sel in combos:
        cells, rates = [], {}
        for f in ("png", "jpg"):
            v = by.get(f, [])
            h = sum(1 for r in v if sel(r))
            rate = 10000.0 * h / len(v) if v else 0.0
            rates[f] = rate
            cells.append(f"{h:>4}={rate:>7.2f}/万")
        ratio = f"{rates['jpg']/rates['png']:.1f}x" if rates.get("png") else "-"
        print(f"{lbl:<26} {cells[0]:>16} {cells[1]:>16} {ratio:>9}")

    # ★ ¥ 对已上线那条的影响
    print(f"\n¥ 排查(¥ 被当成数字算进去, 会把 mw 拉低、把负号宽比顶高)")
    for f in ("png", "jpg"):
        v = by.get(f, [])
        if not v:
            continue
        y = [r for r in v if r["yen"]]
        if not y:
            print(f"  {f}: 一张都没有"); continue
        now = sum(1 for r in y if r["bar_ratio"] >= BAR_HIGH)
        fix = sum(1 for r in y if r["bar_ratio_noyen"] >= BAR_HIGH)
        allnow = sum(1 for r in v if r["bar_ratio"] >= BAR_HIGH)
        print(f"  {f}: 疑似 ¥ {len(y):,} 张 = {100.0*len(y)/len(v):.2f}%; "
              f"其中现在被负号那条报出 {now} 张, 把 ¥ 剔掉后剩 {fix} 张; "
              f"占该格式全部负号误报的 {100.0*now/max(1,allnow):.0f}%")

    months = sorted({r["month"] for r in ok if r["month"]})
    if len(months) > 1:
        print(f"\n按月中位(各月一样 = 没随 app 改版漂)")
        print(f"{'':<10} " + " ".join(f"{m:>12}" for m in months))
        for f in ("png", "jpg"):
            cells = []
            for m in months:
                d = [r["dot_ratio"] for r in by.get(f, [])
                     if r["month"] == m and r["n_dot"] == 1]
                cells.append(f"{np.median(d):.4f}({len(d)})" if d else "-")
            print(f"  {f:<8} " + " ".join(f"{c:>12}" for c in cells))

    hits = [r for r in ok if r["n_dot"] == 1 and not (DOT_LOW <= r["dot_ratio"] <= DOT_HIGH)]
    print(f"\n小数点报出的 {len(hits)} 张(★ 一定要人工看一眼再下结论):")
    for r in sorted(hits, key=lambda r: r["dot_ratio"])[:30]:
        print(f"    面积比 {r['dot_ratio']:.4f}  面积 {r['dot_area']:.0f}  字高 {r['mh']:.0f}  "
              f"{r['fmt']}  {r['name'][:52]}")

    if args.sheet and hits:
        _make_sheet(args.input, hits, args.sheet)


def _make_sheet(input_dir, hits, out_path):
    """把报出来的金额行裁下来叠成一张 PNG。标签只写 ASCII(cv2 画不了中文)。"""
    tiles = []
    for r in sorted(hits, key=lambda r: r["dot_ratio"])[:40]:
        hit = next(Path(input_dir).rglob(r["name"]), None)
        if hit is None:
            continue
        img = cv2.imread(str(hit), cv2.IMREAD_COLOR)
        if img is None:
            continue
        H, W = img.shape[:2]
        box = _locate_amount(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), W, H)
        if box is None:
            continue
        bx0, by0, bx1, by1 = box
        m = int((by1 - by0) * 0.6)
        crop = img[max(0, by0 - m):min(H, by1 + m), :]
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, (900, max(1, int(crop.shape[0] * 900.0 / crop.shape[1]))))
        bar = np.full((26, 900, 3), 240, np.uint8)
        cv2.putText(bar, f"dot={r['dot_ratio']:.4f} area={r['dot_area']:.0f} "
                         f"mh={r['mh']:.0f} {r['fmt']} {r['name'][:44]}",
                    (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(np.vstack([bar, crop]))
    if not tiles:
        print("没有可拼的图"); return
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), np.vstack(tiles))
    print(f"拼图 -> {out_path}  (**一定要人工看一遍再下结论**)")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="负号宽比 + 小数点面积比(两条都不查表)")
    ap.add_argument("input", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=0, help="只抽这么多张, 0 = 全扫")
    ap.add_argument("--replay", type=Path, default=None, help="读之前存的 CSV 重做分析, 不重扫")
    ap.add_argument("--sheet", type=Path, default=None, help="把报出来的金额行拼成一张图")
    ap.add_argument("--since", type=str, default=None, help="只看这个日期(含)之后的, YYYYMMDD")
    ap.add_argument("--until", type=str, default=None, help="只看这个日期(含)之前的, YYYYMMDD")
    ap.add_argument("--seed", type=int, default=20260905)
    args = ap.parse_args()

    for k, v in (("--since", args.since), ("--until", args.until)):
        if v is not None and not re.fullmatch(r"\d{8}", v):
            ap.error(f"{k} 要写成 YYYYMMDD, 收到 {v!r}")

    if args.replay:
        with open(args.replay, encoding="utf-8-sig") as f:
            rows = [dict(r) for r in csv.DictReader(f)]
        for r in rows:
            for k in ("W", "H", "n_digit", "n_dot", "yen"):
                r[k] = int(float(r[k]))
            for k in ("mh", "mw", "bar_ratio", "bar_ratio_noyen", "dot_area", "dot_ratio"):
                r[k] = float(r[k])
        print(f"从 {args.replay} 读回 {len(rows):,} 张, 不重新扫图")
        if args.since or args.until:
            kept = []
            for r in rows:
                m = _TS.search(r["name"])
                if not m:
                    continue
                d = m.group(1)
                if (args.since and d < args.since) or (args.until and d > args.until):
                    continue
                kept.append(r)
            print(f"按日期筛: 留下 {len(kept):,}")
            rows = kept
        _analyse(args, rows)
        return

    files = [p for p in args.input.rglob("*") if p.suffix.lower() in _EXTS]
    print(f"目录里 {len(files):,} 个文件")
    if args.since or args.until:
        kept = []
        for p in files:
            m = _TS.search(p.name)
            if not m:
                continue
            d = m.group(1)
            if (args.since and d < args.since) or (args.until and d > args.until):
                continue
            kept.append(p)
        print(f"按日期筛: 留下 {len(kept):,}")
        files = kept
    if args.limit and len(files) > args.limit:
        import random
        random.Random(args.seed).shuffle(files)
        files = files[: args.limit]
        print(f"抽样 {len(files):,} 张")

    # 边扫边写, 中途停掉不丢
    rows, skipped = [], 0
    fh = writer = None
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fh = open(args.out, "w", newline="", encoding="utf-8-sig")
    try:
        for i, p in enumerate(files, 1):
            if i % 5000 == 0:
                print(f"  ...{i:,}/{len(files):,}, 量到 {len(rows):,}", flush=True)
            try:
                r = measure(str(p))
            except Exception:
                r = None
            if not r:
                skipped += 1
                continue
            rows.append(r)
            if fh is not None:
                if writer is None:
                    writer = csv.DictWriter(fh, fieldnames=list(r.keys()))
                    writer.writeheader()
                writer.writerow(r)
                if len(rows) % 500 == 0:
                    fh.flush()
    except KeyboardInterrupt:
        print(f"\n收到中断, 已量到的 {len(rows):,} 张照常分析。")
    finally:
        if fh is not None:
            fh.flush(); fh.close()

    print(f"判不了 {skipped:,} 张 ({100.0*skipped/max(1,len(files)):.1f}%)")
    if args.out:
        print(f"逐张测量值 -> {args.out}")
    if rows:
        _analyse(args, rows)


if __name__ == "__main__":
    main()
