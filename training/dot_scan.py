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
DOT_LOW, DOT_HIGH = 0.0200, 0.0365   # 面积那条, 已降级成只输出
DOT_FILL_LOW = 0.90                  # ★ 形状那条: 填充率低于此判可疑
                                     #   实心方点 1.000, 圆点 pi/4 = 0.785
                                     #   两侧实测: 圆点假图最大 0.8947, 真图 png 最小 0.9121
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


def corner_icon(img):
    """左上角那个图标的宽高。用来分页型:

    账单详情页是 `<` 返回箭头, 高宽比约 1.7(实测 56x33)。
    商家账单页是 `X` 关闭图标, 近正方形(实测 44x44)。
    ★ 这两种页用的金额字体不一样 —— 商家账单页的负号明显更宽,
      所以拿账单详情页标出来的阈值去判它, 会整页型误报。
    返回 (w, h) 或 None。
    """
    H, W = img.shape[:2]
    g = cv2.cvtColor(img[0:int(H * 0.12), 0:int(W * 0.13)], cv2.COLOR_BGR2GRAY)
    _, dark = cv2.threshold(g, 169, 255, cv2.THRESH_BINARY_INV)
    n, _, st, _ = cv2.connectedComponentsWithStats(dark, 8)
    best = None
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < 100 or not (25 <= h <= 80) or not (15 <= w <= 80):
            continue
        if best is None or y > best[1]:      # 取最靠下的, 导航栏在状态栏下面
            best = (x, y, w, h)
    return (best[2], best[3]) if best else None


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

    # ★ 1. 贴着切块左右边缘的块是**被切断的**, 尺寸不可信, 先剔掉。
    #   实测三张 -499.99 的首块 x=0、高只有其余的 0.857, 旧版据此误判成 ¥。
    #   真正的 ¥ 和真图首字都离边缘很远(x 77~139)。
    sub_w = fg.shape[1]
    digits = [k for k in digits if k[0] > 0 and k[0] + k[2] < sub_w]
    if len(digits) < 4:
        return None

    # ★ 2. 把 ¥ 从数字里剔出去。不剔的后果: n_digit 多算 1,
    #   而 n_digit>=6 正是"大额(>=1000元)"的代用指标 ——
    #   ¥300.00 这种 300 元的图会被算成大额, 富集数字全部虚高。
    #
    #   ¥ 不止一种字形, 单一判据抓不全, 实测要三选一(参照是**其余**数字):
    #     a) 又宽又稀  宽 1.41~1.62 倍且笔画疏     -> 宽版 ¥
    #     b) 明显偏矮  高 0.901 倍                 -> ¥300.30 那种
    #     c) 后面留大空 间隔 1.021 倍中位宽         -> ¥-999.92 那种
    #   对照真图首字: 宽比 0.55~1.00, 高比 0.98~1.00, 间隔 0.23~0.57, 三条都不沾。
    digits.sort(key=lambda k: k[0])
    yen = False
    if len(digits) >= 5:
        rest = digits[1:]
        rest_w = float(np.median([k[2] for k in rest]))
        rest_h = float(np.median([k[3] for k in rest]))
        rest_fill = float(np.median([k[4] / float(k[2] * k[3]) for k in rest]))
        d0 = digits[0]
        f0 = d0[4] / float(d0[2] * d0[3]) if d0[2] and d0[3] else 1.0
        gap0 = (rest[0][0] - (d0[0] + d0[2])) / rest_w if rest_w else 0.0
        if rest_w > 0 and rest_h > 0 and rest_fill > 0:
            wide_sparse = d0[2] > 1.25 * rest_w and f0 < 0.85 * rest_fill
            short = d0[3] < 0.95 * rest_h
            # 宽度条件防误伤: 金额如 -1.50 首字后面就是小数点, 间隔天生大,
            # 但那种情况首字是窄的 "1"
            wide_gap = gap0 > 0.85 and d0[2] > 0.95 * rest_w
            if wide_sparse or short or wide_gap:
                yen = True
                digits = rest
    if len(digits) < 4:
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

    # ¥ 已经在前面剔掉了, 所以 mw 就是干净的数字中位宽
    ds = digits
    mw_noyen = mw

    ic = corner_icon(img)
    if ic is None:
        page = "unknown"
    else:
        ar = ic[1] / ic[0]
        page = "close" if 0.85 <= ar <= 1.15 else ("back" if ar >= 1.4 else "other")

    # ★ 小数点的**形状**。经理 2026-09-05: "昨晚那个图就只有小数点是圆的了"
    #   —— 判别的关键是形状不是大小。原来量的 面积/数字高² 把两者混在一起了:
    #   一个圆点和一个更小的方点, 面积可以完全一样。
    #   填充率 = 面积 / 外接框面积。实心方点 = 1.000, 圆点 = pi/4 = 0.785。
    #   实测真图 png 中位 1.000 / p1 0.997, 阈值 0.90 在 4,302 张上零报出。
    d_w = d_h = 0
    d_fill = 0.0
    if len(dots) == 1:
        d_w, d_h = dots[0][2], dots[0][3]
        d_fill = dots[0][4] / float(d_w * d_h) if d_w and d_h else 0.0

    # ★ 金额的颜色。经理 2026-09-06: "他们需求需要把文字的颜色也要输出 /
    #   我之前是直接二值化了 现在也得改下"。我们这条管线一样是 Otsu 二值化,
    #   颜色全丢了。这里在二值化之前把金额笔画的原始 BGR 留下来。
    #   实测服务器报出的图里有一批金额是绿色的, 当时当成渲染变体放过了 ——
    #   如果颜色和字体相关, 它对 flux 那边的字体分类也是一个现成特征。
    amt = img[cy0:cy1, cx0:cx1]
    ink = amt[sub <= int(cv2.threshold(sub, 0, 255,
                                       cv2.THRESH_BINARY | cv2.THRESH_OTSU)[0])]
    if len(ink) >= 50:
        ib, ig, ir = (float(ink[:, i].mean()) for i in range(3))
    else:
        ib = ig = ir = 0.0

    # ---- 与数字内容无关的字体不变量 ----
    # 数字本身**不是恒定字形**("1" 比 "0" 窄得多), 所以凡是由数字宽度推出来的量
    # 都会被金额内容污染: 实测 mw/mh 分布很紧(11.4%)但大额富集只有 3.59 倍,
    # 远不如负号(12.2~16.0)和圆点(14.4)。下面这些量都不挑数字内容。
    top = float(np.median([k[1] for k in digits]))
    bar_vpos = ((bar[1] + bar[3] / 2.0) - top) / mh          # 负号坐得多高
    right_of_bar = [k for k in ds if k[0] >= bar[0] + bar[2]]
    bar_gap = ((right_of_bar[0][0] - (bar[0] + bar[2])) / mh
               if right_of_bar else 0.0)

    # 字距: 只取相邻且间隔在一个字宽上下的一对, 跨小数点那一跳要排除,
    # 否则变异系数会被那个不规则间隔顶高(实测从 0.05 顶到 0.19)。
    gaps = [b[0] - a[0] for a, b in zip(ds, ds[1:])
            if 0.5 * mw < b[0] - a[0] < 1.8 * mw]
    if len(gaps) >= 3:
        ga = np.array(gaps, float)
        pitch = float(np.median(ga)) / mh
        pitch_cv = float(ga.std() / ga.mean()) if ga.mean() else 0.0
    else:
        pitch = pitch_cv = 0.0

    # 笔画粗细: 距离变换在字形内部的最大值约等于半个笔宽
    st = []
    for k in digits:
        m = fg[k[1]:k[1] + k[3], k[0]:k[0] + k[2]]
        if m.size == 0:
            continue
        dt = cv2.distanceTransform(
            cv2.copyMakeBorder(m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0),
            cv2.DIST_L2, 5)
        if dt.max() > 0:
            st.append(float(dt.max()) * 2.0)
    stroke = (float(np.median(st)) / mh) if len(st) >= 3 else 0.0
    height_cv = float(dh.std() / dh.mean()) if dh.mean() else 0.0

    # 小数点竖直位置。★ 这里要用**没被基线过滤**的候选, 上面的 dots 是按
    # abs(下缘-基线) <= 0.06*mh 挑的, 拿它算就是循环论证。
    dots_free = [k for k in glyphs if k not in digits and k not in bars
                 and k[3] <= 0.30 * mh and k[2] <= 0.60 * mw]
    dot_vpos = (((dots_free[0][1] + dots_free[0][3]) - baseline) / mh
                if len(dots_free) == 1 else 0.0)

    return dict(
        name=os.path.basename(path), W=W, H=H,
        page=page, icon_w=(ic[0] if ic else 0), icon_h=(ic[1] if ic else 0),
        dot_w=d_w, dot_h=d_h, dot_fill=round(d_fill, 4),
        ink_b=round(ib, 1), ink_g=round(ig, 1), ink_r=round(ir, 1),
        green=int(ig > ir + 12),
        mh=mh, mw=mw, n_digit=len(digits),
        bar_ratio=bar[2] / mw,
        bar_ratio_noyen=bar[2] / mw_noyen,
        dot_area=float(dots[0][4]) if len(dots) == 1 else 0.0,
        dot_ratio=(dots[0][4] / (mh * mh)) if len(dots) == 1 else 0.0,
        n_dot=len(dots), yen=int(yen),
        bar_vpos=round(bar_vpos, 5), bar_gap=round(bar_gap, 5),
        pitch=round(pitch, 5), pitch_cv=round(pitch_cv, 5),
        stroke=round(stroke, 5), height_cv=round(height_cv, 5),
        dot_vpos=round(dot_vpos, 5),
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

    # ★ 不要写"误报率"。服务器那个池子是**真实进件流**, 里面本来就有假图
    #   (箭头那条已经在里面抓到过 2 张和经理给的假图同指纹的)。
    #   所以这一栏只能叫"报出率" —— 里面有多少是真欺诈, 要靠人工看或者线上的处置结果才知道。
    #   本机 white/TempFakeImages 那个池子是干净子集, 那里才能叫误报率。
    print(f"\n逐条分解 (报出率; 服务器池是真实进件流, 里面可能有真欺诈, 不要当成纯误报)")
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

    # ★★ 小数点的形状 —— 这才是经理指的那个判别点
    print(f"\n★ 小数点填充率(面积/外接框): 实心方点=1.000, 圆点=pi/4=0.785")
    print(f"{'':<10} {'张数':>8} {'中位':>8} {'p1':>8} {'<0.90':>14} {'<0.85':>14}")
    for f in ("png", "jpg"):
        v = [r for r in ok if r["fmt"] == f and r["n_dot"] == 1 and r.get("dot_fill", 0) > 0]
        if not v:
            continue
        a = np.array([r["dot_fill"] for r in v])
        n90 = sum(1 for x in a if x < 0.90); n85 = sum(1 for x in a if x < 0.85)
        print(f"  {f:<8} {len(v):>8,} {np.median(a):>8.4f} {np.percentile(a,1):>8.4f} "
              f"{n90:>5}={10000.0*n90/len(v):>7.2f}/万 {n85:>5}={10000.0*n85/len(v):>7.2f}/万")
    print("  ★ jpg 那一侧会被压缩把方点的角磨圆, 所以只在 png 上用这条")

    # ★ 颜色 —— 经理说下游要输出文字颜色
    grn = [r for r in ok if r.get("green")]
    print(f"\n★ 金额笔画颜色: 绿色 {len(grn):,} 张 = {100.0*len(grn)/len(ok):.2f}%")
    if grn:
        for lbl, v in (("绿色金额", grn), ("其它", [r for r in ok if not r.get("green")])):
            d = [r["dot_fill"] for r in v if r["n_dot"] == 1 and r.get("dot_fill", 0) > 0]
            b = [r["bar_ratio"] for r in v]
            if not d:
                continue
            print(f"  {lbl:<8} n={len(v):>6,}  小数点填充率中位 {np.median(d):.4f}  "
                  f"负号宽比中位 {np.median(b):.4f}")
        print("  ★ 两行差得大, 就说明颜色和字体是绑在一起的, 那颜色能直接当字体分类的特征")

    # ★★ 按页型拆开 —— 这是 2026-09-05 查出来的关键混杂
    #   商家账单页(左上角是 X 关闭图标)和账单详情页(左上角是 < 返回箭头)
    #   用的金额字体不一样, 负号明显更宽。拿详情页标的阈值去判它会整页型误报。
    print(f"\n★ 按页型拆(左上角图标: back = < 返回箭头, close = X 关闭图标)")
    print(f"{'页型':<10} {'张数':>8} {'占比':>7} {'负号中位':>9} {'负号>=0.78':>14}")
    for pg in ("back", "close", "other", "unknown"):
        v = [r for r in ok if r.get("page") == pg]
        if not v:
            continue
        b = np.array([r["bar_ratio"] for r in v])
        h = sum(1 for x in b if x >= BAR_HIGH)
        print(f"  {pg:<8} {len(v):>8,} {100.0*len(v)/len(ok):>6.2f}% {np.median(b):>9.4f} "
              f"{h:>5} = {10000.0*h/len(v):>8.1f}/万")
    print("  ★ close 那一行的报出率如果远高于 back, 就说明线上那条在整页型误伤商家账单页。")

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

    # ★ 判据换成形状(填充率), 不再用面积。面积那条对真实圆点假图只抓到 10%。
    dot_hits = [r for r in ok if r["n_dot"] == 1 and 0 < r.get("dot_fill", 0) < DOT_FILL_LOW]
    bar_hits = [r for r in ok if r["bar_ratio"] >= BAR_HIGH]
    for r in dot_hits: r["_why"] = f"fill={r['dot_fill']:.4f}"
    for r in bar_hits: r["_why"] = f"BAR={r['bar_ratio']:.4f}"

    # ★★ 这才是最要紧的一个数: 圆点里有多少是负号那条**抓不到**的
    dot_only = [r for r in dot_hits if r["bar_ratio"] < BAR_HIGH]
    print(f"\n★★ 圆点(填充率 < {DOT_FILL_LOW})报出 {len(dot_hits)} 张:")
    for f in ("png", "jpg"):
        v = [r for r in ok if r["fmt"] == f]
        h = [r for r in dot_hits if r["fmt"] == f]
        o = [r for r in dot_only if r["fmt"] == f]
        if not v:
            continue
        print(f"    {f}  {len(h):>4}/{len(v):<7,} = {10000.0*len(h)/len(v):>7.2f}/万"
              f"   其中负号抓不到的 {len(o):>4} = {10000.0*len(o)/len(v):>7.2f}/万")
    print(f"  ★ 「负号抓不到的」那一栏就是这条判据的**净增价值**。"
          f"接近 0 就说明它只是印证, 不是新抓手。")

    print(f"\n填充率最低的 30 张(★ 一定要人工看一眼再下结论):")
    for r in sorted(dot_hits, key=lambda r: r["dot_fill"])[:30]:
        tag = "  <- 负号抓不到" if r["bar_ratio"] < BAR_HIGH else ""
        print(f"    填充 {r['dot_fill']:.4f}  {r.get('dot_w',0)}x{r.get('dot_h',0)}  "
              f"页型 {r.get('page','?'):<7} {r['fmt']}  负号 {r['bar_ratio']:.4f}  "
              f"{r['name'][:44]}{tag}")

    # ★ 已上线那条报出的也要看。服务器实测它报出率是小数点这条的六七倍,
    #   但没人知道那里面有多少是真欺诈 —— 这是目前最该搞清楚的一件事。
    print(f"\n★ 已上线的负号那条报出的 {len(bar_hits)} 张(比小数点那条多得多, 更该人工看):")
    for r in sorted(bar_hits, key=lambda r: -r["bar_ratio"])[:30]:
        print(f"    负号宽比 {r['bar_ratio']:.4f}  字高 {r['mh']:.0f}  "
              f"{r['fmt']}  {r['name'][:52]}")

    if args.sheet:
        want = {"dot": dot_hits, "minus": bar_hits, "both": dot_hits + bar_hits}[args.sheet_of]
        if want:
            _make_sheet(args.input, want, args.sheet)


def _make_sheet(input_dir, hits, out_path):
    """把报出来的金额行裁下来叠成一张 PNG。标签只写 ASCII(cv2 画不了中文)。"""
    tiles = []
    for r in hits[:40]:
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
        cv2.putText(bar, f"{r.get('_why','')} dot={r.get('dot_w',0)}x{r.get('dot_h',0)} "
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
    ap.add_argument("--sheet-of", choices=["dot", "minus", "both"], default="both",
                    help="拼图要画哪一条报出的: 小数点 / 已上线的负号 / 两条都画(默认)")
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
            r.setdefault("page", "unknown")
            r.setdefault("icon_w", 0); r.setdefault("icon_h", 0)
            for k in ("dot_w", "dot_h", "green"):
                r.setdefault(k, 0)
            for k in ("dot_fill", "ink_b", "ink_g", "ink_r"):
                r.setdefault(k, 0.0)
            for k in ("W", "H", "n_digit", "n_dot", "yen", "icon_w", "icon_h",
                      "dot_w", "dot_h", "green"):
                r[k] = int(float(r[k]))
            for k in ("mh", "mw", "bar_ratio", "bar_ratio_noyen", "dot_area",
                      "dot_ratio", "dot_fill", "ink_b", "ink_g", "ink_r"):
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
