r"""蓝图 ¥ 符号字形扫描 —— 不用模型, 只量 ¥ 的笔画粗细和同图数字笔画粗细的比值。

背景
----
经理 2026-09-16: "你试试用上次识别负号的方式 看看能不能识别出来" "只能硬写算法了"
"差距太小了" "不行直接图片定位" "但是也要分安卓和苹果"。

客户(国王支付-玄武)给的判据: **蓝图苹果的 ¥ 是细线, 细线最底部跟金额底部一样;
安卓的是粗线, 有些在中间有些在底部跟金额一样**。

在经理给的两张原图上量过(2.png 假 / true.png 真):

    真图   ¥ 竖笔  8 像素   数字笔画 16 像素   比值 0.500
    假图   ¥ 竖笔 12 像素   数字笔画 16 像素   比值 0.750

★ 物理解释: 真图的 ¥ 是**回退字形**, 走系统字体, 比金额数字细一截;
  P 图的人用金额那个粗体把 ¥ 一起打出来了, 于是 ¥ 和数字一样粗。
  经理原话"他p图用错字体了", 玄武原话"他应该使用不出细线的符号", 说的是同一件事。

★★ 量**同一张图内部的比值**, 不量绝对像素: 分辨率、缩放、机型全都抵消掉。
   和 MinusCheck 量"负号宽/数字宽"是同一个路子。

★★★ 玄武说的"底边对齐"也量到了(真 0.010/0.020, 假 0.028/0.038), 方向是对的,
   但**只差数字高的 2%, 原图上就 2 个像素**, 压缩一次就没了。
   所以 bot_gap 只做输出, **不当判据**。

复用了什么
----------
- 定位: `locate_blue.locate_amount_blue` —— 已有的蓝图金额定位器, 实测 98.0%。
  **不要另写一个**: 2026-09-15 已经因为重复造 `pinyin_probe` 的轮子栽过一次。
- 裁块二值化: 抄 `glyph_baseline._reextract` 的配方(Otsu + 按前景占比自动纠正极性 +
  左侧大幅外扩 + **不设高度下限**)。这里只多留一份 labels 图, 因为要按块量游程。

覆盖边界
--------
- 只对**蓝底转账页**有效。白底账单详情页归 MinusCheck / DotCheck / FontCheck, 这里跳过。
- ★★★★★ **安卓真图和苹果假图的数值一模一样**(六张样本实测都是 0.750)。
  所以这条**必须先分安卓/苹果再判**; 不分平台直接卡一个阈值,
  会把每一张真安卓图都报成假图。本脚本**只量不判**, 分平台交给调用方。
- 量的是"这张图的 ¥ 和同图数字的粗细比", **不是**"这张图一定是假的"。
- 只认"¥ 画错了"这一种痕迹。照着正确字体把 ¥ 打出来就绕过去了。
- 分辨率低的时候 ¥ 竖笔只有 4~6 个像素, 一个像素的压缩误差就能翻盘。
  看报表里"按数字高分档"那一段, 低档的数**不要拿来定阈值**。

用法
----
  # 先小样本看分布(不给阈值就只量不报)
  python training/yen_scan.py c:\projects\China\TempFakeImages --limit 5000 --out D:\probe\yen.csv

  # 标定完之后再用阈值报出偏离的
  python training/yen_scan.py c:\projects\China\TempFakeImages --low 0.40 --high 0.62

  # 只看某个时期(app 改过版的话字形可能变)
  python training/yen_scan.py c:\projects\China\TempFakeImages --since 20260901

★ 报表里有**按分辨率分组的比值中位数**。明显分成两簇 = 苹果和安卓两种字形,
  和玄武说的对得上; 分不成两簇说明这个前提在数据上不成立, 要回头重看。

★★★ 本脚本**只量不判**。真正的判定口径在 `demo/YenCheck.cs`, 而且是**两条判据**:
    竖笔比(分机型, 苹果 [0.50,0.65] / 安卓 [0.64,0.80])
    数字笔画粗细(不分机型, [0.145,0.175]) —— 落在带外说明**数字**字体不对, 不是 ¥ 的问题
  重新标定的时候两条都要看, 只看竖笔比会把"数字偏粗"那一类算漏。

★★ 机型**不要用分辨率去猜**: 实测被裁过/缩过的 iPhone 截图会落到非 iPhone 分辨率上
   (1260x2736 / 1280x2781 / 1280x2774 / 1280x2769 / 960x2079 整组 100% 像苹果),
   拿分辨率当机型代理的话, 光这一项就造出 2.37% 的假报出。线上要用状态栏那个设备模型。

**只读**: 只读图片, 只往 --out 写。
"""
from __future__ import annotations

import argparse
import collections
import csv
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFile

# ★★★ 截断的 JPEG: OpenCV(也就是 C# 那条线上路径)照样能解, PIL 默认直接报错。
#   不打开这个开关, 标定时会把这类图当"读不了"跳过, **而线上是判的** —— 口径就对不上了。
#   实测 400 张里有 2 张(0.5%)是这种, 都只截掉了末尾几个字节, 金额行完好。
ImageFile.LOAD_TRUNCATED_IMAGES = True

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from locate_blue import is_blue_page, locate_amount_blue  # noqa: E402

_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_TS = re.compile(r"_(\d{8})\d{6}")   # 文件名里的时间戳, 形如 _20260902211530

# ¥ 高 / 数字高 的合理区间。六张样本实测 0.691~0.717, 卡宽一点当护栏。
YEN_H_LOW, YEN_H_HIGH = 0.60, 0.80
# 横杠判定: 某一行的前景占 ¥ 宽度的这个比例以上就算横杠所在行。
BAR_FILL = 0.70


def _run_spans(row: np.ndarray):
    """一行里每一段连续 True 的 (起, 止)。止是开区间。"""
    a = np.concatenate(([False], row.astype(bool), [False]))
    d = np.diff(a.astype(np.int8))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def _runs(row: np.ndarray) -> np.ndarray:
    """一行里每一段连续 True 的长度(整数, 只用来找横杠)。"""
    a = np.concatenate(([False], row.astype(bool), [False]))
    d = np.diff(a.astype(np.int8))
    return np.flatnonzero(d == -1) - np.flatnonzero(d == 1)


def _stroke_widths(gray: np.ndarray, mask: np.ndarray, bg: float, fg: float,
                   margin: int = 2) -> list[float]:
    r"""逐行量每一段笔画的**等效宽度**(亚像素), 返回全部段的宽度。

    ★★★★★ 为什么不数二值化之后的像素个数
    -----------------------------------
    `glyph_baseline.py` 写过: "细横杠的宽度对二值化阈值极敏感, 差一个像素就差约 10%"。
    实测就是这样: 同样六张图, 用 `>200` 卡白字量出真假 0.500 / 0.750,
    换成 Otsu 就变成 0.625 / 0.706 —— **判据强弱完全取决于阈值定在哪**, 这不行。

    根子在于抗锯齿: 笔画边缘是一道从底色到字色的斜坡。阈值卡高一点, 细笔画被削掉的
    比例比粗笔画大; 卡低一点反过来。**细的那一根永远被放大误差。**

    等效宽度绕开了这件事: 把这一段的灰度**积分**再除以字底色差 ——
        宽度 = Σ (I - 底色) / (字色 - 底色)
    模糊只是把能量摊开, **积分不变**。所以这个量对阈值、抗锯齿、JPEG 轻度模糊都不敏感,
    而且是连续值, 不会被整数像素卡死(4 像素和 6 像素之间本来什么都取不到)。

    ★ margin 往两边各让 2 个像素, 把斜坡收进来; 同时用相邻段夹住窗口, 防止串到隔壁笔画。

    ★★ 口径(**C# 版必须一模一样, 否则两边的数对不上**):
        - 传进来的 gray / mask 是**这一块自己的外接框**, 窗口夹在框内, 不外扩到框外;
        - 所以贴着框边的那一笔会少收半个像素的斜坡。这是**固定偏差, 两边都有,
          比值里基本抵消**; 实测同一张真图压缩前后比值 0.5880 / 0.5884, 稳得住。
        - 要改口径可以, 但**必须两边一起改**, 然后重新标定阈值。
    """
    out: list[float] = []
    amp = fg - bg
    if amp == 0:
        return out
    W = mask.shape[1]
    for y in range(mask.shape[0]):
        spans = _run_spans(mask[y])
        for i, (s, e) in enumerate(spans):
            lo = max(0, s - margin)
            hi = min(W, e + margin)
            if i > 0:                      # 不越过左边那一段
                lo = max(lo, spans[i - 1][1])
            if i + 1 < len(spans):         # 不越过右边那一段
                hi = min(hi, spans[i + 1][0])
            if hi <= lo:
                continue
            prof = (gray[y, lo:hi].astype(np.float64) - bg) / amp
            out.append(float(np.clip(prof, 0.0, 1.0).sum()))
    return out


def _levels(gray: np.ndarray, fgm: np.ndarray):
    """从裁块里估底色和字色。返回 (底色, 字色) 或 None(对比度不够)。"""
    if fgm.sum() < 20 or (~fgm).sum() < 20:
        return None
    m_fg = float(np.median(gray[fgm]))
    m_bg = float(np.median(gray[~fgm]))
    bg = float(np.percentile(gray[~fgm], 50))
    # 取前景的亮端/暗端当字色, 避开被抗锯齿拉平的边缘像素
    fg = float(np.percentile(gray[fgm], 90 if m_fg > m_bg else 10))
    if abs(fg - bg) < 30:
        return None
    return bg, fg


def _reextract_mask(rgb: np.ndarray, box):
    """行内重取连通块, 连 labels 一起返回。

    ★ 配方抄自 `glyph_baseline._reextract`, 一字不改的三条:
        1. **不设高度下限** —— 定位阶段的高度闸会把小数点之类滤掉;
        2. 左侧大幅外扩 padx —— 定位框的 x0 是按字算的;
        3. Otsu 之后按前景占比自动纠正极性 —— 白图深字/蓝图白字极性相反。
      多出来的只有 labels: 本脚本要按块逐行量游程, 光有外接框不够。
    """
    x0, y0, x1, y1 = box
    pad = max(2, int((y1 - y0) * 0.15))
    padx = max(pad, int((x1 - x0) * 0.30))
    H, W = rgb.shape[:2]
    sub = rgb[max(0, y0 - pad):min(H, y1 + pad), max(0, x0 - padx):min(W, x1 + pad)]
    if sub.size == 0:
        return [], None, None
    g = cv2.cvtColor(sub, cv2.COLOR_RGB2GRAY)
    th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    if th.mean() > 127:                       # 前景过半 = 极性反了
        th = 255 - th
    n, lab, st, _ = cv2.connectedComponentsWithStats(th, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < 6:                             # 只挡真正的噪点, 不设高度下限
            continue
        out.append(dict(x=int(x), y=int(y), w=int(w), h=int(h), area=int(a),
                        mask=(lab[y:y + h, x:x + w] == i),
                        gray=g[y:y + h, x:x + w]))
    out.sort(key=lambda c: c["x"])
    # ★ 二值图只用来**找**笔画在哪; 笔画有多宽是拿灰度积分算的, 见 _stroke_widths
    return out, g, (th > 0)


def measure(path):
    """量一张图。返回 (数据字典, None) 或 (None, 跳过原因)。"""
    try:
        with Image.open(path) as im:
            rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
    except Exception:
        return None, "读不了"
    if rgb.ndim != 3:
        return None, "不是彩色图"
    H, W = rgb.shape[:2]

    if not is_blue_page(rgb):
        return None, "不是蓝底页"
    loc = locate_amount_blue(rgb)
    if not loc:
        return None, "定位不到金额行"

    glyphs, gcrop, fgm = _reextract_mask(rgb, (loc[0], loc[1], loc[2], loc[3]))
    if len(glyphs) < 4:                       # ¥ 加至少三个数字
        return None, "块数不够"
    lv = _levels(gcrop, fgm)
    if lv is None:
        return None, "金额行对比度不够"
    bg, fg = lv

    yen = glyphs[0]                           # 最左边那块
    rest = glyphs[1:]
    tall = max(k["h"] for k in rest)
    cw = gcrop.shape[1]          # ★ 裁块宽度。不能拿"最右那块的右边"当界,
                                 #   那样每次都会把最右边一个数字扔掉(C# 版是按裁块宽算的)
    # ★ 贴着裁块左右边缘的块是被切断的, 尺寸不可信 —— 和 MinusCheck 同样的处理
    digits = [k for k in rest
              if k["h"] >= 0.80 * tall and k["x"] > 0 and k["x"] + k["w"] < cw]
    if len(digits) < 3:
        return None, "数字不够三个"

    d_top = min(k["y"] for k in digits)
    d_bot = max(k["y"] + k["h"] for k in digits)
    d_h = d_bot - d_top
    if d_h <= 0:
        return None, "数字高算不出来"

    hs = np.array([k["h"] for k in digits], float)
    if hs.std() / hs.mean() > 0.08:
        return None, "数字高度不齐"

    # ---- ¥ 的两道护栏: 高度比例 + 必须正好两道横杠 ----
    # ★ 没有这两条, 版式一变就会把别的字当成 ¥ 量, 而且量出来的数看着很正常。
    yen_h_ratio = yen["h"] / d_h
    if not (YEN_H_LOW <= yen_h_ratio <= YEN_H_HIGH):
        return None, "最左块高度比例不像 ¥"
    ymask = yen["mask"]
    bar_rows = ymask.sum(axis=1) >= BAR_FILL * yen["w"]
    bars = _runs(bar_rows)
    if bars.size != 2:
        return None, f"最左块有 {bars.size} 道横杠, 不像 ¥"

    # ---- 笔画粗细(亚像素等效宽度, 不是数像素) ----
    d_runs: list[float] = []
    for k in digits:
        d_runs.extend(_stroke_widths(k["gray"], k["mask"], bg, fg))
    if not d_runs:
        return None, "数字量不出笔画"
    d_stroke = float(np.median(d_runs))
    if d_stroke <= 0:
        return None, "数字笔画为零"

    # ¥ 竖笔: 取**最后一道横杠以下**那几行, 那里只剩竖笔。
    # ★ 不要固定取"底部百分之多少": 有的字形横杠压得很低, 会把横杠当竖笔量, 比值直接翻倍。
    last_bar_end = int(np.flatnonzero(bar_rows)[-1]) + 1
    stem_rows = ymask[last_bar_end:]
    if stem_rows.shape[0] < 2:
        return None, "横杠以下没有竖笔"
    s_runs = _stroke_widths(yen["gray"][last_bar_end:], stem_rows, bg, fg)
    if not s_runs:
        return None, "竖笔量不出来"
    y_stem = float(np.median(s_runs))

    # 横杠厚度也用等效宽度量: 把 ¥ 转置, 横杠就变成竖直的一段
    bar_lo = int(np.flatnonzero(bar_rows)[0])
    bmask = ymask[bar_lo:last_bar_end].T
    bgray = np.ascontiguousarray(yen["gray"][bar_lo:last_bar_end].T)
    b_runs = _stroke_widths(bgray, bmask, bg, fg)
    y_bar = float(np.median(b_runs)) if b_runs else float(np.median(bars))

    return dict(
        W=W, H=H,
        digit_h=int(d_h),
        n_digit=len(digits),
        d_stroke=round(d_stroke, 2),
        y_stem=round(y_stem, 2),
        y_bar=round(y_bar, 2),
        stem_ratio=round(y_stem / d_stroke, 4),        # ★ 主判据候选
        bar_ratio=round(y_bar / d_stroke, 4),
        yen_h_ratio=round(yen_h_ratio, 4),
        # 玄武说的底边对齐, 只输出不判定 —— 差距只有数字高的 2%
        bot_gap=round((d_bot - (yen["y"] + yen["h"])) / d_h, 4),
        stroke_norm=round(d_stroke / d_h, 4),          # 笔画/字高, 同字体应当很稳
    ), None


def _pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="扫描蓝图 ¥ 符号的笔画粗细比")
    ap.add_argument("input", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=0, help="只抽这么多张, 0 = 全扫")
    ap.add_argument("--min-digit-h", type=int, default=0,
                    help="数字高低于这个值的不进统计。0 = 都进, 只在报表里分档看")
    ap.add_argument("--low", type=float, default=None,
                    help="比值低于此值报出。不给就只量不报")
    ap.add_argument("--high", type=float, default=None,
                    help="比值高于此值报出。不给就只量不报")
    ap.add_argument("--min-group", type=int, default=200,
                    help="某个分辨率不够这么多张就不单列。默认 200")
    ap.add_argument("--since", type=str, default=None, help="只看这个日期(含)之后的, YYYYMMDD")
    ap.add_argument("--until", type=str, default=None, help="只看这个日期(含)之前的, YYYYMMDD")
    ap.add_argument("--seed", type=int, default=20260916)
    args = ap.parse_args()

    for _k, _v in (("--since", args.since), ("--until", args.until)):
        if _v is not None and not (len(_v) == 8 and _v.isdigit()):
            raise SystemExit(f"!! {_k} 要写成 YYYYMMDD 八位数字, 收到的是 {_v!r}")

    files = []
    for dp, _, fns in os.walk(args.input):
        for fn in fns:
            if os.path.splitext(fn)[1].lower() in _EXTS:
                files.append(os.path.join(dp, fn))
    print(f"候选 {len(files):,} 张", flush=True)
    if not files:
        raise SystemExit("!! 一张图都没找到")

    if args.since or args.until:
        kept, nodate, outside = [], 0, 0
        for p in files:
            m = _TS.search(os.path.basename(p))
            if not m:
                nodate += 1
                continue
            d = m.group(1)
            if (args.since and d < args.since) or (args.until and d > args.until):
                outside += 1
                continue
            kept.append(p)
        rng = f"{args.since or '不限'} ~ {args.until or '不限'}"
        print(f"按日期筛({rng}): 留下 **{len(kept):,}** 张, 不在范围内 {outside:,}, "
              f"取不到日期 {nodate:,}", flush=True)
        if not kept:
            raise SystemExit("!! 这个日期范围里一张都没有")
        files = kept

    if args.limit and args.limit < len(files):
        import random
        random.Random(args.seed).shuffle(files)
        files = files[:args.limit]

    rows = []
    skips: collections.Counter = collections.Counter()
    for i, p in enumerate(files, 1):
        m, why = measure(p)
        if m is None:
            skips[why] += 1
        else:
            m["file"] = os.path.basename(p)
            mo = _TS.search(os.path.basename(p))
            m["month"] = mo.group(1)[:6] if mo else ""
            rows.append(m)
        if i % 2000 == 0:
            print(f"  {i:,}/{len(files):,}  量到 {len(rows):,}", flush=True)

    print(f"\n量到 **{len(rows):,}** 张 / 共 {len(files):,} 张")
    print("跳过的原因:")
    for why, n in skips.most_common():
        print(f"    {why:<28} {n:,}")
    if not rows:
        raise SystemExit("!! 一张都没量到 —— 先确认这批是不是蓝底转账页")

    # ---- 按数字高分档: 低分辨率的数不能拿来定阈值 ----
    print(f"\n{'数字高':>12}{'张数':>9}{'比值中位':>10}{'p5':>8}{'p95':>8}{'笔画/字高':>11}")
    for lo, hi in [(0, 60), (60, 80), (80, 100), (100, 130), (130, 10 ** 6)]:
        g = [r for r in rows if lo <= r["digit_h"] < hi]
        if not g:
            continue
        v = np.array([r["stem_ratio"] for r in g])
        sn = np.array([r["stroke_norm"] for r in g])
        name = f"{lo}~{hi}" if hi < 10 ** 6 else f">={lo}"
        print(f"{name:>12}{len(g):>9}{np.median(v):>10.3f}{_pct(v, 5):>8.3f}"
              f"{_pct(v, 95):>8.3f}{np.median(sn):>11.4f}")

    use = [r for r in rows if r["digit_h"] >= args.min_digit_h]
    if not use:
        raise SystemExit(f"!! --min-digit-h {args.min_digit_h} 把所有图都筛掉了")
    vals = np.array([r["stem_ratio"] for r in use])

    # ---- 直方图: 是不是两簇, 一眼就能看出来 ----
    print(f"\n比值分布 (n={len(use):,}, 数字高 >= {args.min_digit_h})")
    edges = np.arange(0.20, 1.26, 0.05)
    hist, _ = np.histogram(vals, bins=edges)
    top = max(int(hist.max()), 1)
    for k in range(len(hist)):
        print(f"  {edges[k]:.2f}~{edges[k+1]:.2f} {hist[k]:>8,} {'#' * int(60 * hist[k] / top)}")
    print(f"  中位 {np.median(vals):.4f}   p5 {_pct(vals,5):.4f}   p95 {_pct(vals,95):.4f}")

    # ---- 数字自己的笔画粗细 ----
    # ★ 这一栏不能漏: YenCheck.cs 是**两条判据**(竖笔比 + 数字笔画粗细), 只看竖笔比
    #   重新标定的话, 会把"数字偏粗"那一类的报出量算漏。
    sn = np.array([r["stroke_norm"] for r in use])
    print(f"\n数字笔画粗细(笔画/字高) —— 和机型无关, 两个平台一样")
    print("  " + "  ".join(f"p{q}={np.percentile(sn, q):.4f}" for q in (0.1, 1, 50, 99, 99.9)))
    for lo, hi in ((0.145, 0.175), (0.150, 0.170)):
        n = int(((sn < lo) | (sn > hi)).sum())
        print(f"  带 [{lo}, {hi}] 报出 {n:,} = {100.0 * n / len(sn):.3f}%")
    print("  ★ YenCheck.cs 现用 [0.145, 0.175]; 落在带外说明**数字**字体不对, 不是 ¥ 的问题。")

    # ---- 按分辨率分组: 苹果和安卓机型不同, 分辨率是平台的代理变量 ----
    by = collections.defaultdict(list)
    for r in use:
        by[(r["W"], r["H"])].append(r)
    print(f"\n{'分辨率':>14}{'张数':>9}{'比值中位':>10}{'p5':>8}{'p95':>8}{'数字高中位':>12}")
    shown = 0
    for sz in sorted(by, key=lambda k: -len(by[k])):
        g = by[sz]
        if len(g) < args.min_group:
            continue
        v = np.array([r["stem_ratio"] for r in g])
        dh = np.array([r["digit_h"] for r in g])
        print(f"{f'{sz[0]}x{sz[1]}':>14}{len(g):>9}{np.median(v):>10.3f}"
              f"{_pct(v,5):>8.3f}{_pct(v,95):>8.3f}{np.median(dh):>12.0f}")
        shown += 1
    if not shown:
        print(f"{'(没有分辨率够 ' + str(args.min_group) + ' 张)':>14}")
    print("★ 这些中位数明显分成两簇 = 苹果和安卓两种字形, 和玄武说的对得上。")
    print("  分不成两簇的话, '平台不同字形不同'这个前提在这批数据上不成立, 要回头重看。")

    # ---- 有阈值才报 ----
    flagged = []
    if args.low is not None or args.high is not None:
        for r in use:
            v = r["stem_ratio"]
            if (args.low is not None and v < args.low) or \
               (args.high is not None and v > args.high):
                flagged.append(r)
        print(f"\n**报出 {len(flagged):,} 张 / 实际判定 {len(use):,} 张 "
              f"= {100.0 * len(flagged) / max(len(use), 1):.4f}%**")
        print(f"  判据: 比值不在 [{args.low}, {args.high}] 之内")
        for r in sorted(flagged, key=lambda r: -r["stem_ratio"])[:30]:
            print(f"    比值 {r['stem_ratio']:.3f}  竖笔 {r['y_stem']:.0f}/笔画 {r['d_stroke']:.0f}"
                  f"  数字高 {r['digit_h']}  {r['W']}x{r['H']}  {r['file'][:50]}")
        print("★ 报出来的**一定要人工看一眼**再下结论。安卓真图和苹果假图的数是一样的。")
    else:
        print("\n(没给 --low/--high, 只量不报。先看上面的分布再定阈值。)")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        cols = ["file", "W", "H", "month", "digit_h", "n_digit", "d_stroke", "y_stem", "y_bar",
                "stem_ratio", "bar_ratio", "yen_h_ratio", "bot_gap", "stroke_norm"]
        fset = {id(r) for r in flagged}
        with args.out.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols + ["flagged"])
            for r in rows:
                w.writerow([r.get(c, "") for c in cols] + [1 if id(r) in fset else 0])
        print(f"\n明细 -> {args.out}")


if __name__ == "__main__":
    main()
