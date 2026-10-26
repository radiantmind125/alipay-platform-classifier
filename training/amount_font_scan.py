r"""蓝图金额字体检查(demo/AmountFontCheck.cs 的 Python 对数版): 金额数字是不是支付宝自己那套数字字体。

背景
----
2026-10-26 客户(A Fei)报了一张 P 图金额的假图, 玄武说"这个是金额字体"。经理给了原图:
假图 1220x2712 安卓 到账成功 ¥100.00, 真图 1320x2868 苹果 到账成功 ¥100.00(经理自己当场截的)。

支付宝的金额数字是 App 自带的字体, **苹果安卓一模一样**(同样 102 像素高的 0 都是 61 像素宽,
小数点都是 17x19)。假图的数字是另一套字体: 0 胖了一圈, 小数点扁了。

判据(都是同一张图里自己比, 不看绝对像素, 也不用知道机型)
--------------------------------------------------------
1. "0" 的宽 / 数字高
   "0" = 正好一个孔、孔在正中、孔高 >= 字高 0.6 的数字块(支付宝的 4 也是一个封闭孔, 但孔只有字高一半, 这样就排开了)。
   本机真图 7,213 个 0(1,841 页): 0.580 ~ 0.620。假图 0.690。
2. 小数点的高 / 数字高(数字高 >= 60 才判, 小图一个像素差太多)
   本机真图 2,218 页: 0.173 ~ 0.224。假图 0.147(扁)。

护栏(都是测出误报才加的): 最左块要像 ¥; 数字碰到裁块上下边不判; 数字高 < 60 不判;
数字正上方 / 正下方 3 行里"不是蓝底也不是白字"的颜色 > 5% 不判(框线、下划线、通知横幅、描边);
最胖那个 0 的上下沿是平的(> 0.6)不判。

只读图片。只管蓝底页; 白底账单详情页的金额归 MinusCheck / DotCheck / FontCheck。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFile, ImageOps

ImageFile.LOAD_TRUNCATED_IMAGES = True

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from yen_scan import (BAR_FILL, YEN_H_HIGH, YEN_H_LOW, _reextract_mask, _runs,  # noqa: E402
                      is_blue_page, locate_amount_blue)

# ---- 判定口径, 和 demo/AmountFontCheck.cs 的常数一一对应(改一边必须改另一边) ----
ZERO_WIDTH_MAX = 0.65          # "0" 宽 / 数字高 超过就报
DOT_HEIGHT_MIN = 0.16          # 小数点高 / 数字高 低于就报
MIN_DIGIT_H = 60               # 数字高低于这个整张不判(小图二值化会把 0 的上下圆弧削掉, 0 显得胖)
MIN_DIGIT_H_FOR_DOT = 80       # 小数点那条要的最小数字高(数字高 60~70 时真图和假图只隔一个像素)
DIGIT_TALL = 0.85              # 块高 >= 数字高的这个比例才算数字
HOLE_MIN_AREA = 3              # 孔至少这么多像素(挡 JPEG 噪点)
ZERO_EDGE_MAX = 0.60           # 最胖那个 0 的上下沿平整度超过这个就不判(被框线/横幅压住削平了)
BAND = 3                       # 数字行正上方 / 正下方各看几行
BAND_FOREIGN_MAX = 0.05        # 这几行里"不是蓝底也不是白字"(离 底色->白 线段 > 80)的像素占比超过这个就不判


def holes(mask: np.ndarray):
    """一个字块里的孔: 外接框外补一圈 0, 背景按 4 连通标记, 不碰外圈的就是孔。
    返回 [(中心x, 中心y, 宽, 高, 面积)], 坐标相对字块外接框。"""
    inv = (1 - np.pad(mask.astype(np.uint8), 1)).astype(np.uint8)
    n, _lab, st, cen = cv2.connectedComponentsWithStats(inv, connectivity=4)
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if x == 0 or y == 0 or x + w >= inv.shape[1] or y + h >= inv.shape[0]:
            continue
        if a < HOLE_MIN_AREA:
            continue
        out.append((float(cen[i][0]) - 1.0, float(cen[i][1]) - 1.0, int(w), int(h), int(a)))
    return out


def is_zero(k, hs) -> bool:
    if len(hs) != 1:
        return False
    cx, cy, w, h, _a = hs[0]
    return (0.40 <= cx / k["w"] <= 0.60 and 0.40 <= cy / k["h"] <= 0.60 and h >= 0.60 * k["h"])


def edge_fill(mask: np.ndarray) -> float:
    """字块最上 2 行和最下 2 行里, 字块自己的像素占字宽的最大比例(0 的上下是圆弧, 正常不到 0.5)。"""
    h, w = mask.shape
    rows = [r for r in (0, 1, h - 2, h - 1) if 0 <= r < h]
    return max(float(mask[r].sum()) / w for r in rows)


def _band_foreign(rgb, loc, fgm, ch, cw, c0, c1, d_top, d_bot):
    """数字行正上方 BAND 行和正下方 BAND 行(横向 [c0, c1))里, 颜色离"底色 -> 白"线段超过 80 的像素占比。
    底色 = Otsu 分到底色那边的像素每个通道的下中位数。和 C# 一样按 R、G、B 顺序算, 拿距离平方和 6400 比。"""
    x0, y0, x1, y1 = loc[:4]
    pad = max(2, int((y1 - y0) * 0.15))
    padx = max(pad, int((x1 - x0) * 0.30))
    H, W = rgb.shape[:2]
    sub = rgb[max(0, y0 - pad):min(H, y1 + pad), max(0, x0 - padx):min(W, x1 + pad)].astype(np.float64)
    bgp = sub[~fgm]
    if not len(bgp):
        return 0.0
    a = np.array([np.sort(bgp[:, c])[(len(bgp) + 1) // 2 - 1] for c in range(3)], np.float64)
    d = 255.0 - a
    L = d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
    rows = list(range(max(0, d_top - BAND), d_top)) + list(range(d_bot, min(ch, d_bot + BAND)))
    if not rows or c1 <= c0:
        return 0.0
    pxs = sub[rows, c0:c1].reshape(-1, 3) - a
    t = (pxs[:, 0] * d[0] + pxs[:, 1] * d[1] + pxs[:, 2] * d[2]) / L if L else np.zeros(len(pxs))
    t = np.clip(t, 0.0, 1.0)
    e0, e1, e2 = pxs[:, 0] - t * d[0], pxs[:, 1] - t * d[1], pxs[:, 2] - t * d[2]
    far = (e0 * e0 + e1 * e1 + e2 * e2) > 6400
    return float(far.sum()) / len(far)


def measure(path):
    """量一张图。返回 (数据字典, None) 或 (None, 跳过原因)。比值不舍入(和 C# 一样拿原值去比阈值)。"""
    try:
        with Image.open(path) as im:
            # ★ 按 EXIF 方向转正, 和 OpenCvSharp ImDecode(Color) 的行为一致
            rgb = np.asarray(ImageOps.exif_transpose(im).convert("RGB"), dtype=np.uint8)
    except Exception:  # noqa: BLE001
        return None, "读不了"
    H, W = rgb.shape[:2]
    if not is_blue_page(rgb):
        return None, "不是蓝底页"
    loc = locate_amount_blue(rgb)
    if not loc:
        return None, "定位不到金额行"
    glyphs, gcrop, fgm = _reextract_mask(rgb, (loc[0], loc[1], loc[2], loc[3]))
    # ★ 和 C# 一样按 (x, y) 排, x、y 都一样时保持 OpenCV 标号的先后(稳定排序)
    glyphs = sorted(glyphs, key=lambda k: (k["x"], k["y"]))
    if len(glyphs) < 3:
        return None, "块数不够"
    cw, ch = gcrop.shape[1], gcrop.shape[0]
    rest = glyphs[1:]                          # 最左边那块当 ¥, 不进统计
    tall = max(k["h"] for k in rest)
    digits = [k for k in rest
              if k["h"] >= 0.80 * tall and k["x"] > 0 and k["x"] + k["w"] < cw]
    if len(digits) < 3:                       # 金额至少 x.xx 三位
        return None, "数字不够三个"
    # 金额行上下被切到了(图裁得正好卡在金额行上 / 定位框没框全)就不判: 切掉一截的 0 更胖, 切剩一条的小数点更扁
    if any(k["y"] == 0 or k["y"] + k["h"] >= ch for k in digits + [glyphs[0]]):
        return None, "金额行上下被切到了"
    d_top = min(k["y"] for k in digits)
    d_bot = max(k["y"] + k["h"] for k in digits)
    dh = d_bot - d_top
    if dh <= 0:
        return None, "数字高算不出来"
    hs_ = np.array([k["h"] for k in digits], float)
    if hs_.std() / hs_.mean() > 0.08:
        return None, "数字高度不齐"

    # ---- ¥ 护栏(和 yen_scan / YenCheck 同一口径): 最左块高比 0.60~0.80, 而且正好两道横杠 ----
    # ★ 不能省: 图只截了上面一截时定位器会锁到状态栏的时间, 系统字体的 0 本来就胖(苹果真图截上 1/6 量出 0.652)
    yen = glyphs[0]
    if not (YEN_H_LOW <= yen["h"] / dh <= YEN_H_HIGH):
        return None, "最左块高度比例不像 ¥"
    bar_rows = yen["mask"].sum(axis=1) >= BAR_FILL * yen["w"]
    if _runs(bar_rows).size != 2:
        return None, "最左块横杠不是两道, 不像 ¥"

    # ---- 判据一的量: 认出 0, 取最胖的那个(连它上下沿的平整度一起记) ----
    n_zero, zero_w, zero_hole_w, zero_edge = 0, 0.0, 0.0, 0.0
    for k in digits:
        if k["h"] < DIGIT_TALL * dh:
            continue
        hs = holes(k["mask"])
        if not is_zero(k, hs):
            continue
        n_zero += 1
        wr = k["w"] / dh
        if wr > zero_w:
            zero_w = wr
            zero_edge = edge_fill(k["mask"])
        hw = hs[0][2] / k["w"]
        if hw > zero_hole_w:
            zero_hole_w = hw

    # ---- 判据二的量: 最后两位数字前面紧挨着的那个小块, 底边和数字底齐 ----
    dot = None
    seq = [k for k in rest if k["x"] > 0 and k["x"] + k["w"] < cw]
    big = [k for k in seq if k["h"] >= DIGIT_TALL * dh]
    if len(big) >= 2:
        x_last2 = big[-2]["x"]
        cands = [k for k in seq if k["x"] + k["w"] <= x_last2 and k["h"] < 0.35 * dh and k["w"] < 0.35 * dh
                 and abs(d_bot - (k["y"] + k["h"])) <= 0.08 * dh]
        if cands:
            k = cands[-1]
            # 它和最后两个数字之间不能再夹着别的数字
            if not [b for b in big if k["x"] < b["x"] < x_last2]:
                dot = k

    # ---- 数字行正上方 / 正下方几行里, 有多少"不是蓝底也不是白字"的颜色 ----
    band_foreign = _band_foreign(rgb, loc, fgm, ch, cw,
                                 min(k["x"] for k in digits), max(k["x"] + k["w"] for k in digits), d_top, d_bot)

    return dict(
        W=W, H=H, digit_h=int(dh), n_digit=len(digits), n_zero=n_zero,
        zero_w=zero_w if n_zero else None,
        zero_hole_w=zero_hole_w if n_zero else None,
        zero_edge=zero_edge if n_zero else None,
        dot_h=dot["h"] / dh if dot else None,
        dot_w=dot["w"] / dh if dot else None,
        band_foreign=band_foreign,
    ), None


def judge(m):
    """AmountFontCheck.Check 的判定翻版。返回 (结论, 理由): 0 判不了 / 1 正常 / 2 可疑。"""
    if m is None:
        return 0, "没量到"
    if m["digit_h"] < MIN_DIGIT_H:
        return 0, f"数字高 {m['digit_h']} < {MIN_DIGIT_H}"
    if m["band_foreign"] > BAND_FOREIGN_MAX:
        return 0, f"数字上下贴着别的颜色的东西({m['band_foreign']:.2f}), 像框线、横幅或描边"
    if m["n_zero"] and m["zero_edge"] > ZERO_EDGE_MAX:
        return 0, f"0 的上沿或下沿是平的({m['zero_edge']:.2f}), 像被框线或横幅压住了"
    bad = []
    if m["n_zero"] and m["zero_w"] > ZERO_WIDTH_MAX:
        bad.append(f"0 宽/数字高 {m['zero_w']:.4f}")
    dot_usable = m["dot_h"] is not None and m["digit_h"] >= MIN_DIGIT_H_FOR_DOT
    if dot_usable and m["dot_h"] < DOT_HEIGHT_MIN:
        bad.append(f"小数点高/数字高 {m['dot_h']:.4f}")
    if bad:
        return 2, "; ".join(bad)
    if not m["n_zero"] and not dot_usable:
        return 0, "金额里没有 0 也没有能量的小数点"
    return 1, ""


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    for p in sys.argv[1:]:
        m, why = measure(p)
        v, reason = judge(m)
        print(v, Path(p).name, m if m else why, reason)
