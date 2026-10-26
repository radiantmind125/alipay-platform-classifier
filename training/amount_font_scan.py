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

只读图片。只管蓝底页; 白底账单详情页的金额归 MinusCheck / DotCheck / FontCheck。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFile

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


def measure(path):
    """量一张图。返回 (数据字典, None) 或 (None, 跳过原因)。"""
    try:
        with Image.open(path) as im:
            rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
    except Exception:  # noqa: BLE001
        return None, "读不了"
    H, W = rgb.shape[:2]
    if not is_blue_page(rgb):
        return None, "不是蓝底页"
    loc = locate_amount_blue(rgb)
    if not loc:
        return None, "定位不到金额行"
    glyphs, gcrop, _fgm = _reextract_mask(rgb, (loc[0], loc[1], loc[2], loc[3]))
    if len(glyphs) < 3:
        return None, "块数不够"
    cw = gcrop.shape[1]
    rest = glyphs[1:]                          # 最左边那块当 ¥, 不进统计
    tall = max(k["h"] for k in rest)
    digits = [k for k in rest
              if k["h"] >= 0.80 * tall and k["x"] > 0 and k["x"] + k["w"] < cw]
    if len(digits) < 3:                       # 金额至少 x.xx 三位
        return None, "数字不够三个"
    # 金额行上下被切到了(图裁得正好卡在金额行上 / 定位框没框全)就不判: 切掉一截的 0 更胖, 切剩一条的小数点更扁
    ch_ = gcrop.shape[0]
    if any(k["y"] == 0 or k["y"] + k["h"] >= ch_ for k in digits + [glyphs[0]]):
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

    zeros = []
    for k in digits:
        if k["h"] < DIGIT_TALL * dh:
            continue
        hs = holes(k["mask"])
        if is_zero(k, hs):
            zeros.append((k["w"] / dh, hs[0][2] / k["w"]))

    # 小数点: 数字行里最后两个数字前面紧挨着的那个小块, 底边和数字底齐
    dot = None
    seq = sorted([k for k in rest if k["x"] > 0 and k["x"] + k["w"] < cw], key=lambda k: k["x"])
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

    return dict(
        W=W, H=H, digit_h=int(dh), n_digit=len(digits), n_zero=len(zeros),
        zero_w=round(max(z[0] for z in zeros), 4) if zeros else None,
        zero_hole_w=round(max(z[1] for z in zeros), 4) if zeros else None,
        dot_h=round(dot["h"] / dh, 4) if dot else None,
        dot_w=round(dot["w"] / dh, 4) if dot else None,
    ), None


def judge(m):
    """AmountFontCheck.Check 的判定翻版。返回 (结论, 理由): 0 判不了 / 1 正常 / 2 可疑。"""
    if m is None:
        return 0, "没量到"
    if m["digit_h"] < MIN_DIGIT_H:
        return 0, f"数字高 {m['digit_h']} < {MIN_DIGIT_H}"
    bad = []
    if m["zero_w"] is not None and m["zero_w"] > ZERO_WIDTH_MAX:
        bad.append(f"0 宽/数字高 {m['zero_w']:.4f}")
    if m["dot_h"] is not None and m["digit_h"] >= MIN_DIGIT_H_FOR_DOT and m["dot_h"] < DOT_HEIGHT_MIN:
        bad.append(f"小数点高/数字高 {m['dot_h']:.4f}")
    if bad:
        return 2, "; ".join(bad)
    if m["zero_w"] is None and (m["dot_h"] is None or m["digit_h"] < MIN_DIGIT_H_FOR_DOT):
        return 0, "金额里没有 0 也没有能量的小数点"
    return 1, ""


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    for p in sys.argv[1:]:
        m, why = measure(p)
        v, reason = judge(m)
        print(v, Path(p).name, m if m else why, reason)
