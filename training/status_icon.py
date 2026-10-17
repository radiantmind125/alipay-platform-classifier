r"""蓝图页头的图标判状态(纯看图, 不做 OCR): 到账失败 / 提交成功(还没到账) 的页头和成功页长得不一样。

为什么(2026-10-15)
------------------
经理 10/14 要把 到账失败 这类异常页挑出来。status_scan.py 用 OCR 读页头能挑, 但服务器上一张要 3~6 秒,
89 万张蓝图扫一遍要 30 多小时; 交给经理的检查(TimelineCheck 那些)也都是只看图的 C#, 拿不到 OCR 的字。
看了样本(1.8 万张随机图里的 8 张 到账失败、12 张 提交成功, 和成功页比):
    成功页(转账成功 到账成功 支付成功 代付成功 授权成功): 标题左边一个空心细圈里打勾, 标题下面是大号白字金额 ¥100.00
    到账失败: 标题正上方一个实心白圆, 圆里是蓝色感叹号 !, 标题下面一行灰字 资金将原路返回你的付款账户, 页头里没有大金额
    提交成功: 同样的实心白圆, 圆里是蓝色时钟指针, 下面一行 预计2小时内到账 查看到账进度
量出来(样本 8 + 12 张): 实心白圆横向正中(圆心 0.50 W), 直径 0.05~0.066 W, 圆顶在 0.058~0.076 H,
    白色占外接框 0.69~0.72, 填掉圆里的字以后占 0.74~0.76(正圆是 0.785);
    圆里的蓝字: 感叹号 高/宽 4.5~6.3、两块(一竖 + 一点); 时钟 高/宽 1.7 左右、一块、偏右
成功页标题左边的细圈: 白色只占外接框 0.11~0.12(空心), 圆心在 0.40 W 左右 —— 和实心圆分得很开。

判法
----
    页头不是支付宝蓝(5%~20% 高度平均色 B > 150 且 B > R + 60)   -> CannotDetermine(不是蓝图)
    在 2%~17% 高度里找白色连通块(三个通道都 > 215), 挑 横向正中 / 直径 0.035~0.09 W / 近正方 / 实心 的
        找到, 圆里蓝字 字宽 <= 0.16 直径(感叹号)        -> Suspicious, 到账失败一类
        找到, 圆里蓝字 字宽 0.19~0.34 直径、一块(时钟)   -> Suspicious(可选), 提交成功 还没到账
        找到, 圆里认不出                          -> CannotDetermine, 记下来
    没找到实心圆, 找到成功页标题左边的空心细圈     -> Ok
    蓝图上都没有, 但卡片里是实心绿圆 + 白色时钟     -> Suspicious(可选), 等待对方领取(10/17 加, 见 GREEN_* 的注)
    都没有(通知条盖住了页头等)                     -> CannotDetermine
  白图(页头 5%~20% 平均色三个通道都 > 200), 10/17 加, 见 AMOUNT_* 的注:
    金额下面橙字(处理中 / 等待对方领取)            -> Suspicious(可选), 白图未完成
    金额下面红字(已全额退款)                       -> Suspicious, 白图已全额退款
    金额上面是"退"字头像(退款成功的记录)           -> Suspicious, 白图退款记录
    以上三条只在支付宝账单详情卡片上判, 别的页面(平台收银台等)记 白图异常但不是支付宝卡片, CannotDetermine
    都没有 -> CannotDetermine, 白图没看出异常(白图上不说 Ok: 交易关闭 这类灰字分不出来)
★ 授权成功 / 代付成功 的页头和成功页一模一样, 只能靠字分(status_scan)。
★ 通知条盖住页头的(1.8 万张里 8 张 到账失败 有 1 张), 这里判不了, 交给 status_scan(读 资金将原路返回)。

用法
----
    python training/status_icon.py --src 目录或名单 [...] --out 输出目录 --workers 20
    -> icon.csv(每张一行: verdict kind 圆的位置和大小 圆里字的形状)、summary.txt
    全库: 加 --flagged-only, icon.csv 只写 Suspicious 和蓝图上判不了的
本地: 样本 到账失败 8 张 7 张判出(1 张通知条盖住页头, 判不了), 提交成功 12/12; 成功页、代付、授权 40 张 0 误报;
    本地全部 1,925 张蓝图 0 误报(判出的 2 张都是真的 提交成功 拼音页); 8 进程 209 张/秒(OCR 的 35 倍)
"""
from __future__ import annotations

import os
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

ROI = (0.02, 0.17)              # 找图标的高度范围(占页面高度)
WHITE = 215                     # 三个通道都大于这个算白
# 实心白圆(到账失败 / 提交成功)
DISK_CX = 0.04                  # 圆心离页面正中不超过 0.04 W
DISK_D = (0.035, 0.09)          # 直径占页宽
DISK_TOP = (0.035, 0.13)        # 圆顶占页高
DISK_ASPECT = (0.85, 1.18)
DISK_WHITE = (0.55, 0.80)       # 白色占外接框
DISK_FILLED = (0.68, 0.84)      # 填掉圆里的字以后占外接框(正圆 0.785)
# 圆里的蓝字(10/16 改): 只认红通道 < GLYPH_R 的像素(蓝字 R 只有 20~60, JPEG 在白圆里压出来的灰点 R 在 200 以上),
#   而且先把圆往里收一圈(去掉圆边上的过渡色); 小于全部 5% 的碎块不算。量 字宽 / 圆直径、字高 / 圆直径:
#   样本 7 张到账失败 + 12 张提交成功各过 7 种压缩缩放(JPEG 75 / 40、缩到 0.5、0.75 + JPEG 60、0.5 + JPEG 40、宽 360 + JPEG 60):
#       感叹号 字宽 0.048~0.125   时钟 0.217~0.275   中间是空的;  字高 两种都在 0.44~0.49(压得最狠时掉到 0.13~0.28)
#   老办法(圆里不是白的都算字, 看 高/宽 和块数)在 JPEG 40 下会把感叹号认成时钟, 不用了。
GLYPH_R = 160
BANG_MAX_W = 0.16               # 感叹号: 字宽 <= 0.16 直径
CLOCK_W = (0.19, 0.34)          # 时钟指针: 字宽 0.19~0.34 直径, 而且只有一块
GLYPH_MIN_H = 0.30              # 字高 < 0.30 直径: 压坏了, 认不出
# 成功页标题左边的空心细圈
RING_CX = (0.25, 0.48)
RING_D = (0.025, 0.065)
RING_WHITE = (0.05, 0.30)
RING_FILLED = (0.68, 0.86)

# ---- 10/17 加: 等待对方领取(蓝图 转账详情页: 蓝色顶栏 + 白卡片, 卡片里一个实心绿圆, 圆里白色时钟) ----
#   样本 17 张(1.8 万张里 3 张 + 没找到图标的 2,000 张里 14 张): 绿圆圆心 0.500 W, 直径 0.157~0.172 W, 圆顶 0.161~0.172 H,
#   填满占外接框 0.78~0.79; 圆里白色时钟 字宽 0.220~0.229 字高 0.424~0.430 直径、一块(过 3 种压缩缩放 0.203~0.237 / 0.393~0.441);
#   顶栏(2%~8% 高度)是支付宝蓝。微信支付成功页也是大绿圆(圆里打勾), 但顶栏不是蓝的, 字也不是时钟。
GREEN_ROI = (0.05, 0.40)
GREEN_D = (0.12, 0.22)
GREEN_TOP = (0.10, 0.25)
GREEN_FILLED = (0.70, 0.86)
GREEN_GLYPH_W = (0.17, 0.30)
GREEN_GLYPH_H = (0.33, 0.50)
# ---- 10/17 加: 白图账单详情(页头白) ----
#   金额(卡片里最大的一行黑字, 12%~40% 高度、横向居中; 1.8 万张里的金额在 16%~33%)下面那行状态字:
#     处理中 / 等待对方领取 是橙字, 已全额退款 是红字; 交易成功 / 退款成功 是灰字。
#   量 金额下面 0.15~1.4 个金额字高、金额中心左右 0.3 W 里的 橙 / 红 像素数 / 金额字高^2(1.8 万张里的样本):
#     处理中 橙 0.32~0.35   等待对方领取 橙 0.35~0.70   已全额退款 红 0.54~0.55(橙 0)   退款成功 / 交易成功 0
#   本地 9,621 张(拼音账单详情 7,500 张为主)判出来的逐张看过, 都对; 样本 + 400 张正常白图过 10 种压缩缩放:
#     正常白图 0 次判可疑; 未完成 / 已全额退款 / 退款记录 压得狠了掉成判不了, 不会认错种类; 等待对方领取 绿圆 每种都 19/19
#   退款成功 的记录看头像: 金额上面居中一个蓝色实心圆, 圆里白色"退"字(TUI_REF), 和默认头像(蓝圆里白色人形)比形状:
#     退款记录 15/17 张 IoU 0.83~1.00(1 张头像被屏幕顶切掉 0.55、1 张没有退字头像), 人形头像 0.39~0.48
#   ★ 这两条只在支付宝账单详情的卡片上判(page_type.card_gate 是 A / A0): 平台收银台这类页面金额下面也常有红字。
#     card_gate 慢, 只在要判可疑的时候才量。
AMOUNT_ROI = (0.12, 0.40)       # 从 12% 起: 拼音页顶栏(返回箭头 + 拼音标题, 0.08~0.11)比金额字还高, 会被当成金额
AMOUNT_MAX_W = 0.60             # 金额那一行不会超过 0.6 页宽(顶栏那种整行的不是)
AMOUNT_DARK = 100               # 金额黑字: 三个通道最大值 < 100
AMOUNT_H = (0.012, 0.05)        # 一个字的高 占页高
WARM_MIN = 0.10                 # 橙 + 红 像素 / 金额字高^2 >= 这个: 金额下面是彩色状态字
TUI_IOU = 0.70
TUI_N = 32
TUI_REF = [
    "................................", "................................", "................................", "................................",
    "................................", "................................", "................................", "................................",
    "........##...##########.........", "........###..##########.........", ".........##..###.....##.........", ".........##..##########.........",
    ".............##########.........", ".......#####.###.....##.........", ".......#####.##########.........", "........####.###########........",
    ".........###.###..#...##........", ".........###.###.#######........", ".........###.###.######.........", ".........###.######.####........",
    "........####.#####...####.......", ".......########........##.......", ".......###.##############.......", ".......##...#############.......",
    "................................", "................................", "................................", "................................",
    "................................", "................................", "................................", "................................",
]
_TUI = np.array([[c == "#" for c in row] for row in TUI_REF])
# 判可疑的种类; flagged-only 时 icon.csv 只写这些和下面几种判不了的
SUSPICIOUS_KINDS = ("到账失败", "提交成功", "等待对方领取", "白图未完成", "白图已全额退款", "白图退款记录")
KEEP_KINDS = ("实心圆里认不出", "没找到图标", "白图异常但不是支付宝卡片")

KEYS = ["path", "W", "H", "verdict", "kind", "disk", "glyph", "secs", "src"]


def blue_header(bgr) -> bool:
    h = bgr.shape[0]
    b, g, r = bgr[int(h * 0.05):int(h * 0.20)].reshape(-1, 3).mean(0)
    return b > 150 and b > r + 60


def _shape(lab, k, x, y, w, h, area):
    """一个白色连通块: (白色占外接框, 填掉里面的洞以后占外接框, 洞的掩膜)。"""
    comp = (lab[y:y + h, x:x + w] == k).astype(np.uint8)
    filled = comp.copy()
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(filled, cnts, -1, 1, -1)
    return area / (w * h), float(filled.sum()) / (w * h), (filled == 1) & (comp == 0)


def _glyph(crop, filled, w, h):
    """圆里的蓝字: (字宽 / 直径, 字高 / 直径, 块数, 中心横向偏移 / 直径)。见 GLYPH_R 的注。"""
    inner = cv2.erode(filled, np.ones((3, 3), np.uint8), iterations=max(1, w // 20))
    ink = ((crop[:, :, 2] < GLYPH_R) & (inner == 1)).astype(np.uint8)
    total = int(ink.sum())
    if total < 4:
        return 0.0, 0.0, 0, 0.0
    n, lab, st, _c = cv2.connectedComponentsWithStats(ink, 8)
    keep = [j for j in range(1, n) if st[j][4] >= 0.05 * total]
    ys, xs = np.nonzero(np.isin(lab, keep))
    gw, gh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    return gw / w, gh / h, len(keep), (xs.mean() - w / 2) / w


def _filled(lab, k, x, y, w, h):
    comp = (lab[y:y + h, x:x + w] == k).astype(np.uint8)
    filled = comp.copy()
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(filled, cnts, -1, 1, -1)
    return filled


def green_waiting(bgr):
    """蓝图 转账详情页的 等待对方领取(实心绿圆里白色时钟)。返回 (圆的说明, 字的说明) 或 None。"""
    H, W = bgr.shape[:2]
    b, g, r = bgr[int(H * 0.02):int(H * 0.08)].reshape(-1, 3).mean(0)
    if not (b > 150 and b > r + 60):
        return None
    y0 = int(H * GREEN_ROI[0])
    roi = bgr[y0:int(H * GREEN_ROI[1])]
    ri = roi.astype(np.int16)
    green = ((ri[:, :, 1] > 140) & (ri[:, :, 1] - ri[:, :, 2] > 60) & (ri[:, :, 1] - ri[:, :, 0] > 30)).astype(np.uint8)
    n, lab, st, _c = cv2.connectedComponentsWithStats(green, 8)
    for k in range(1, n):
        x, y, w, h, area = (int(v) for v in st[k])
        d, cx, top = w / W, (x + w / 2) / W, (y + y0) / H
        if (abs(cx - 0.5) > DISK_CX or not (GREEN_D[0] <= d <= GREEN_D[1]) or not (GREEN_TOP[0] <= top <= GREEN_TOP[1])
                or not (DISK_ASPECT[0] <= w / max(h, 1) <= DISK_ASPECT[1])):
            continue
        filled = _filled(lab, k, x, y, w, h)
        fa = float(filled.sum()) / (w * h)
        if not (GREEN_FILLED[0] <= fa <= GREEN_FILLED[1]):
            continue
        inner = cv2.erode(filled, np.ones((3, 3), np.uint8), iterations=max(1, w // 20))
        ink = ((roi[y:y + h, x:x + w].min(2) > 200) & (inner == 1)).astype(np.uint8)       # 圆里的白字
        total = int(ink.sum())
        if total < 4:
            continue
        m, gl, gs, _c2 = cv2.connectedComponentsWithStats(ink, 8)
        keep = [j for j in range(1, m) if gs[j][4] >= 0.05 * total]
        ys, xs = np.nonzero(np.isin(gl, keep))
        gw, gh = (xs.max() - xs.min() + 1) / w, (ys.max() - ys.min() + 1) / h
        if GREEN_GLYPH_W[0] <= gw <= GREEN_GLYPH_W[1] and GREEN_GLYPH_H[0] <= gh <= GREEN_GLYPH_H[1] and len(keep) == 1:
            return (f"green cx {cx:.3f} top {top:.3f} d {d:.3f} filled {fa:.2f}",
                    f"w {gw:.3f} h {gh:.3f} parts {len(keep)} off {(xs.mean() - w / 2) / w:+.3f}")
    return None


def amount_row(bgr):
    """白图卡片里的金额(12%~40% 高度里字最高、横向居中、宽 0.12~0.6 页宽、至少 3 个字的黑字行)。返回 (上, 下, 左, 右) 或 None。"""
    H, W = bgr.shape[:2]
    y0 = int(H * AMOUNT_ROI[0])
    roi = bgr[y0:int(H * AMOUNT_ROI[1])]
    dark = (roi.max(2) < AMOUNT_DARK).astype(np.uint8)
    n, _lab, st, _c = cv2.connectedComponentsWithStats(dark, 8)
    comps = [tuple(int(v) for v in st[k]) for k in range(1, n)
             if AMOUNT_H[0] * H <= st[k][3] <= AMOUNT_H[1] * H and st[k][2] <= 0.12 * W]
    rows = []                                   # [左, 上, 右, 下, 字数, 最高的字]
    for x, y, w, h, _a in sorted(comps, key=lambda c: c[1]):
        for r in rows:
            if min(r[3], y + h) - max(r[1], y) > 0.5 * min(h, r[3] - r[1]):
                r[0], r[1], r[2], r[3] = min(r[0], x), min(r[1], y), max(r[2], x + w), max(r[3], y + h)
                r[4] += 1
                r[5] = max(r[5], h)
                break
        else:
            rows.append([x, y, x + w, y + h, 1, h])
    cand = [r for r in rows if r[4] >= 3 and 0.35 <= (r[0] + r[2]) / 2 / W <= 0.65 and 0.12 * W <= r[2] - r[0] <= AMOUNT_MAX_W * W]
    if not cand:
        return None
    r = max(cand, key=lambda r: r[5])
    return r[1] + y0, r[3] + y0, r[0], r[2]


def status_line(bgr, row):
    """金额下面那行的 (橙, 红) 像素 / 金额字高^2。"""
    H, W = bgr.shape[:2]
    a0, a1, x0, x1 = row
    h = a1 - a0
    cx = (x0 + x1) // 2
    band = bgr[a1 + int(0.15 * h):min(H, a1 + int(1.4 * h)), max(0, cx - int(0.3 * W)):min(W, cx + int(0.3 * W))].astype(np.int16)
    b, g, r = band[:, :, 0], band[:, :, 1], band[:, :, 2]
    warm = (r > 180) & (r - b > 90) & (r - g > 40)
    # 橙 / 红 按 G - B 分, 不按 G 的高低: JPEG 把红字和白底混成粉色(255,140,140)时 G 会升到 100 以上, 但 G ≈ B;
    #   橙字(255,128,0)混白以后(255,190,128) G 仍比 B 高 60 左右。按 G 的高低分, 已全额退款 压过以后全被认成橙字(10/17 量过)
    orange = warm & (g - b >= 40)
    return float(orange.sum()) / (h * h), float((warm & ~orange).sum()) / (h * h)


def tui_iou(bgr, row):
    """金额上面居中的蓝色实心圆头像, 圆里白字和"退"字模板的 IoU; 没有这种头像返回 None。"""
    H, W = bgr.shape[:2]
    y0 = max(0, row[0] - int(0.13 * H))
    roi = bgr[y0:row[0]]
    ri = roi.astype(np.int16)
    blue = ((ri[:, :, 0] > 190) & (ri[:, :, 0] - ri[:, :, 2] > 110) & (ri[:, :, 1] > 90) & (ri[:, :, 1] < 190)).astype(np.uint8)
    n, lab, st, _c = cv2.connectedComponentsWithStats(blue, 8)
    for k in range(1, n):
        x, y, w, h, _a = (int(v) for v in st[k])
        if w < 0.04 * W or abs((x + w / 2) / W - 0.5) > 0.05 or not (DISK_ASPECT[0] <= w / max(h, 1) <= DISK_ASPECT[1]):
            continue
        filled = _filled(lab, k, x, y, w, h)
        if float(filled.sum()) / (w * h) < 0.65:
            continue
        gray = cv2.resize(cv2.cvtColor(roi[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY), (TUI_N, TUI_N), interpolation=cv2.INTER_AREA)
        disk = cv2.resize(filled * 255, (TUI_N, TUI_N), interpolation=cv2.INTER_AREA) > 127
        inner = cv2.erode(disk.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2) == 1
        m = (gray > 200) & inner
        return float((m & _TUI).sum()) / max(1, int((m | _TUI).sum()))
    return None


def _alipay_card(bgr) -> bool:
    """支付宝账单详情的卡片(page_type.card_gate 是 A / A0)。慢, 只在要判可疑时调。"""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import page_type as pt
    return pt.card_gate(bgr)[0] in ("A", "A0")


def _white(out, bgr):
    row = amount_row(bgr)
    if row is None:
        out["kind"] = "白图找不到金额"
        return out
    H = bgr.shape[0]
    orange, red = status_line(bgr, row)
    iou = None
    kind = ""
    if orange + red >= WARM_MIN:
        kind = "白图已全额退款" if red > orange else "白图未完成"
    else:
        iou = tui_iou(bgr, row)
        if iou is not None and iou >= TUI_IOU:
            kind = "白图退款记录"
    out["disk"] = f"amount y {row[0] / H:.3f}-{row[1] / H:.3f}" + (f" tui_iou {iou:.2f}" if iou is not None else "")
    out["glyph"] = f"orange {orange:.3f} red {red:.3f}"
    if not kind:
        out["kind"] = "白图没看出异常"
    elif _alipay_card(bgr):
        out.update(verdict="Suspicious", kind=kind)
    else:
        out["kind"] = "白图异常但不是支付宝卡片"
        out["glyph"] += f" ({kind})"
    return out


def check(bgr) -> dict:
    """-> {verdict: Ok / Suspicious / CannotDetermine, kind: 到账失败 / 提交成功 / 等待对方领取 / 白图未完成 / 白图已全额退款 /
    白图退款记录 / 成功 / ..., disk, glyph}"""
    H, W = bgr.shape[:2]
    out = {"verdict": "CannotDetermine", "kind": "", "disk": "", "glyph": ""}
    if W < 300 or H < 400:
        out["kind"] = "图太小"
        return out
    if not blue_header(bgr):
        b, g, r = bgr[int(H * 0.05):int(H * 0.20)].reshape(-1, 3).mean(0)
        if min(b, g, r) > 200:
            return _white(out, bgr)
        out["kind"] = "不是蓝图"
        return out
    y0 = int(H * ROI[0])
    roi = bgr[y0:int(H * ROI[1])]
    white = (roi.min(2) > WHITE).astype(np.uint8)
    n, lab, st, _c = cv2.connectedComponentsWithStats(white, 8)
    ring = None
    for k in range(1, n):
        x, y, w, h, area = (int(v) for v in st[k])
        d, cx, top = w / W, (x + w / 2) / W, (y + y0) / H
        if not (DISK_ASPECT[0] <= w / max(h, 1) <= DISK_ASPECT[1]) or d < min(DISK_D[0], RING_D[0]) or d > DISK_D[1]:
            continue
        fw, fa, hole = _shape(lab, k, x, y, w, h, area)
        if (abs(cx - 0.5) <= DISK_CX and DISK_D[0] <= d <= DISK_D[1] and DISK_TOP[0] <= top <= DISK_TOP[1]
                and DISK_WHITE[0] <= fw <= DISK_WHITE[1] and DISK_FILLED[0] <= fa <= DISK_FILLED[1]):
            filled = (hole | (lab[y:y + h, x:x + w] == k)).astype(np.uint8)
            gw, gh, parts, off = _glyph(roi[y:y + h, x:x + w], filled, w, h)
            out["disk"] = f"cx {cx:.3f} top {top:.3f} d {d:.3f} white {fw:.2f} filled {fa:.2f}"
            out["glyph"] = f"w {gw:.3f} h {gh:.3f} parts {parts} off {off:+.3f}"
            if gh >= GLYPH_MIN_H and gw <= BANG_MAX_W:
                out.update(verdict="Suspicious", kind="到账失败")
            elif gh >= GLYPH_MIN_H and CLOCK_W[0] <= gw <= CLOCK_W[1] and parts == 1:
                out.update(verdict="Suspicious", kind="提交成功")
            else:
                out.update(kind="实心圆里认不出")
            return out
        if (RING_CX[0] <= cx <= RING_CX[1] and RING_D[0] <= d <= RING_D[1]
                and RING_WHITE[0] <= fw <= RING_WHITE[1] and RING_FILLED[0] <= fa <= RING_FILLED[1]):
            ring = f"cx {cx:.3f} top {top:.3f} d {d:.3f} white {fw:.2f} filled {fa:.2f}"
    if ring:
        out.update(verdict="Ok", kind="成功", disk=ring)
        return out
    gw_ = green_waiting(bgr)
    if gw_:
        out.update(verdict="Suspicious", kind="等待对方领取", disk=gw_[0], glyph=gw_[1])
    else:
        out["kind"] = "没找到图标"
    return out


def work(task):
    path, src = task
    t0 = time.time()
    r = {"path": path, "src": src}
    try:
        buf = np.fromfile(path, np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None
        if bgr is None:
            r.update(verdict="CannotDetermine", kind="读不出")
        else:
            r.update(W=bgr.shape[1], H=bgr.shape[0], **check(bgr))
    except Exception as e:  # noqa: BLE001
        r.update(verdict="CannotDetermine", kind=f"出错 {type(e).__name__}")
    r["secs"] = round(time.time() - t0, 3)
    return r


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import page_type as pt
    ap = argparse.ArgumentParser(description="蓝图页头图标判状态(只看图)")
    ap.add_argument("--src", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--flagged-only", action="store_true",
                    help="icon.csv 只写 Suspicious 和蓝图上判不了的(全库几十万张时用, 汇总照样数全部)")
    a = ap.parse_args()
    todo = pt.collect(a.src)
    todo, dups = pt.dedupe_gwcz(todo)
    if a.n and a.n < len(todo):
        random.Random(a.seed).shuffle(todo)
        todo = todo[:a.n]
    a.out.mkdir(parents=True, exist_ok=True)
    print(f"  图 {len(todo):,} 张   进程 {a.workers}   去重跳过 {dups:,}", flush=True)
    t0 = time.time()
    rows = []
    step = max(2000, len(todo) // 40)
    with ProcessPoolExecutor(a.workers) as pool:
        for i, r in enumerate(pool.map(work, todo, chunksize=16), 1):
            rows.append(r)
            if i % step == 0 or i == len(todo):
                el = time.time() - t0
                print(f"    {i:,}/{len(todo):,}   {i / max(el, 1e-9):.1f} 张/秒   "
                      + " ".join(f"{k}:{v:,}" for k, v in Counter(f"{x['verdict']}/{x['kind']}" for x in rows).most_common(8)), flush=True)
    wall = time.time() - t0
    tmp = a.out / "icon.csv.tmp"
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=KEYS, extrasaction="ignore")
        w.writeheader()
        w.writerows(r for r in rows if not a.flagged_only
                    or r.get("verdict") == "Suspicious" or r.get("kind") in KEEP_KINDS)
    os.replace(tmp, a.out / "icon.csv")
    c = Counter(f"{x['verdict']}/{x['kind']}" for x in rows)
    lines = ["=" * 70, f"  蓝图页头图标 status_icon   图 {len(rows):,} 张   {wall:.0f} 秒   {len(rows) / max(wall, 1e-9):.1f} 张/秒", "=" * 70]
    lines += [f"  {k:<36}{v:>9,}" for k, v in c.most_common()]
    lines.append("=" * 70)
    (a.out / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
