r"""蓝底成功页 标题前的圈对号检查(demo/CircleCheck.cs 的 Python 对数版)。

背景
----
2026-10-26 经理在一张真图(到账成功)上圈了三样: 标题前的圈对号、¥、0, 说"顺便帮我写一个这个圈符号的检测代码,
这个对号的检测帮我也搞好"。¥ 归 YenCheck, 0 归 AmountFontCheck, 这里管圈对号。

量出来的
--------
- 支付宝自己的圈是正圆(1320x2868 上外径 54x54, 线粗 3), 对号在圈里的位置、大小在所有机型上都一样。
- 9-16 假图 2.png、1025 假图 和 TG 小图 013.jpg 是同一个贴上去的图标: 外框 58x56, 横向拉宽, 是椭圆。
- 1026 假图(安卓到账成功)标题前根本没有圈对号。

判据
----
1. 圈不圆: 从圈心沿 72 个方向取剖面, 内外沿各用自己那一侧的底色算半高位置, 中线 = 内外沿的平均;
   对边相加得 36 个方向的直径, 每个方向取前后 20 度扇区平均, 最大 / 最小 > 1.023 就报。
   本机真图原图 2,161 张最大 1.0064; 真图压缩缩小的变体 3 万多个最大 1.0187; 假图连变体最小 1.0274, 一般 1.03 ~ 1.04。
2. 标题前没有圈对号: 找不到圈的时候, 找金额上面那一行白色标题字, 标题第一个字左边那一块
   (圈本来该在的地方)一点东西都没有, 而且标题行周围是干净的蓝底, 就报。
3. 对号: 圈里要有对号, 对号相对圈的大小和位置要在真图范围里(见 CK_* 常量)。

都只管蓝底页; 高 / 宽 < 1.6(拍屏照片有透视)不判; 圈太小(外径 < 24 像素)不判;
"没有圈"那条在小图(金额行高 < 60)、标题发灰、标题最左边像图标、标题上面还有图标、标题附近有别的颜色时不判。

本机 10,986 张: 真图 0 报(蓝底真图 2,339 张里 Ok 2,161、不判 178); 四张已知假图全报;
以前的可疑集(AI 图 / 红队缩放攻击)77 张里报 11 张(都是圈不圆)。连同各种变体和刁难图共 31,710 个输入: 真图 0 报;
和 C# 逐项一样(不一样的 28 个都是截断 JPEG 这种解码层面的差别)。

用法: python training/circle_scan.py 清单.txt 输出.csv [--workers 8]
只读图片。
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

from locate_blue import is_blue_page, locate_amount_blue  # noqa: E402

OK, SUSPICIOUS, CANNOT = 1, 2, 0

MIN_ASPECT = 1.6             # 高 / 宽 低于这个不判: 拍屏(4:3 照片有透视, 圆会变椭圆)、裁过的图、横屏
WHITE_MIN = 140              # 标题带里的"白": 三个通道都 > 140(蓝底的 R 才二十多)
MIN_AREA = 6                 # 连通块至少这么多像素
MAX_COMPONENTS = 20000       # 标题带里白块超过这么多不判(上限保护)
HOLE_MIN_AREA = 3            # 孔至少这么多像素
RING_CX = (0.10, 0.62)       # 圈心横坐标 / 图宽
RING_H = (0.010, 0.08)       # 圈高 / 图宽
RING_AR = (0.8, 1.25)        # 圈的宽 / 高
HOLE_BIG = 0.5               # 大孔: 宽、高都 >= 圈的这个比例
TITLE_MIN = 3                # 圈右边同一行至少这么多个字块
NANG = 72                    # 射线条数
STEP = 0.25                  # 射线上的采样步长(像素)
R_LO, R_HI = 0.62, 1.30      # 射线从 0.62 R0 量到 1.30 R0(R0 = 外框半径), 避开里面的对号
MIN_CONTRAST = 40            # 剖面峰值比两侧底色至少亮这么多
MIN_RAYS = 60                # 有效射线至少这么多条
MIN_SECTORS = 18             # 有效扇区至少这么多个(共 36 个)
SECTOR_HALF = 20             # 扇区半宽(度)
MIN_RING_D = 24.0            # 圈外径小于这个不判
ROUND_MAX = 1.023            # 直径 最大 / 最小 超过就报: 真图原图 <= 1.0064, 压缩缩小的变体 <= 1.0187; 假图连变体 >= 1.0274

# 对号相对圈(外径 D)的范围: 真图 2,161 张是 宽 0.393~0.521 高 0.242~0.335 横偏 0.019~0.043 竖偏 -0.007~0.021, 再放宽
CK_W = (0.34, 0.60)          # 对号宽 / D
CK_H = (0.20, 0.40)          # 对号高 / D
CK_DX = (-0.05, 0.11)        # 对号中心 - 圈心(横) / D
CK_DY = (-0.08, 0.10)        # 对号中心 - 圈心(竖) / D
CK_MIN_D = 30.0              # 圈外径小于这个只看圆不圆, 对号不判(小图对号被糊掉一圈)

# 标题前没有圈(都按金额行高 adh 算)
ROW_H = (0.12, 0.70)         # 标题字块高 / adh
ROW_W_MAX = 1.2              # 字块宽上限 / adh
ROW_CY = (0.35, 2.20)        # 字块中心在金额上沿以上 0.35~2.20 adh(真图 0.67~1.35, 1026 假图 1.77)
ROW_X = (0.15, 0.85)         # 字块中心横坐标 / 图宽(排掉右上角的 完成 / 回首页)
ROW_TR = (0.18, 0.75)        # 标题行高 / adh
ROW_MIN_TR = 16              # 标题行高至少这么多像素
MAIN_H = 0.5                 # 定标题左右沿只用高 >= 行高这个比例的字块
MIN_ADH_MISSING = 60         # 金额行高 < 60 像素(约宽 650 以下的小图)不判"没有圈": 小图里对号和圈粘在一起, 圈认不出来
LEAD_SQUARE = (0.75, 1.33)   # 最左字块近似正方形 ...
LEAD_HOLE = 0.10             # ... 且孔的总面积 >= 外框 10%: 当它是没认出来的圈, 不判
TITLE_BRIGHT = 200           # 标题字像素三通道最小值的 90 分位低于这个(发灰)不判
ROW_CX = (0.30, 0.75)        # 标题行中心横坐标 / 图宽
LEAD_GAP = 0.15              # 最左一簇和第二簇隔开 >= 0.15 adh: 当它是没认出来的图标, 不判
LEAD_TALL = 1.10             # 最左一簇比其余各簇高度中位数高 10% 以上: 同上
ABOVE_H = 0.12               # 标题正上方有 >= 0.12 adh 高的白块(比如 "支" 字标志): 是别的页型, 不判
ABOVE_PAD = 0.5              # "正上方" 横向比标题行左右各宽 0.5 adh
RING_FILL = 0.35             # 标题行别处找圈: 白像素 <= 外框 35%(真圈 0.19 ~ 0.21) ...
RING_HOLE = 0.75             # ... 且正好一个宽高都 >= 外框 75% 的孔(真圈 0.89)
SLOT_X = (0.95, 0.10)        # 空位横向: 标题左沿往左 0.95 adh 到 0.10 adh
SLOT_HALF = 0.42             # 空位竖向: 标题行中心上下各 0.42 adh
SLOT_MIN_W = 0.5             # 空位裁剪后至少这么宽(adh)
SLOT_DIST2 = 1600            # 离底色 > 40 算有东西
SLOT_TOL = 0.002             # 空位里有东西的像素占比 <= 这个算空(JPEG 噪点)
FOREIGN_DIST2 = 6400         # 离 "底色 -> 白" 线段 > 80 算别的颜色
FOREIGN_MAX = 0.01           # 标题行上下半行范围里别的颜色占比超过这个不判


def load_rgb(path):
    with Image.open(path) as im:
        return np.asarray(ImageOps.exif_transpose(im).convert("RGB"), np.uint8)


def holes_box(mask: np.ndarray):
    """字块里的孔(外接框外补一圈 0, 背景 4 连通, 不碰外圈的); 返回 [(x, y, w, h, 面积)], 坐标相对字块外接框。"""
    inv = (1 - np.pad(mask.astype(np.uint8), 1)).astype(np.uint8)
    n, _lab, st, _cen = cv2.connectedComponentsWithStats(inv, connectivity=4)
    out = []
    for i in range(1, n):
        x, y, w, h, a = (int(v) for v in st[i])
        if x == 0 or y == 0 or x + w >= inv.shape[1] or y + h >= inv.shape[0] or a < HOLE_MIN_AREA:
            continue
        out.append((x - 1, y - 1, w, h, a))
    return out


def _lower_median(v: np.ndarray) -> int:
    """整数下中位数(和 C# 直方图口径一样: 第一个累计数 >= ceil(n/2) 的值)。"""
    s = np.sort(v)
    return int(s[(len(s) + 1) // 2 - 1])


def _bilinear(img, xs, ys):
    h, w = img.shape
    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    ok = bool(((x0 >= 0) & (y0 >= 0) & (x0 + 1 < w) & (y0 + 1 < h)).all())
    if not ok:
        return None
    fx = xs - x0
    fy = ys - y0
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy)
            + img[y0 + 1, x0] * (1 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy)


def _cross(p, rs, k0, k1, level):
    """剖面 p 从下标 k0 往 k1 走, 第一次越过 level 的亚像素半径; 没越过返回 None。"""
    step = 1 if k1 >= k0 else -1
    for k in range(k0, k1, step):
        a, b = p[k], p[k + step]
        if (a - level) * (b - level) <= 0 and a != b:
            t = (level - a) / (b - a)
            return rs[k] + t * (rs[k + step] - rs[k])
    return None


def ring_rays(mnf, lab, label, x, y, w, h):
    """沿 72 个方向量圈的内沿 / 外沿半径。返回 (cx, cy, rin[72], rout[72]), 量不到的是 nan。"""
    ys, xs = np.nonzero(lab[y:y + h, x:x + w] == label)
    cx = x + float(xs.sum()) / len(xs)
    cy = y + float(ys.sum()) / len(ys)
    r0 = (w + h) / 4.0
    lo, hi = R_LO * r0, R_HI * r0
    n = int(np.ceil((hi - lo) / STEP))
    rs = lo + np.arange(n) * STEP
    rin = np.full(NANG, np.nan)
    rout = np.full(NANG, np.nan)
    H, W = lab.shape
    for k in range(NANG):
        t = 2.0 * np.pi * k / NANG
        px = cx + rs * np.cos(t)
        py = cy + rs * np.sin(t)
        p = _bilinear(mnf, px, py)
        if p is None:
            continue
        pk = int(np.argmax(p))
        if pk == 0 or pk == n - 1:
            continue
        top = p[pk]
        b_in = p[:pk].min()
        b_out = p[pk:].min()
        if top - max(b_in, b_out) < MIN_CONTRAST:
            continue
        a = _cross(p, rs, pk, 0, (top + b_in) / 2.0)
        b = _cross(p, rs, pk, n - 1, (top + b_out) / 2.0)
        if a is None or b is None:
            continue
        # 半高范围(各外扩 1 像素)里碰到别的白块(对号、标题字)就不要这条
        k_lo = max(0, int(np.searchsorted(rs, a)) - 4)
        k_hi = min(n - 1, int(np.searchsorted(rs, b)) + 4)
        hit = False
        for kk in range(k_lo, k_hi + 1):
            gx = int(np.floor(px[kk] + 0.5))
            gy = int(np.floor(py[kk] + 0.5))
            if 0 <= gx < W and 0 <= gy < H:
                v = lab[gy, gx]
                if v != 0 and v != label:
                    hit = True
                    break
        if hit:
            continue
        rin[k], rout[k] = a, b
    return cx, cy, rin, rout


def _sector_mean(D, center_deg):
    vals = []
    for j in range(NANG // 2):
        ang = j * (360.0 / NANG)
        m = ((ang - center_deg) + 90.0) % 180.0
        if abs(m - 90.0) <= SECTOR_HALF and not np.isnan(D[j]):
            vals.append(D[j])
    return _seq_sum(vals) / len(vals) if len(vals) >= 3 else None


def _seq_sum(vals) -> float:
    """从左到右逐个相加(和 C# 一样; Python 3.12 的 sum() 对浮点做补偿求和, numpy 是两两求和, 末位会不同)。"""
    acc = 0.0
    for v in vals:
        acc += float(v)
    return acc


def _find_ring(W, comps, lab):
    """圈: 外框近似正方形、正好一个大孔、右边同一行 >= 3 个标题字块; 多个取最靠下的。"""
    best = None
    for (x, y, w, h, a, i) in comps:
        if not (RING_CX[0] * W <= x + w / 2.0 <= RING_CX[1] * W):
            continue
        if not (RING_H[0] * W <= h <= RING_H[1] * W and RING_AR[0] <= w / h <= RING_AR[1]):
            continue
        big = [o for o in holes_box(lab[y:y + h, x:x + w] == i) if o[2] >= HOLE_BIG * w and o[3] >= HOLE_BIG * h]
        if len(big) != 1:
            continue
        cy = y + h / 2.0
        right = [c for c in comps if x + w < c[0] < x + w + 8 * h
                 and abs((c[1] + c[3] / 2.0) - cy) < 0.6 * h and c[3] >= 0.4 * h]
        if len(right) < TITLE_MIN:
            continue
        # 取最靠下的; 一样低取最靠左的(不靠连通域编号顺序, 各版本 OpenCV 编号可能不同)
        if best is None or y > best[0][1] or (y == best[0][1] and x < best[0][0]):
            best = ((x, y, w, h, a, i), big[0], right)
    return best


def _row_clusters(row):
    """一行字块按横向重叠并成簇, 返回 [(左, 右, 上, 下, [字块...])] 从左到右。"""
    out = []
    for c in sorted(row, key=lambda c: (c[0], c[1])):
        l, r, t, b = c[0], c[0] + c[2], c[1], c[1] + c[3]
        if out and l < out[-1][1]:
            pl, pr, pt, pb, mem = out[-1]
            out[-1] = (pl, max(pr, r), min(pt, t), max(pb, b), mem + [c])
        else:
            out.append((l, r, t, b, [c]))
    return out


def _title_row(comps, adh, ay0_band, W):
    """金额上面那一行白色标题字: 竖向重叠 >= 一半的字块连成行, 取最靠下、>= 3 块的那一行。"""
    cand = [c for c in comps if ROW_H[0] * adh <= c[3] <= ROW_H[1] * adh and c[2] <= ROW_W_MAX * adh
            and ROW_CY[0] * adh <= ay0_band - (c[1] + c[3] / 2.0) <= ROW_CY[1] * adh
            and ROW_X[0] * W <= c[0] + c[2] / 2.0 <= ROW_X[1] * W]
    n = len(cand)
    if n < TITLE_MIN:
        return None
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for p in range(n):
        for q in range(p + 1, n):
            t = max(cand[p][1], cand[q][1])
            b = min(cand[p][1] + cand[p][3], cand[q][1] + cand[q][3])
            if b - t >= 0.5 * min(cand[p][3], cand[q][3]):
                ra, rb = find(p), find(q)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
    groups = {}
    for p in range(n):
        groups.setdefault(find(p), []).append(cand[p])
    best = None
    for g in groups.values():
        if len(g) < TITLE_MIN:
            continue
        bot = max(c[1] + c[3] for c in g)
        left = min(c[0] for c in g)
        # 取最靠下的行; 一样低取最靠左的
        if best is None or bot > best[0] or (bot == best[0] and left < best[1]):
            best = (bot, left, g)
    return None if best is None else best[2]


def _bg_and_foreign(band, wmask, x0, x1, y0, y1, fy0, fy1):
    """底色 = [y0,y1) x [x0,x1) 里非白像素每个通道的下中位数; 再算 [fy0,fy1) x [x0,x1) 里离"底色->白"线段 > 80 的占比。"""
    reg = band[y0:y1, x0:x1].reshape(-1, 3)
    nw = ~wmask[y0:y1, x0:x1].reshape(-1)
    if nw.sum() == 0:
        return None, 1.0
    bg = np.array([_lower_median(reg[nw, c]) for c in range(3)], np.float64)
    d = 255.0 - bg
    L = d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
    pxs = band[fy0:fy1, x0:x1].reshape(-1, 3).astype(np.float64) - bg
    if len(pxs) == 0:
        return bg, 0.0
    t = (pxs[:, 0] * d[0] + pxs[:, 1] * d[1] + pxs[:, 2] * d[2]) / L if L else np.zeros(len(pxs))
    t = np.clip(t, 0.0, 1.0)
    e0, e1, e2 = pxs[:, 0] - t * d[0], pxs[:, 1] - t * d[1], pxs[:, 2] - t * d[2]
    far = (e0 * e0 + e1 * e1 + e2 * e2) > FOREIGN_DIST2
    return bg, float(far.sum()) / len(far)


def check(rgb: np.ndarray) -> dict:
    """主入口: RGB 图 -> 结论和量到的数(键名和 C# 的 CircleResult 对应)。"""
    res = dict(verdict=CANNOT, reason="", ring_found=0, roundness=np.nan, ratio_hv=np.nan, ring_d=np.nan,
               ring_d_adh=np.nan, n_rays=0, check_found=0, ck_w=np.nan, ck_h=np.nan, ck_dx=np.nan, ck_dy=np.nan,
               row_found=0, title_bright=np.nan, lead_gap=np.nan, lead_tall=np.nan, slot_frac=np.nan, foreign=np.nan)
    H, W = rgb.shape[:2]
    if W < 64 or H < 128:
        res["reason"] = "图太小"
        return res
    if H < MIN_ASPECT * W:
        res["reason"] = "长宽比不像手机截图(拍屏 / 裁过 / 横屏), 不判"
        return res
    if not is_blue_page(rgb):
        res["reason"] = "不是蓝底页, 本类只管蓝底页"
        return res
    loc = locate_amount_blue(rgb)
    if not loc:
        res["reason"] = "定位不到金额行"
        return res
    ax0, ay0, ax1, ay1 = loc[:4]
    adh = ay1 - ay0
    top, bot = int(0.04 * H), int(ay0 - 0.15 * adh)
    if bot - top < 20:
        res["reason"] = "金额上面没地方放标题"
        return res
    band = rgb[top:bot]
    mn = band.min(axis=2)
    wmask = mn > WHITE_MIN
    n, lab, st, _ = cv2.connectedComponentsWithStats(wmask.astype(np.uint8), connectivity=8)
    if n - 1 > MAX_COMPONENTS:
        res["reason"] = "白块太多(花图), 不判"
        return res
    comps = [(int(st[i][0]), int(st[i][1]), int(st[i][2]), int(st[i][3]), int(st[i][4]), i)
             for i in range(1, n) if st[i][4] >= MIN_AREA]

    ring = _find_ring(W, comps, lab)
    if ring is not None:
        return _judge_ring(res, ring, comps, mn, lab, adh)
    return _judge_missing(res, band, mn, lab, wmask, comps, adh, ay0 - top, W)


def _judge_ring(res, ring, comps, mn, lab, adh):
    (x, y, w, h, a, label), (hx, hy, hw, hh, _ha), _right = ring
    res["ring_found"] = 1
    cx, cy, rin, rout = ring_rays(mn.astype(np.float64), lab, label, x, y, w, h)
    rmid = (rin + rout) / 2.0
    half = NANG // 2
    Dm = rmid[:half] + rmid[half:]
    Do = rout[:half] + rout[half:]
    nray = int((~np.isnan(rmid)).sum())
    res["n_rays"] = nray
    do_ok = [float(v) for v in Do if not np.isnan(v)]
    ring_d = _seq_sum(do_ok) / len(do_ok) if do_ok else float("nan")
    res["ring_d"] = ring_d
    res["ring_d_adh"] = ring_d / adh if adh else float("nan")
    secs = [_sector_mean(Dm, c) for c in range(0, 180, 5)]
    good = [s for s in secs if s is not None]
    if secs[0] is not None and secs[18] is not None:
        res["ratio_hv"] = secs[0] / secs[18]
    if len(good) >= MIN_SECTORS:
        res["roundness"] = max(good) / min(good)

    # 对号: 中心落在大孔外接框里的白块
    inside = [c for c in comps if c[5] != label and x + hx <= c[0] + c[2] / 2.0 <= x + hx + hw
              and y + hy <= c[1] + c[3] / 2.0 <= y + hy + hh]
    if inside and not np.isnan(ring_d):
        res["check_found"] = 1
        l = min(c[0] for c in inside)
        r = max(c[0] + c[2] for c in inside)
        t = min(c[1] for c in inside)
        b = max(c[1] + c[3] for c in inside)
        res["ck_w"] = (r - l) / ring_d
        res["ck_h"] = (b - t) / ring_d
        res["ck_dx"] = ((l + r) / 2.0 - cx) / ring_d
        res["ck_dy"] = ((t + b) / 2.0 - cy) / ring_d

    if np.isnan(ring_d) or ring_d < MIN_RING_D:
        res["reason"] = "圈太小, 不判"
        return res
    if nray < MIN_RAYS or np.isnan(res["roundness"]):
        res["reason"] = "圈的边量不全(被别的东西挡了), 不判"
        return res
    if res["roundness"] > ROUND_MAX:
        res["verdict"] = SUSPICIOUS
        res["reason"] = f"圈不圆: 直径 最大/最小 = {res['roundness']:.4f}(真图 <= 1.019)"
        return res
    if not res["check_found"]:
        res["reason"] = "圈里没有对号, 不是成功页的图标, 不判"
        return res
    if ring_d < CK_MIN_D:
        res["verdict"] = OK
        res["reason"] = f"圈是正圆({res['roundness']:.4f}); 图小, 对号只看有没有"
        return res
    bad = []
    if not (CK_W[0] <= res["ck_w"] <= CK_W[1]):
        bad.append(f"对号宽/圈 {res['ck_w']:.3f}")
    if not (CK_H[0] <= res["ck_h"] <= CK_H[1]):
        bad.append(f"对号高/圈 {res['ck_h']:.3f}")
    if not (CK_DX[0] <= res["ck_dx"] <= CK_DX[1]):
        bad.append(f"对号横向偏移 {res['ck_dx']:+.3f}")
    if not (CK_DY[0] <= res["ck_dy"] <= CK_DY[1]):
        bad.append(f"对号竖向偏移 {res['ck_dy']:+.3f}")
    if bad:
        res["verdict"] = SUSPICIOUS
        res["reason"] = "对号不对: " + ", ".join(bad)
        return res
    res["verdict"] = OK
    res["reason"] = f"圈是正圆({res['roundness']:.4f}), 对号位置大小正常"
    return res


def _judge_missing(res, band, mn, lab, wmask, comps, adh, ay0_band, W):
    if adh < MIN_ADH_MISSING:
        res["reason"] = "没找到圈; 图太小, 不判有没有圈"
        return res
    row = _title_row(comps, adh, ay0_band, W)
    if row is None:
        res["reason"] = "没找到圈, 也没找到白色标题行, 不判"
        return res
    rt = min(c[1] for c in row)
    rb = max(c[1] + c[3] for c in row)
    tr = rb - rt
    # 字块按横向重叠并成簇(一个字上下几块会并成一簇), 只拿够高的簇(>= 行高一半)定标题左右沿:
    # 发灰的圈压暗后只剩几个碎点, 不能让碎点把左沿拉过去
    cl = [k for k in _row_clusters(row) if k[3] - k[2] >= MAIN_H * tr]
    if len(cl) < 2:
        res["reason"] = "没找到圈, 标题行够高的字簇不到两个, 不判"
        return res
    rl, rr = cl[0][0], cl[-1][1]
    if tr < ROW_MIN_TR or not (ROW_TR[0] * adh <= tr <= ROW_TR[1] * adh) or rr - rl < 2.0 * tr \
            or not (ROW_CX[0] * W <= (rl + rr) / 2.0 <= ROW_CX[1] * W):
        res["reason"] = "没找到圈, 标题行不像标题, 不判"
        return res
    res["row_found"] = 1
    # 标题字够不够白: 页面被半透明遮罩压暗时圈也发灰, 白色门槛下圈会散掉, 这种不判
    members = [c for k in cl for c in k[4]]
    vals = np.sort(np.concatenate([mn[c[1]:c[1] + c[3], c[0]:c[0] + c[2]][lab[c[1]:c[1] + c[3], c[0]:c[0] + c[2]] == c[5]]
                                   for c in members]))
    res["title_bright"] = int(vals[int(np.ceil(0.9 * len(vals))) - 1])
    if res["title_bright"] < TITLE_BRIGHT:
        res["reason"] = f"标题字发灰({res['title_bright']}), 页面可能被压暗, 不判"
        return res
    res["lead_gap"] = (cl[1][0] - cl[0][1]) / adh
    others = sorted(k[3] - k[2] for k in cl[1:])
    m = len(others)
    med = others[m // 2] if m % 2 else (others[m // 2 - 1] + others[m // 2]) / 2.0
    res["lead_tall"] = (cl[0][3] - cl[0][2]) / med
    if res["lead_gap"] >= LEAD_GAP or res["lead_tall"] > LEAD_TALL:
        res["reason"] = "标题行最左边有个单独的东西(可能是没认出来的图标), 不判"
        return res
    # 最左一簇近似正方形又带孔: 多半是对号和圈粘在一起、没认出来的圈
    lw, lh = cl[0][1] - cl[0][0], cl[0][3] - cl[0][2]
    hole_area = sum(o[4] for c in cl[0][4] for o in holes_box(lab[c[1]:c[1] + c[3], c[0]:c[0] + c[2]] == c[5]))
    if LEAD_SQUARE[0] <= lw / lh <= LEAD_SQUARE[1] and hole_area >= LEAD_HOLE * lw * lh:
        res["reason"] = "标题行最左边的字块像个圈(带孔), 不判"
        return res
    # 标题正上方还有白块(不碰标题带上沿, 那是状态栏): 图标在标题上面的别的页型
    for c in comps:
        if c[1] > 0 and c[1] + c[3] <= rt and c[3] >= ABOVE_H * adh \
                and rl - ABOVE_PAD * adh <= c[0] + c[2] / 2.0 <= rr + ABOVE_PAD * adh:
            res["reason"] = "标题上面还有图标或字(别的页型), 不判"
            return res
    # 标题这一行别处有个圈(比如整张图是镜像的, 圈跑到标题右边): 有图标只是不在左边, 不判。
    # 圈 = 近似正方形、细(像素 <= 外框 35%)、正好一个占外框 75% 以上的大孔; 字里的 口 日 贝 孔小、笔画粗, 不会算进来
    for (x, y, w, h, a, i) in comps:
        if not (RING_H[0] * W <= h <= RING_H[1] * W and RING_AR[0] <= w / h <= RING_AR[1] and a <= RING_FILL * w * h):
            continue
        if not (rt - 0.5 * tr <= y + h / 2.0 <= rb + 0.5 * tr):
            continue
        if len([o for o in holes_box(lab[y:y + h, x:x + w] == i) if o[2] >= RING_HOLE * w and o[3] >= RING_HOLE * h]) == 1:
            res["reason"] = "标题这一行别处有个圈(镜像图或别的版式), 不判"
            return res

    bh, bw = band.shape[:2]
    rcy = (rt + rb) / 2.0
    sx0 = max(0, int(rl - SLOT_X[0] * adh))
    sx1 = max(0, int(rl - SLOT_X[1] * adh))
    sy0 = max(0, int(rcy - SLOT_HALF * adh))
    sy1 = min(bh, int(rcy + SLOT_HALF * adh))
    if sx1 - sx0 < SLOT_MIN_W * adh or sy1 - sy0 < 4:
        res["reason"] = "标题左边没有空间, 不判"
        return res
    gx0, gx1 = int(0.05 * W), int(0.95 * W)
    fy0 = max(0, int(rt - 0.5 * tr))
    fy1 = min(bh, int(rb + 0.5 * tr))
    bg, foreign = _bg_and_foreign(band, wmask, gx0, gx1, rt, rb, fy0, fy1)
    res["foreign"] = foreign
    if bg is None:
        res["reason"] = "标题行没有底色, 不判"
        return res
    if foreign > FOREIGN_MAX:
        res["reason"] = f"标题行附近有别的颜色({foreign:.3f}), 不判"
        return res
    slot = band[sy0:sy1, sx0:sx1].reshape(-1, 3).astype(np.float64) - bg
    d2 = slot[:, 0] * slot[:, 0] + slot[:, 1] * slot[:, 1] + slot[:, 2] * slot[:, 2]
    res["slot_frac"] = float((d2 > SLOT_DIST2).sum()) / len(d2)
    if res["slot_frac"] <= SLOT_TOL:
        res["verdict"] = SUSPICIOUS
        res["reason"] = "标题前没有圈对号(标题左边是空的蓝底)"
        return res
    res["reason"] = "标题前有东西, 但认不出是圈对号, 不判"
    return res


def main() -> None:
    import argparse
    import csv
    import os
    from multiprocessing import Pool

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description="蓝底成功页 圈对号检查(批量)")
    ap.add_argument("list", help="图片清单, 一行一个路径")
    ap.add_argument("out", help="输出 csv")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(k, "1")
    paths = [l.strip() for l in open(args.list, encoding="utf-8") if l.strip()]
    with Pool(args.workers) as pool:
        rows = pool.map(_one, paths, chunksize=16)
    keys = ["path", "W", "H", "verdict", "reason", "ring_found", "roundness", "ratio_hv", "ring_d", "ring_d_adh",
            "n_rays", "check_found", "ck_w", "ck_h", "ck_dx", "ck_dy", "row_found", "lead_gap", "lead_tall",
            "slot_frac", "foreign", "title_bright"]
    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    from collections import Counter
    print("verdicts", dict(Counter(r["verdict"] for r in rows)))


def _one(path):
    try:
        rgb = load_rgb(path)
    except Exception as e:  # noqa: BLE001
        return dict(path=path, W=0, H=0, verdict=CANNOT, reason=f"读不了: {type(e).__name__}")
    r = check(rgb)
    r.update(path=path, W=rgb.shape[1], H=rgb.shape[0])
    return r


if __name__ == "__main__":
    main()
