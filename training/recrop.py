r"""切图修正: 每一段按**它自己的字**定上下边, 不再整行共用一条"成员中位"的边。

为什么(2026-10-23)
------------------
make_pinyin_pairs 切行时, 行的上下边取的是这一行所有字块的**中位**(故意的: 防 〈 箭头这种高元件把边顶进上面的拼音带),
再只留 1 像素余量; split_segments 切段时整行共用这条边。于是一行里**比一般字高、或者位置偏高偏低**的那一段就被切掉一截:
    大号红色券面额(0.9元 -> 只剩下半截, 老师读成 U.7元)、蓝图标题行右上角的 回首页(上半截没了)、大一号字体的名字 ...
金标准上模型 h 剩下的错里, 这是最大的一类(标签修完以后)。

做法(只改上下边, 左右不动):
    整页: 和 make_pinyin_pairs.process 一模一样地算 字块 / 拼音块 / 拼音带 / 行 / 每行的 label 边 (la, lb) 和 input 边 (ia, ib)
    每一段 x 区间 [a, b) 里, 挑属于这一段的字块:
        横向至少一半在段里; 纵向和这一行的 label 边 [la, lb] 重叠至少自身高度的 30%(上一行下伸、下一行顶的不算);
        中心在 [la - 0.35 行高, lb + 0.35 行高] 里(行边夹在两行中间时, 上下两行的字中心在外面);
        不是拼音块(annotation_labels 判的), 中心不在拼音带里; 不是超高的竖线 / 框 / 图标列(高 > 2.4 倍行高)、不是又扁又长的横线
        (试过按"填满外接框的比例"去掉实心图标: 不行, 粗体大号金额也填 0.56~0.64, 和图标 0.55 分不开)
    新的 input 边 = 原来的边 和 这些字块的上下沿 取外包(上下各留 2 像素), 最多各往外扩 2 倍行高, 不越过上下邻行
★ 原来的边一律保留(只往外扩, 不往里收): 拼音还在上面, 正常的段和原来一模一样。

用法
----
    python training/recrop.py demo --src 图目录或名单 [...] --out 输出目录 [--n 40]      # 出前后对比核对图
    python training/recrop.py regold --gold 答案目录 [...] --out-root 输出 [--min-ext 0.15 --suffix _rcB]   # 金标准从原图重切
    python training/recrop.py recut --pairs 配对目录 [...] [--workers 20]      # 训练集的段重切到 <目录>\seg_input_rc (可接着跑)
    python training/recrop.py relist --src 旧训练集目录 --out 新训练集目录 [--min-rc 0.97]   # 清单换成重切图
    (v3 用 --rc-name seg_input_rc3: 新切法的图放另一个目录, 和 v2 的 seg_input_rc 不混; 指纹不一样也会被拦)

v3(10-24, step 45): 券面大数字(中心落在 元 的拼音带里被当成拼音)、实心按钮(白字是按钮里的洞, 不是块)、
    蓝图 回首页(两行上下交叠)还被切的三个原因, 加了 P1-P7 七条规则(见常数旁边的说明)。原因是在本地 9,068 张页面上找的;
    段边变了多少: 300 张拼音页 1.6%、113 张蓝图 6.1%(10-24 复核在 2,500 张拼音页上 2.15%), 没有一段比旧切法还小;
    端口版和量过的补丁 8,919 / 8,919 段一模一样。
    已知的小毛病(10-24 复核, 不影响训练): FRAG 段的行带夹在两行中间时, 上下两行的字都擦边, 会切成两行(569 段 FRAG 里 31 段);
    试过要求字真伸进带里, 但会把 去看看 / 通知TA / 回首页 这种本来就只擦边几像素的又切掉一截(83 段), 所以不改。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from erase_pinyin import annotation_labels, local_background, text_mask  # noqa: E402
from make_pinyin_pairs import FALLBACK_SHIFT_RATIO, LABEL_PAD, MIN_ROW_H, _pinyin_bands, _rows  # noqa: E402

PAD_Y = 2          # 新边上下各留几像素
CAP = 2.0          # 每边最多往外扩几倍行高
V_OVERLAP = 0.3    # 字块和行的 label 边至少重叠自身高度的这么多
CENTER = 0.35      # 字块中心离这一行的 label 边不能超过这么多倍行高(大号金额底对齐, 中心正好在行顶附近)
TALL = 2.4         # 比行高这么多倍还高的是竖线 / 框 / 一列图标, 不算字(核对图: 大号金额最高 2.0 倍, 图标列 2.49 倍)
CONTAINER_H, CONTAINER_N = 1.2, 3   # 比行高 1.2 倍高、外接框里整个装着 >= 3 块别的: 按钮外圈 / 图标 / 头像, 不算字
# ---- v3(10-24, step 45): 券面大数字 / 实心按钮 / 蓝图 回首页 还被切的三个原因, 原因在本地 9,068 张页面上找的(RECROP-GAP-2026-10-23.md) ----
# P1 拼音带只挡"带那么大"的块: 块和带的重叠 >= 自身高度的这么多才算拼音。
#    大号券面额(0.5元 的 0.5)中心正好落在 元 上面那排 yuán 的带里, 原来整块被当拼音扔掉 -> 金额顶上一截切掉。
BAND_FRAC = 0.5
# P2 实心按钮(红底 / 蓝底白字): text_mask 把整个按钮当一块墨, 白字是按钮里的"洞", 根本不是块。
#    按钮本身永远不按它扩; 它的洞(字和字上的拼音)当字块用。判按钮: 宽 >= 1.8 倍高, 上下沿中间一半是连续的墨, 洞 >= 2 个。
BTN_W, BTN_EDGE, BTN_HOLES = 1.8, 0.9, 2
# 实心: 墨占外接框 >= BTN_FILL, 洞的总面积 <= 墨的面积。卡片外框 / 横幅边框 / 头像(10-23 核对图: 墨 0.04~0.14, 洞是墨的 5~27 倍)
#   里面整个是洞, 不是按钮; 真按钮 0.54~0.84, 洞(字)是墨的 0.17~0.66
BTN_FILL, BTN_HOLE_AREA = 0.45, 1.0
# P3 FRAG: 这一段在 label 带里一个自己的字都没有(全是上一行 / 按钮 / 下一行伸进来的一截) ->
#    按伸进带里的那些字块(按钮的话是它的洞)的全高切, 不按邻行卡(邻行就是这些字自己的那一行), 只按 CAP 卡(band_below 留着没用)。
FRAG_MIN_H = 0.4   # 伸进来的块至少这么高(倍行高)才算字, 更小的是点 / 碎屑, 不为它扩
FRAG_MIN_N = 2     # 伸进来的(不算按钮的洞)至少这么多块才按它扩: 单独一块多半是图标
# P4 这一段"自己的字"其实全是旁边**交叠的那一行**的字(那一行的带把它们整个罩住, 比这一行罩得多):
#    这一段是那一行的字伸进来的一截(蓝图 回首页: 回首页 一行 和 大标题 支付成功 一行上下交叠), 当 FRAG 处理, 不按邻行卡。
#    只要有一个字更属于这一行(比如 澳門百老匯 和状态栏 16:19 交叠), 照旧按邻行卡 —— 不把别的行拉进正常段。
# P5 比行高还高的块(大号金额), VOV 按行高算: 和 label 带重叠 >= 0.3 x min(自身高, 行高)。
#    2元 的 2 高 1.26 倍行高、底伸进带里 0.32 倍行高, 按自身高只有 0.26 被当成上一行。中心规则(CENTER)照旧挡真正的上一行大字。
# P6 自己的字被邻行的边卡住时(券卡: 2.46 的顶比左边标题行的底高 15 像素), 只在两样都满足时越过邻行的边扩到字的边:
#    被卡的块和这一段里另一个没被卡的字**同色**(2.46 和旁边的 元 都是红的; 图标和旁边的字颜色不一样 —— 不按颜色时 10-23 核对图上
#    主要是把图标补全), 而且被卡掉的那一截(段的 x 范围里)除了自己的字什么都没有。
P6_COLOR = 40      # 同色: BGR 各通道均值差都不超过这么多
P6_REF_A = 20      # 拿来比颜色的字块至少这么多像素(小数点 . 也算: 2.1元 的 元 和底下红条粘成一块, 只剩 . 能比)
# P7 按钮的字(洞, >= FRAG_MIN_H 行高)碰到 label 带, 就把这个按钮上的字整个算进来 —— 不管按钮外框过不过 TALL / 中心 / 外圈。
#    (段里同时有别的东西, 比如右边一条蓝色侧栏, 不走 FRAG, 按钮的字原来就没人管; 10-23 核对图: 去看看 立即领 通知TA 还是切掉一半)


def layout(img: np.ndarray) -> dict:
    """整页的字块、拼音、行 —— 和 make_pinyin_pairs.process 同一套算法、同一组常数。"""
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = text_mask(gray, local_background(gray))
    n, _lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    _, kept, _, big = annotation_labels(mask)
    bands = _pinyin_bands(kept, st, big)
    rows = _rows(big, bands)
    out = {"H": H, "W": W, "st": st, "kept": set(int(k) for k in kept), "bands": bands, "rows": [], "lab": _lab, "img": img}
    if not rows:
        return out
    char_h = float(np.median([b - a for a, b in rows]))
    fallback = int(round(FALLBACK_SHIFT_RATIO * char_h))
    for i, (y0, y1) in enumerate(rows):
        above = [(a, b) for a, b in bands if b <= y0 + 2 and (y0 - b) < (y1 - y0)]
        la = max(0, y0 - LABEL_PAD)
        if above:
            la = min(y0, max(la, max(b for _, b in above) + 1))
        lb = min(H, y1 + LABEL_PAD)
        if lb - la < MIN_ROW_H:
            continue
        ia = max(0, (min(a for a, _ in above) - 1) if above else (y0 - fallback))
        out["rows"].append({"row": i, "la": la, "lb": lb, "ia": ia, "ib": lb, "has_py": int(bool(above))})
    return out


def _encloses(st, i: int, near: list) -> int:
    """第 i 块的外接框里整个装着几块别的(按钮外圈、图标、头像里面都装着字或花纹; 字基本不装)。"""
    x, y, w, h = (int(v) for v in st[i][:4])
    n = 0
    for j in near:
        if j == i:
            continue
        xj, yj, wj, hj = (int(v) for v in st[j][:4])
        if xj > x and yj > y and xj + wj < x + w and yj + hj < y + h:
            n += 1
    return n


def _button_holes(L: dict, i: int):
    """第 i 块是不是实心按钮(红底 / 蓝底, 白字是洞); 是的话返回洞的外接框 [(x, y, w, h, A)](页坐标), 不是返回 None。
    判法: 宽 >= BTN_W 倍高; 上沿、下沿附近有一行在中间一半宽度上 >= BTN_EDGE 是墨(圆角长条的直边); 洞(墨围住、碰不到外接框边的空白)>= BTN_HOLES 个。
    粗体字粘成一串也会有洞(口 日 0 8 的内圈), 但上下沿不是一整条直边; 描边按钮(白圈)里面是一整个洞, 不到 2 个。"""
    cache = L.setdefault("btn", {})
    if i in cache:
        return cache[i]
    x, y, w, h, A = (int(v) for v in L["st"][i][:5])
    res = None
    if h >= 8 and w >= BTN_W * h and A >= BTN_FILL * w * h:
        m = (L["lab"][y:y + h, x:x + w] == i).astype(np.uint8)
        c0, c1 = w // 4, w - w // 4
        k = max(2, h // 10)
        top = max(m[r, c0:c1].mean() for r in range(0, k + 1))
        bot = max(m[h - 1 - r, c0:c1].mean() for r in range(0, k + 1))
        if top >= BTN_EDGE and bot >= BTN_EDGE:
            inv = cv2.copyMakeBorder(1 - m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=1)
            ffm = np.zeros((h + 4, w + 4), np.uint8)
            cv2.floodFill(inv, ffm, (0, 0), 2)
            holes = (inv[1:-1, 1:-1] == 1).astype(np.uint8)
            n, _l, hs, _c = cv2.connectedComponentsWithStats(holes, connectivity=8)
            boxes = [(x + int(s[0]), y + int(s[1]), int(s[2]), int(s[3]), int(s[4])) for s in hs[1:] if s[4] >= 6]
            if len(boxes) >= BTN_HOLES and sum(bx[4] for bx in boxes) <= BTN_HOLE_AREA * A:
                res = boxes
    cache[i] = res
    return res


def seg_extent(L: dict, la: int, lb: int, a: int, b: int, with_mode: bool = False):
    """这一段自己的字块的上下沿 (top, bottom); 一个都没有返回 None。(numpy 一次筛完, 51 万段也快)
    补丁: P1 拼音带只挡带那么大的块; P2 实心按钮换成它的洞; P3 一个自己的字都没有时按伸进带里的字(FRAG)。
    with_mode=True 时返回 (top, bottom, "own" | "frag")。"""
    st, bands = L["st"], L["bands"]
    if "kept_mask" not in L:
        km = np.zeros(len(st), bool)
        km[[k for k in L["kept"] if k < len(st)]] = True
        L["kept_mask"] = km
    row_h = max(1, lb - la)
    X, Y, Wc, Hc, A = (st[:, k].astype(np.int64) for k in range(5))
    idx = np.arange(len(st))
    base = (idx > 0) & (A >= 6)
    xin = (np.minimum(X + Wc, b) - np.maximum(X, a)) >= 0.5 * Wc
    # P2: 段里、行附近的实心按钮 -> 按钮自己不算, 它的洞当字块(父块号记着, FRAG 时用)
    nearb = np.nonzero(base & xin & (Y < lb + 3 * row_h) & (Y + Hc > la - 3 * row_h) & (Wc >= BTN_W * Hc) & (Hc >= 8))[0]
    btn, ps = [], []
    for i in nearb:
        hb = _button_holes(L, int(i))
        if hb:
            btn.append(int(i))
            ps.extend((hx, hy, hw, hh, ha, int(i)) for hx, hy, hw, hh, ha in hb)
    is_btn = np.zeros(len(st), bool)
    is_btn[btn] = True
    kept = L["kept_mask"].copy()
    parent = np.full(len(st), -1, np.int64)
    if ps:
        P = np.array(ps, dtype=np.int64)
        X = np.concatenate([X, P[:, 0]]); Y = np.concatenate([Y, P[:, 1]]); Wc = np.concatenate([Wc, P[:, 2]])
        Hc = np.concatenate([Hc, P[:, 3]]); A = np.concatenate([A, P[:, 4]])
        kept = np.concatenate([kept, np.zeros(len(P), bool)])
        is_btn = np.concatenate([is_btn, np.zeros(len(P), bool)])
        parent = np.concatenate([parent, P[:, 5]])
        idx = np.arange(len(X))
        base = (idx > 0) & (A >= 6)
        xin = (np.minimum(X + Wc, b) - np.maximum(X, a)) >= 0.5 * Wc
    cy = Y + Hc / 2
    pin = np.zeros(len(X), bool)                       # P1: 中心在带里, 而且身子一半以上在带里 -> 拼音
    for p, q in bands:
        pin |= (cy >= p) & (cy <= q) & ((np.minimum(Y + Hc, q) - np.maximum(Y, p)) >= BAND_FRAC * Hc)
    shape_ok = ~((Hc > TALL * row_h) | ((Wc > 3 * row_h) & (Hc < 0.25 * row_h)))
    vov = (np.minimum(Y + Hc, lb) - np.maximum(Y, la)) >= V_OVERLAP * np.minimum(Hc, row_h)
    ctr = (cy >= la - CENTER * row_h) & (cy <= lb + CENTER * row_h)
    common = base & ~kept & ~is_btn & xin & ~pin
    sel = common & shape_ok & vov & ctr
    sel &= (parent < 0) | (Hc >= FRAG_MIN_H * row_h)  # 按钮的洞单个算字时至少这么高(6x4 的小洞不算, 10-23 去看看 只扩了 3 像素)
    nreal = len(st)
    # 按钮按原规则本来就算这一段的(过了 VOV / 中心 / TALL ...): 不按按钮外框扩, 按它的**全部**洞(字 + 字上的拼音)扩
    #   (单个洞各判各的会把按钮字切掉: 行带比按钮字低时, 字的洞过不了 VOV —— 10-23 看进度 / 去查看 被切了底)
    btn_pass = [int(i) for i in btn if (~kept[i] & xin[i] & ~pin[i] & shape_ok[i] & vov[i] & ctr[i])]
    #   按钮上有一个洞(字)单独算进了这一段, 这个按钮上的字就整个算(字是一个整体, 10-23 通知TA 只扩到一半)
    btn_pass = sorted(set(btn_pass) | set(int(p_) for p_ in parent[sel & (parent >= 0)]))
    if len(btn):
        touch_g = (parent >= 0) & (Hc >= FRAG_MIN_H * row_h) & ((np.minimum(Y + Hc, lb) - np.maximum(Y, la)) > 0) & xin
        btn_pass = sorted(set(btn_pass) | set(int(p_) for p_ in parent[touch_g] if xin[int(p_)]))

    def drop_containers(cand):
        if not len(cand) or not (Hc[cand] > CONTAINER_H * row_h).any():
            return cand
        Xr, Yr, Wr, Hr, Ar = (st[:, k].astype(np.int64) for k in range(5))
        ir = np.arange(nreal)
        nearm = (ir > 0) & (Ar >= 6) & (np.minimum(Xr + Wr, b) > np.maximum(Xr, a)) & (Yr < lb + 3 * row_h) & (Yr + Hr > la - 3 * row_h)
        near = np.nonzero(nearm)[0]
        keep = []
        for i in cand:
            if i < nreal and Hc[i] > CONTAINER_H * row_h:
                inside = ((Xr[near] > X[i]) & (Yr[near] > Y[i]) & (Xr[near] + Wr[near] < X[i] + Wc[i]) & (Yr[near] + Hr[near] < Y[i] + Hc[i])
                          & (near != i)).sum()
                if inside >= CONTAINER_N:
                    continue            # 外圈: 按钮框 / 图标 / 头像, 里面装着别的块 —— 不按它扩(10-23: 免费领 92 -> 157 像素, 字被缩小)
            keep.append(i)
        return np.array(keep, dtype=np.int64)

    cand = drop_containers(np.nonzero(sel)[0])
    if btn_pass:
        cand = np.union1d(cand, np.nonzero(base & xin & np.isin(parent, btn_pass))[0]).astype(np.int64)
    mode = "own"
    if len(cand):
        ov_self = (np.minimum(Y[cand] + Hc[cand], lb) - np.maximum(Y[cand], la)) / np.maximum(1, Hc[cand])
        ov_other = np.zeros(len(cand))
        for r in L["rows"]:
            if r["la"] == la and r["lb"] == lb:
                continue
            if r["lb"] <= la - 3 * row_h or r["la"] >= lb + 3 * row_h:
                continue
            ov_other = np.maximum(ov_other, (np.minimum(Y[cand] + Hc[cand], r["lb"]) - np.maximum(Y[cand], r["la"])) / np.maximum(1, Hc[cand]))
        if (ov_other >= ov_self).all() and (ov_other > ov_self).any():
            cand = cand[:0]             # 全是别的行的字(两行的带都整个罩住的碎块算两边都行) -> 下面按 FRAG 切
    if not len(cand):
        # P3 FRAG: 带里只有伸进来的一截。碰到 label 带的字块(够高、不是拼音 / 竖线 / 横线 / 外圈), 加上碰到带的按钮的全部洞
        touch = (np.minimum(Y + Hc, lb) - np.maximum(Y, la)) > 0
        f = common & shape_ok & touch & (Hc >= FRAG_MIN_H * row_h) & (parent < 0)
        cand = drop_containers(np.nonzero(f)[0])
        if len(cand) < FRAG_MIN_N:
            cand = cand[:0]             # 只有一块: 多半是图标 / 头像 / 圆圈, 不为它扩(10-23 核对图: 导航图标、闪购圆标)
        btn_touch = [int(i) for i in np.nonzero(is_btn[:nreal] & touch[:nreal] & xin[:nreal])[0]]
        if btn_touch:
            cand = np.union1d(cand, np.nonzero(base & xin & np.isin(parent, btn_touch))[0]).astype(np.int64)
        mode = "frag"
    if not len(cand):
        return None
    top, bot = int(Y[cand].min()), int((Y[cand] + Hc[cand]).max())
    if with_mode:
        L["_own"] = (np.stack([X[cand], Y[cand], Wc[cand], Hc[cand]], 1), set(int(i) for i in cand if i < nreal) | set(int(p_) for p_ in parent[cand] if p_ >= 0))
        L["_own_real"] = set(int(i) for i in cand if i < nreal and not is_btn[i])
        hb = cand[parent[cand] >= 0]
        L["_btn_ext"] = (int(Y[hb].min()), int((Y[hb] + Hc[hb]).max())) if len(hb) else None
        return top, bot, mode
    return top, bot


def neighbours(L: dict, la: int, lb: int) -> tuple:
    """这一行上面那行的 label 底、下面那行的 input 顶(含它的拼音)、下面最近的拼音带顶 —— 新边不许越过, 免得把邻行切进来。"""
    rows = sorted(L["rows"], key=lambda r: r["la"])
    up, down = 0, L["H"]
    for r in rows:
        if r["la"] < la:
            up = max(up, r["lb"] + 1)
        elif r["la"] > la:
            down = min(down, r["ia"] - 1)
    # ★ 这一行下面的拼音带一定是**下一行**的(自己的拼音在上面): 往下扩不许进去。
    #   下一行字小、没被聚成行时(芭芭农场 限时翻倍、按钮上的 免费领), 只有它的拼音带拦得住
    for a, b in L["bands"]:
        if a >= lb - 2:                 # 带在这一行底下(行里面的小字拼音不算)
            down = min(down, a - 1)
    return up, down


def band_below(L: dict, lb: int) -> int:
    """这一行下面最近的拼音带顶(下一行的拼音), 没有就是页底。FRAG 往下扩只拿它卡。"""
    down = L["H"]
    for a, b in L["bands"]:
        if a >= lb - 2:
            down = min(down, a - 1)
    return down


def new_box(L: dict, la: int, lb: int, ia: int, ib: int, a: int, b: int) -> tuple:
    """这一段新的 input 上下边 (ia2, ib2): 原来的边和自己字块的外包, 只扩不收;
    每边最多扩 CAP 倍行高, 而且不越过上下邻行(2026-10-23 核对图: 图标、按钮底色跨两行, 会把下一行带进来)。
    补丁 P3: FRAG 段(带里只有别的行伸进来的一截)不按邻行卡 —— 那些字就是邻行的, 要切全; 只按 CAP 卡。"""
    ext = seg_extent(L, la, lb, a, b, with_mode=True)
    if ext is None:
        return ia, ib
    row_h = max(1, lb - la)
    top, bot, mode = ext
    if mode == "frag":
        # FRAG 不按邻行、也不按下面的拼音带卡: 要补全的字就在那儿(10-23 去看看 的字和下一行的拼音带同高), 只按 CAP
        ia2 = max(0, int(ia - CAP * row_h), min(ia, top - PAD_Y))
        ib2 = min(L["H"], int(ib + CAP * row_h), max(ib, bot + PAD_Y))
        return ia2, ib2
    up, down = neighbours(L, la, lb)
    ia2 = max(0, int(ia - CAP * row_h), min(up, ia), min(ia, top - PAD_Y))
    ib2 = min(L["H"], int(ib + CAP * row_h), max(down, ib), max(ib, bot + PAD_Y))
    be = L.get("_btn_ext")
    if be:
        # 按钮上的字被按钮圈住, 外面的东西进不来: 不按邻行卡, 只按 CAP(10-23: 去看看 的字和下一行的拼音带同高, 被 down 卡掉一半)
        ia2 = min(ia2, max(0, int(ia - CAP * row_h), be[0] - PAD_Y))
        ib2 = max(ib2, min(L["H"], int(ib + CAP * row_h), be[1] + PAD_Y))
    if top - PAD_Y < ia2 or bot + PAD_Y > ib2:
        own_ids = L["_own"][1]
        real = [i for i in own_ids if i in L.get("_own_real", own_ids)]
        st = L["st"]
        refs = [i for i in real if st[i][1] >= ia2 and st[i][1] + st[i][3] <= ib2 and st[i][4] >= P6_REF_A]
        if refs:
            rc = [_color(L, i) for i in refs]
            def same(i):
                c = _color(L, i)
                return any(np.abs(c - r_).max() <= P6_COLOR for r_ in rc)
            up_c = [i for i in real if st[i][1] < ia2 and same(i)]
            if up_c:
                want_t = max(0, int(ia - CAP * row_h), min(ia, min(int(st[i][1]) for i in up_c) - PAD_Y))
                if want_t < ia2 and _clean(L, a, b, want_t, ia2, own_ids, row_h):
                    ia2 = want_t
            dn_c = [i for i in real if st[i][1] + st[i][3] > ib2 and same(i)]
            if dn_c:
                want_b = min(L["H"], int(ib + CAP * row_h), max(ib, max(int(st[i][1] + st[i][3]) for i in dn_c) + PAD_Y))
                if want_b > ib2 and _clean(L, a, b, ib2, want_b, own_ids, row_h):
                    ib2 = want_b
    return ia2, ib2


def _color(L: dict, i: int) -> np.ndarray:
    """第 i 块墨的平均颜色(BGR)。"""
    cache = L.setdefault("_col", {})
    if i not in cache:
        x, y, w, h = (int(v) for v in L["st"][i][:4])
        m = L["lab"][y:y + h, x:x + w] == i
        cache[i] = L["img"][y:y + h, x:x + w][m].reshape(-1, 3).mean(0) if m.any() else np.zeros(3)
    return cache[i]


def _clean(L: dict, a: int, b: int, z0: int, z1: int, own_ids: set, row_h: int) -> bool:
    """[z0, z1) x [a, b) 这块地方除了这一段自己的字(和装着它们的按钮)以外没有别的墨(大底板 / 细横线不算)。"""
    st = L["st"]
    X, Y, Wc, Hc, A = (st[:, k].astype(np.int64) for k in range(5))
    idx = np.arange(len(st))
    m = (idx > 0) & (A >= 6) & (X < b) & (X + Wc > a) & (Y < z1) & (Y + Hc > z0)
    m &= ~((Hc > TALL * row_h) | ((Wc > 3 * row_h) & (Hc < 0.25 * row_h)))
    for i in np.nonzero(m)[0]:
        if int(i) not in own_ids:
            return False
    return True


def demo(a) -> None:
    """本地看效果: 每张图按现有算法切行切段, 新旧上下边差 >= 3 像素的段出前后对比图。"""
    import random
    from PIL import Image, ImageDraw
    from make_check_sheet import _cjk_font
    from split_segments import PAD, segments
    import page_type as pt
    font = _cjk_font(18)
    paths = [p for p, _t in pt.collect(a.src)]
    random.Random(a.seed).shuffle(paths)
    paths = paths[: a.n]
    pairs, n_seg, n_changed = [], 0, 0
    for p in paths:
        img = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        L = layout(img)
        for r in L["rows"]:
            lab = img[r["la"]:r["lb"], :]
            for x0, x1 in segments(lab):
                sa, sb = max(0, x0 - PAD), min(L["W"], x1 + PAD + 1)
                n_seg += 1
                ia2, ib2 = new_box(L, r["la"], r["lb"], r["ia"], r["ib"], sa, sb)
                if (r["ia"] - ia2) + (ib2 - r["ib"]) >= 3:
                    n_changed += 1
                    pairs.append((Path(p).name, r["row"], img[r["ia"]:r["ib"], sa:sb], img[ia2:ib2, sa:sb],
                                  r["ia"] - ia2, ib2 - r["ib"]))
    print(f"  图 {len(paths)}  段 {n_seg:,}  上下边变了的 {n_changed:,} ({n_changed / max(1, n_seg):.1%})", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    random.Random(1).shuffle(pairs)
    per = 30
    for k in range(0, min(len(pairs), a.sheets * per), per):
        part = pairs[k:k + per]
        rowsimg = []
        for name, row, old, new, up, down in part:
            def fit(im):
                s = min(1.0, 520 / max(1, im.shape[1]))
                return cv2.resize(im, (max(1, int(im.shape[1] * s)), max(1, int(im.shape[0] * s))))
            o, n_ = fit(old), fit(new)
            h = max(o.shape[0], n_.shape[0]) + 6
            can = np.full((h + 22, 1100, 3), 255, np.uint8)
            can[22:22 + o.shape[0], 0:o.shape[1]] = o
            can[22:22 + n_.shape[0], 560:560 + n_.shape[1]] = n_
            pil = Image.fromarray(can[:, :, ::-1])
            ImageDraw.Draw(pil).text((4, 1), f"{name[:40]} r{row}   旧(左)  新(右) 上扩 {up} 下扩 {down}", font=font, fill=(200, 0, 0))
            rowsimg.append(np.array(pil)[:, :, ::-1])
        sheet = np.vstack(rowsimg)
        cv2.imencode(".png", sheet)[1].tofile(str(a.out / f"recrop_{k // per:02d}.png"))
    print(f"  核对图 -> {a.out}", flush=True)


def regold(a) -> None:
    """把金标准的段从原图重切一遍(新切法), 出一份同样格式的答案目录(img + 三个 csv 原样拷), 好拿同一套答案比新旧切图。
    每段先按**旧**切法从原图切一遍, 和答案目录里存的裁图逐像素比 —— 一样才说明行、段、原图都对得上。"""
    import csv
    import shutil
    seg_cache, pl_cache = {}, {}
    # ★ _segments.csv 里记的原图路径是切图那天的(比如 D 盘的 alipay-ai-data, 后来整个搬到了 E 盘 —— 2026-10-23 服务器上就是这样报错的)。
    #   找不到时: 按文件名查 page_type.csv(服务器上新跑的, 路径是现在的) -> 再试换盘符 -> 都没有才算找不到(用旧图, 记下来)
    index = {}
    for fn in a.page_index:
        if Path(fn).exists():
            with open(fn, encoding="utf-8-sig", newline="") as f:
                for r in csv.DictReader(f):
                    index.setdefault(Path(r["path"]).name.lower(), r["path"])
    how = {"原路径": 0, "按文件名找到": 0, "换盘符找到": 0}

    def resolve(src: str):
        if Path(src).exists():
            how["原路径"] += 1
            return src
        q = index.get(Path(src).name.lower())
        if q and Path(q).exists():
            how["按文件名找到"] += 1
            return q
        for drv in "EDCF":
            q = drv + src[1:]
            if len(src) > 2 and src[1] == ":" and Path(q).exists():
                how["换盘符找到"] += 1
                return q
        return None

    def segs_of(d: Path) -> dict:
        if d not in seg_cache:
            with (d / "_segments.csv").open(encoding="utf-8-sig", newline="") as f:
                seg_cache[d] = {r["file"]: r for r in csv.DictReader(f)}
        return seg_cache[d]

    def rows_of(d: Path) -> dict:
        if d not in pl_cache:
            with (d / "_pairs_labeled.csv").open(encoding="utf-8-sig", newline="") as f:
                pl_cache[d] = {r["file"]: r for r in csv.DictReader(f)}
        return pl_cache[d]

    for g in a.gold:
        out = a.out_root / f"{g.name}{a.suffix}"
        if out.exists():
            print(f"  ★ {out} 已经有了, 不覆盖", flush=True)
            sys.exit(2)
        (out / "img").mkdir(parents=True)
        for fn in ("_gold_candidates.csv", "_gold.csv", "_relevance.csv"):
            if (g / fn).exists():
                shutil.copyfile(g / fn, out / fn)
        with (g / "_gold_candidates.csv").open(encoding="utf-8-sig", newline="") as f:
            cand = list(csv.DictReader(f))
        same = differ = missing = changed = 0
        log = {}
        jobs = {}                                     # 原图 -> [(答案行, 段行, 行行)]: 一张原图只解码、只算一次字块
        for r in cand:
            sp = Path(r["src_path"])
            d = sp.parent.parent
            if not (d / "_segments.csv").exists():
                d = a.pairs_root / d.name             # 路径盘符 / 根目录不一样时按目录名找
            try:
                s_ = segs_of(d)[sp.name]
                row = rows_of(d)[s_["row_file"]]
            except (KeyError, FileNotFoundError):
                missing += 1
                shutil.copyfile(g / "img" / r["file"], out / "img" / r["file"])
                log[r["id"]] = (r["id"], r["file"], "段或行找不到, 用旧图", 0, 0)
                continue
            jobs.setdefault(s_["source"], []).append((r, s_, row))
        for src0, items in jobs.items():
            src = resolve(src0)
            img = None
            if src:
                try:
                    img = cv2.imdecode(np.fromfile(src, np.uint8), cv2.IMREAD_COLOR)
                except Exception:  # noqa: BLE001
                    img = None
            L = layout(img) if img is not None else None
            for r, s_, row in items:
                if img is None:
                    missing += 1
                    shutil.copyfile(g / "img" / r["file"], out / "img" / r["file"])
                    log[r["id"]] = (r["id"], r["file"], "原图找不到或读不出, 用旧图", 0, 0)
                    continue
                la, lb = int(row["label_y0"]), int(row["label_y1"])
                ia, ib = int(row["input_y0"]), int(row["input_y1"])
                x0, x1 = int(s_["x0"]), int(s_["x1"])
                old = img[ia:ib, x0:x1]
                stored = cv2.imdecode(np.fromfile(str(g / "img" / r["file"]), np.uint8), cv2.IMREAD_COLOR)
                if stored is not None and stored.shape == old.shape and np.array_equal(stored, old):
                    same += 1
                else:
                    differ += 1
                ia2, ib2 = new_box(L, la, lb, ia, ib, x0, x1)
                if max(ia - ia2, ib2 - ib) < a.min_ext * max(1, lb - la):
                    ia2, ib2 = ia, ib                 # 只差几像素的不动(B 方案: 只改真被切掉一截的段)
                cv2.imencode(".png", img[ia2:ib2, x0:x1])[1].tofile(str(out / "img" / r["file"]))
                if (ia2, ib2) != (ia, ib):
                    changed += 1
                log[r["id"]] = (r["id"], r["file"], "ok" if (stored is not None and stored.shape == old.shape and np.array_equal(stored, old)) else "旧切法对不上",
                                ia - ia2, ib2 - ib)
        log = [log[r["id"]] for r in cand if r["id"] in log]
        with (out / "_recrop_log.csv").open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "file", "status", "up", "down"])
            w.writerows(log)
        print(f"  {g.name}: 原图 {len(jobs)} 张  " + "  ".join(f"{k} {v}" for k, v in how.items()), flush=True)
        for k in how:
            how[k] = 0
        print(f"  {g.name}: 段 {len(cand)}  旧切法和存的裁图逐像素一样 {same}  不一样 {differ}  找不到 {missing}  新切法上下边变了 {changed}"
              f"  -> {out}", flush=True)
        if differ:
            print(f"  ★ {g.name}: 有 {differ} 段旧切法对不上存的裁图(行 / 段 / 原图不一致), 先看看", flush=True)
        # 切法指纹: 和训练图(seg_input_rc)的对得上, 新旧切法的比较才是同一版切的
        (out / "_recut_version.txt").write_text(recut_version() + "\n", encoding="utf-8")
        # 给服务器脚本读的(纯 ASCII)
        print(f"REGOLD_STATS {g.name} total={len(cand)} same={same} differ={differ} missing={missing} changed={changed}", flush=True)
    print("REGOLD_OK", flush=True)


def load_index(files: list) -> dict:
    """文件名(小写) -> 现在的路径: 从 page_type.csv 的 path 列来(原图搬过家时按文件名找)。"""
    import csv
    index = {}
    for fn in files:
        if Path(fn).exists():
            with open(fn, encoding="utf-8-sig", newline="") as f:
                for r in csv.DictReader(f):
                    index.setdefault(Path(r["path"]).name.lower(), r["path"])
    return index


def resolve_path(src: str, index: dict) -> tuple:
    """(现在的路径或 None, 怎么找到的)。原路径 -> 按文件名查 index -> 换盘符。"""
    if Path(src).exists():
        return src, "orig"
    q = index.get(Path(src).name.lower())
    if q and Path(q).exists():
        return q, "index"
    if len(src) > 2 and src[1] == ":":
        for drv in "EDCF":
            q = drv + src[1:]
            if Path(q).exists():
                return q, "drive"
    return None, "missing"


def _num_consts(mod) -> str:
    """一个模块里所有大写的数字常数(按名字排好): 切法函数读的 OVERLAP / DIFF_THRESHOLD / MIN_RUN ... 一个不漏。
    只取数字 —— 集合之类的 repr 顺序每个进程不一样, 放进来指纹就不稳了。"""
    return " ".join(f"{k}={v!r}" for k, v in sorted(vars(mod).items())
                    if k.isupper() and isinstance(v, (int, float)) and not isinstance(v, bool))


def recut_version() -> str:
    """新切法的指纹: 三个模块的数字常数 + 切行切段那几个函数的源码。代码一改指纹就变 —— 接着跑时不会拿上一版切的图充数。
    金标准重切(regold)也记同一个指纹, 服务器脚本核对训练图和金标准图是同一版切的。"""
    import hashlib
    import inspect
    import erase_pinyin
    import make_pinyin_pairs
    fns = (layout, _encloses, seg_extent, neighbours, new_box, _button_holes, band_below, _color, _clean,
           make_pinyin_pairs._pinyin_bands, make_pinyin_pairs._rows,
           erase_pinyin.annotation_labels, erase_pinyin.text_mask, erase_pinyin.local_background)
    consts = "|".join(_num_consts(m) for m in (sys.modules[__name__], make_pinyin_pairs, erase_pinyin))
    src = consts + "".join(inspect.getsource(f).replace("\r\n", "\n") for f in fns)
    return hashlib.sha1(src.encode("utf-8")).hexdigest()[:12]


def _done(p: Path) -> bool:
    """切好了: 文件在而且不是空的(写的时候先写 .tmp 再改名, 半截文件不会顶着正式名字)。"""
    try:
        return p.stat().st_size > 0
    except OSError:
        return False


_INDEX = {}
_RC = "seg_input_rc"


def _recut_init(index: dict, rc_name: str = "seg_input_rc") -> None:
    global _INDEX, _RC
    cv2.setNumThreads(1)
    _INDEX = index
    _RC = rc_name           # ★ 子进程拿不到主进程改的全局, 靠 initializer 传(Windows 是 spawn)


def _recut_page(job) -> list:
    """一张原图: 解码一次、算一次字块, 把它的每一段重切到 seg_input_rc。
    每段先按旧切法切, 和 seg_input 里存的逐像素比; 对不上(行 / 段 / 原图不一致)或原图找不到的**什么都不写**,
    relist 时这些段自然留旧图、算进覆盖率; 再跑一遍会重试它们(2026-10-23 复核: 以前拷旧图进 seg_input_rc,
    再跑时被当成"切过了", 闸门和覆盖率都看不出来)。
    已经切好的(seg_input_rc 里有、不是空的)跳过 —— 停了再跑接着做; 指纹不对的整个目录在 recut() 里就拦下了。"""
    import os
    src, items = job
    out = []
    todo = []
    for it in items:
        if _done(Path(it["dir"]) / _RC / it["file"]):
            out.append((it["dir"], it["file"], "done_before", 0, 0))
        else:
            todo.append(it)
    if not todo:
        return out
    path, how = resolve_path(src, _INDEX)
    img = None
    if path:
        try:
            img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
        except Exception:  # noqa: BLE001
            img = None
    L = layout(img) if img is not None else None
    for it in todo:
        d = Path(it["dir"])
        dst = d / _RC / it["file"]
        stored_p = d / "seg_input" / it["file"]
        if img is None:
            out.append((it["dir"], it["file"], "page_missing", 0, 0))
            continue
        la, lb, ia, ib, x0, x1 = (int(it[k]) for k in ("la", "lb", "ia", "ib", "x0", "x1"))
        stored = cv2.imdecode(np.fromfile(str(stored_p), np.uint8), cv2.IMREAD_COLOR) if stored_p.exists() else None
        old = img[ia:ib, x0:x1]
        if stored is None or stored.shape != old.shape or not np.array_equal(stored, old):
            out.append((it["dir"], it["file"], "old_mismatch", 0, 0))
            continue
        ia2, ib2 = new_box(L, la, lb, ia, ib, x0, x1)
        tmp = dst.with_name(dst.name + ".tmp")
        cv2.imencode(".png", img[ia2:ib2, x0:x1])[1].tofile(str(tmp))
        os.replace(tmp, dst)
        out.append((it["dir"], it["file"], "ok_" + how, ia - ia2, ib2 - ib))
    return out


def recut(a) -> None:
    """训练集的段全部按新切法重切到 <目录>\\seg_input_rc\\ (旧的 seg_input 一张不动)。只切在训练里的段(bad != 1 而且有字)。
    按原图分组并行; 每段先核对旧切法 == 存的 seg_input(对不上的不写); 出 _recut_log.csv(追加, 每次带时间)和纯 ASCII 的 RECUT_STATS。
    seg_input_rc 里记着切法指纹(_recut_version.txt): 和现在的代码对不上就整个停下(RECUT_VERSION_MISMATCH), 不混两版的图。"""
    import csv
    import time
    from collections import Counter
    from concurrent.futures import ProcessPoolExecutor
    ver = recut_version()
    print(f"  切法指纹 {ver}", flush=True)
    for d in a.pairs:                                  # 先全部核对一遍再动手, 不切到一半才发现
        rc_dir = d / a.rc_name
        vf = rc_dir / "_recut_version.txt"
        if rc_dir.exists() and any(q.suffix.lower() == ".png" for q in rc_dir.iterdir()):
            have = vf.read_text(encoding="utf-8").strip() if vf.exists() else "none"
            if have != ver:
                print(f"RECUT_VERSION_MISMATCH {d.name} have={have} code={ver}", flush=True)
                sys.exit(3)
    index = load_index(a.page_index)
    print(f"  按文件名找原图的表 {len(index):,} 张", flush=True)
    run = time.strftime("%Y-%m-%d %H:%M:%S")
    for d in a.pairs:
        with (d / "_pairs_labeled.csv").open(encoding="utf-8-sig", newline="") as f:
            rows = {r["file"]: r for r in csv.DictReader(f)}
        jobs = {}
        n_skip = n_row_missing = 0
        with (d / "_segments.csv").open(encoding="utf-8-sig", newline="") as f:
            for s in csv.DictReader(f):
                if s.get("bad") == "1" or not (s.get("text") or "").strip():
                    n_skip += 1
                    continue
                r = rows.get(s["row_file"])
                if r is None:
                    n_row_missing += 1                 # 在训练里但找不到行: 单独记, 给闸门看(留旧图)
                    continue
                jobs.setdefault(s["source"], []).append({
                    "dir": str(d), "file": s["file"], "x0": s["x0"], "x1": s["x1"],
                    "la": r["label_y0"], "lb": r["label_y1"], "ia": r["input_y0"], "ib": r["input_y1"]})
        rc_dir = d / a.rc_name
        rc_dir.mkdir(exist_ok=True)
        (rc_dir / "_recut_version.txt").write_text(ver + "\n", encoding="utf-8")
        for t in rc_dir.glob("*.tmp"):                 # 上次停在写到一半的
            t.unlink()
        n_seg = sum(len(v) for v in jobs.values())
        print(f"  {d.name}: 原图 {len(jobs):,} 张  要切 {n_seg:,} 段  (不在训练里 跳过 {n_skip:,}, 在训练里但找不到行 {n_row_missing:,})", flush=True)
        cnt, log = Counter(), []
        done = 0
        with ProcessPoolExecutor(a.workers, initializer=_recut_init, initargs=(index, a.rc_name)) as ex:
            for res in ex.map(_recut_page, list(jobs.items()), chunksize=4):
                for dd, fn, st_, up, down in res:
                    cnt[st_.split("_")[0] if st_.startswith("ok") else st_] += 1
                    if st_.startswith("ok") and (up or down):
                        cnt["changed"] += 1
                    log.append((run, fn, st_, up, down))
                done += 1
                if done % 2000 == 0:
                    print(f"    {done:,}/{len(jobs):,} 张图", flush=True)
        lp = d / "_recut_log.csv"
        new = not lp.exists()
        with lp.open("a", encoding="utf-8-sig" if new else "utf-8", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["run", "file", "status", "up", "down"])
            w.writerows(log)
        print(f"RECUT_STATS {d.name} total={n_seg} ok={cnt['ok']} changed={cnt['changed']} old_mismatch={cnt['old_mismatch']} "
              f"page_missing={cnt['page_missing']} done_before={cnt['done_before']} row_missing={n_row_missing}", flush=True)
    print("RECUT_OK", flush=True)


def relist(a) -> None:
    """训练集清单换成重切的图: 路径里 \\seg_input\\ 换成 \\seg_input_rc\\(新图在才换, 不在就留旧图并记数), 字表原样拷。标签一字不动。
    换成新图的不到 --min-rc 就什么都不写(RELIST_LOW): 目录在 = 覆盖率过了关。train.txt 最后写(先 .tmp 再改名), 它在 = 写完了。
    ★ val.txt 也换成新图: run 目录里 train_log 的验证集数字和旧模型(旧图验证)不能直接比, 新旧模型只在金标准上比。"""
    import os
    import shutil
    if (a.out / "train.txt").exists():
        print(f"  ★ {a.out} 已经有了, 不覆盖", flush=True)
        sys.exit(2)
    tot = rc = 0
    lists = {}
    for name in ("train.txt", "val.txt"):
        lines_out = []
        for ln in (a.src / name).read_text(encoding="utf-8").splitlines():
            if "\t" not in ln:
                continue
            p, txt = ln.split("\t", 1)
            q = p.replace("\\seg_input\\", f"\\{a.rc_name}\\").replace("/seg_input/", f"/{a.rc_name}/")
            tot += 1
            if q != p and _done(Path(q)):
                rc += 1
                p = q
            lines_out.append(f"{p}\t{txt}")
        lists[name] = lines_out
    cov = rc / max(1, tot)
    print(f"  {a.src.name} -> {a.out.name}: {tot:,} 段, 换成重切图的 {rc:,} ({cov:.2%})", flush=True)
    print(f"RELIST_STATS total={tot} rc={rc}", flush=True)
    if cov < a.min_rc:
        print(f"RELIST_LOW cov={cov:.4f} min={a.min_rc} - nothing written", flush=True)
        sys.exit(4)
    a.out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(a.src / "charset.txt", a.out / "charset.txt")
    (a.out / "val.txt").write_text("\n".join(lists["val.txt"]) + "\n", encoding="utf-8")
    tmp = a.out / "train.txt.tmp"
    tmp.write_text("\n".join(lists["train.txt"]) + "\n", encoding="utf-8")
    os.replace(tmp, a.out / "train.txt")
    print("RELIST_OK", flush=True)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="切图修正: 每段按自己的字定上下边")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo")
    d.add_argument("--src", nargs="+", required=True)
    d.add_argument("--out", type=Path, required=True)
    d.add_argument("--n", type=int, default=40)
    d.add_argument("--seed", type=int, default=1)
    d.add_argument("--sheets", type=int, default=3)
    g = sub.add_parser("regold")
    g.add_argument("--gold", type=Path, nargs="+", required=True, help="答案目录(gold gold2): img + _gold_candidates.csv ...")
    g.add_argument("--out-root", type=Path, required=True, help="新答案目录放哪(出 <名字>_rc)")
    g.add_argument("--pairs-root", type=Path, default=Path(r"E:\alipay-ai-data"), help="src_path 里的目录找不到时到这里按目录名找")
    g.add_argument("--min-ext", type=float, default=0.0, help="新边比旧边多出不到这么多倍行高就不动(0 = 全按新切法)")
    g.add_argument("--suffix", default="_rc", help="新答案目录名 = 原名 + 这个")
    g.add_argument("--page-index", nargs="*", default=[r"E:\_move_logs\pagetype_pinyin_v4\page_type.csv", r"E:\_move_logs\pagetype_census_v4\page_type.csv"],
                   help="原图搬过家时按文件名找现在的路径(page_type.csv 的 path 列)")
    c = sub.add_parser("recut")
    c.add_argument("--pairs", type=Path, nargs="+", required=True, help="配对目录(里面有 _segments.csv / _pairs_labeled.csv / seg_input)")
    c.add_argument("--workers", type=int, default=8)
    c.add_argument("--page-index", nargs="*", default=[r"E:\_move_logs\pagetype_pinyin_v4\page_type.csv", r"E:\_move_logs\pagetype_census_v4\page_type.csv"])
    c.add_argument("--rc-name", default="seg_input_rc", help="重切图放到 <目录>\\这个名字(v3 用 seg_input_rc3, 不和 v2 的混)")
    rl = sub.add_parser("relist")
    rl.add_argument("--src", type=Path, required=True, help="原训练集目录(train.txt / val.txt / charset.txt)")
    rl.add_argument("--out", type=Path, required=True)
    rl.add_argument("--min-rc", type=float, default=0.97, help="换成重切图的段至少这么多, 不够就什么都不写")
    rl.add_argument("--rc-name", default="seg_input_rc", help="清单里 \\seg_input\\ 换成 \\这个名字\\")
    sub.add_parser("version", help="打印现在代码的切法指纹(RECUT_VERSION xxx), 给服务器脚本核对")
    a = ap.parse_args()
    if a.cmd == "demo":
        demo(a)
    elif a.cmd == "regold":
        regold(a)
    elif a.cmd == "recut":
        recut(a)
    elif a.cmd == "relist":
        relist(a)
    elif a.cmd == "version":
        print(f"RECUT_VERSION {recut_version()}", flush=True)


if __name__ == "__main__":
    main()
