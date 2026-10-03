r"""账单详情页"处理进度"时间轴的位置检查(经理说的 112) —— Python 参考实现 + 批量扫描。

现象
----
真图: 时间轴的蓝色圆点比上面取值列的文字(比如"招商银行")往右缩进几个像素。
假图: 圆点左缘和文字左缘对齐, 甚至还往左凸出来一点。
经理的办法是从取值列文字左缘往下拉一条竖线, 碰到蓝色就是假图; 这里就是把这条线量准。
C# 版是 demo/TimelineCheck.cs, 两边逐步对应, 改一边要同步改另一边。

量法
----
1. 找圆点: 蓝色(OpenCV HSV 的 H 98~118, S>=120, V>=150), 圆形, 里面有白色对勾(按亮度找, 见 find_circles),
   在页面宽度 22%~42% 之间; 同一列竖着排开 >=2 个才算时间轴。
2. 取值列左缘: 第一个圆点上方 6D 以内(D = 圆点直径), 圆点左右 [-1.5D, +4D] 窗口里的深色字(灰度 < 150),
   紧贴在上面的拼音并进同一行(见 _text_rows),
   取离时间轴最近的两行, 各取最左的墨点, 再取两者中更靠左的那个。
   ★ 为什么取两行里更靠左的: 每个字左边留白不一样("中"比"账"多两三个像素),
     只看一行会把字形差当成位置差。两行取靠左, 更接近文字真正的起点。
   ★ 窗口不从页面左边开始: 不然会量到返回箭头、标签栏(早先一版有 +6D 的离谱值就是这么来的)。
3. 偏移 = (圆点左缘 - 取值列左缘) / D。 <= -0.01 判可疑(圆点比文字还靠左, 经理那条线一定碰到蓝色)。
   苹果机型原尺寸截图(分辨率在 IOS_RES 里)判得更严: < +0.07 就判可疑, 见下面服务器数据。
4. 判不了(CannotDetermine)的几种: 圆点直径 < 28 像素; 紧挨着上方的一行从圆点右边开始
   (那是上一步的步骤文字, 说明第一个圆点没找到); 偏移 < -0.30(假图只差几个像素,
   差出这么多是版面没认对, 比如时间轴顶着导航栏, 量到了标题); 偏移 > +0.60(真图最多 +0.28,
   再大是标签栏的字伸进了窗口)。

本地 13,815 张实测(其中 489 张有时间轴):
    正常字体真图       +0.19 ~ +0.24   (各种分辨率都在这个范围)
    拼音手写字体真图   +0.038 起(2 像素)  (这种字体的字几乎不留左边白, 是最贴近的一类)
    假图 112 false     -0.068
    早先负号检查挑出的 6 张 1206 宽苹果图   -0.036(都带"处理进度"标签, 是另一种做法的假图)

服务器 2026-09-01~16 新下载的图, 每 8 张取 1 张, 101,380 张里 28,165 张有时间轴(28,028 张量得了):
    每个常见分辨率都是两团, 中间是空的。苹果 12,990 张: 真图最低 +0.15, 可疑那团 -0.044 ~ 0.0。判可疑 60 张(0.21%)。
    可疑的逐张看过: 标签写成"进度处理"、订单号不是 2026MMDD20004001110 开头、
    客服电话断行("客服9 / 5533")、同一天同一分辨率扎堆 —— 都是假图的特征。
    唯一误判是一张 581 宽、手写字体、被缩过的图(圆点直径 24), 所以直径 < 28 不判。

缩放和重新压缩(本地 495 张时间轴图 x 20 种缩放/JPEG 压缩, 真图 9,640 份): 误判 0, 时间轴一张不丢,
假图 140/140 判可疑。早先的版本在这组上误判 23、丢时间轴 470 —— 原因和改法见 find_circles 和 _text_rows。
THRESHOLD 改成 0(经理那条线碰到圆点边就算假): 服务器样本多抓 5 张, 都是假图;
但缩小过的手写字体真图(原图只缩进 1~3 像素的)压力测试里约 0.5% 会被误判, 所以默认留在 -0.01。

用法
----
    python -u training/timeline_scan.py --out E:/x/tl.csv --sheets E:/x/tl_sheets D:/download2/OtherImages
    --every 5   每 5 张取 1 张(按文件名排序后均匀取, 覆盖整个时间段)
    --workers 8 进程数(默认 CPU 数 - 1)
"""
from __future__ import annotations

import os
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")      # 必须在 import numpy 之前, 见 pick_pinyin.py

import argparse  # noqa: E402
import csv  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

cv2.setNumThreads(1)
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

THRESHOLD = -0.01        # 偏移 <= 这个值判可疑
MIN_DIAMETER = 28        # 圆点直径小于这个像素数不判: 图被缩小过, 一个像素就是 0.04D, 手写字体真图会掉下去
GUARD = -0.30            # 偏移 < 这个值: 版面没认对, 不判
GUARD_HIGH = 0.60        # 偏移 > 这个值也不判: 真图最多 +0.28, 再大是量到了别的东西(实测是标签栏的字伸进窗口)
STEP_TEXT = 0.6          # 从 圆点左缘 + 0.6D 往右才开始的行是步骤文字, 不是取值列
LABEL_INK = 0.1          # "处理进度"标签灰字像素 / D^2 >= 这个值算有标签

# 苹果机型原尺寸截图的分辨率。服务器 12,990 张这些分辨率的真图偏移最低 +0.15,
# 可疑那团最高 0.0, 中间空着, 所以阈值放在 +0.07(两边各 3 个多像素)。
# 只认完全相等的分辨率: 被缩放过的苹果图不在表里, 走通用阈值。
IOS_RES = {(1179, 2556), (1290, 2796), (1170, 2532), (1320, 2868), (1206, 2622), (1284, 2778),
           (1125, 2436), (1242, 2688), (828, 1792), (750, 1334), (1242, 2208)}
IOS_THRESHOLD = 0.07     # 苹果原尺寸截图: 偏移 < 这个值判可疑


def _runs(on):
    """布尔序列里连续为真的段, [(起, 止)] 闭区间。"""
    segs, i, n = [], 0, len(on)
    while i < n:
        if on[i]:
            a = i
            while i < n and on[i]:
                i += 1
            segs.append((a, i - 1))
        i += 1
    return segs


def find_circles(bgr):
    """带白色对勾的蓝色实心圆, [(x, y, w, h)] 整图坐标。"""
    H, W = bgr.shape[:2]
    x0, x1 = int(0.15 * W), int(0.55 * W)
    hsv = cv2.cvtColor(bgr[:, x0:x1], cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr[:, x0:x1], cv2.COLOR_BGR2GRAY)
    m = cv2.inRange(hsv, (98, 120, 150), (118, 255, 255))
    k = max(5, int(W * 0.012)) | 1
    # 先用横条开运算去掉圆点之间那根细竖线(它比 k 窄), 再闭运算把对勾的白缝补上
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1)))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, _lab, st, _cen = cv2.connectedComponentsWithStats(m, connectivity=8, ltype=cv2.CV_32S)
    out = []
    for i in range(1, n):
        x, y, w, h, a = (int(v) for v in st[i])
        if not (0.030 * W <= w <= 0.056 * W and 0.80 * h <= w <= 1.25 * h and a * 2 >= w * h):
            continue
        if not (0.22 * W <= x + x0 <= 0.42 * W):
            continue
        # 圆心一块里要有白色对勾: 亮(灰度 > 170)的像素 >= 4%。
        # ★ 看亮度, 不看饱和度: JPEG 的 4:2:0 色度抽样会把蓝色糊到细细的对勾上, 饱和度一下就上去了,
        #   压过一次的图第一个圆点常被漏掉; 亮度不受影响(对勾 ~255, 蓝底 ~105)。
        r = max(1, w // 2)
        inner = gray[y + h // 4: y + h // 4 + r, x + w // 4: x + w // 4 + r]
        if inner.size == 0:
            continue
        white = int((inner > 170).sum())
        if white * 25 < inner.shape[0] * inner.shape[1]:
            continue
        out.append((x + x0, y, w, h))
    return out


def timeline(cs):
    """同一列(左缘差 <= 0.15D, 直径差 <= 0.2D)、竖着间隔 2.2D~7D 排开的圆点, 取最长的一组。"""
    best = []
    for c in cs:
        grp = sorted([d for d in cs if abs(d[0] - c[0]) <= 0.15 * c[2] and abs(d[2] - c[2]) <= 0.2 * c[2]],
                     key=lambda d: (d[1], d[0]))
        gaps = [(b[1] - a[1]) / a[2] for a, b in zip(grp, grp[1:])]
        if gaps and all(2.2 <= g <= 7.0 for g in gaps) and len(grp) > len(best):
            best = grp
    return best if len(best) >= 2 else []


def _dark_grey(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    s = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 1]
    # 正文黑字(排除蓝色)。灰度门槛 150 而不是 110: 图被缩小后细笔画会变浅,
    # 110 会把字最左边那一竖丢掉, 文字左缘往右跑几个像素, 真图就被判成圆点凸出来了
    dark = (g < 150) & (s < 60)
    grey = (g >= 120) & (g <= 205) & (s < 40)         # 标签灰字
    return dark, grey


def _text_rows(runs, D):
    """文字行: 高 >= 0.4D 的段, 再把紧贴在它上面的矮段(拼音那一行)并进来。

    拼音字体的页面, 拼音在汉字正上方, 而且往左伸出去一点 —— 原尺寸时拼音和汉字连成一段,
    左缘就带上了拼音; 图一缩小, 拼音就断开成 3~6 像素高的一段, 不够 0.4D 被丢掉,
    左缘只剩汉字, 真图就被判成圆点凸出来。
    ★ 只往已经够高的行上并, 矮段自己永远不成行: 无条件把碎段拼起来, 会把灰色日期行的碎片
      拼成一个新行, 把真正的取值行挤出最近两行, 反而造出新的误判(实测 -38 像素)。
    每一步的间隙 <= max(1, round(0.2D)), 一共最多往上并 0.6D, 碰到另一个够高的行就停。
    """
    gap = max(1, round(0.2 * D))
    rows = []
    for i, (a, b) in enumerate(runs):
        if b - a + 1 < 0.4 * D:
            continue
        top = a
        for pa, pb in reversed(runs[:i]):
            if pb - pa + 1 >= 0.4 * D or top - pb - 1 > gap or a - pa > 0.6 * D:
                break
            top = pa
        rows.append((top, b))
    return rows


def measure_img(bgr, path=""):
    """没有时间轴返回 None; 否则返回一行结果(verdict 是 Ok / Suspicious / CannotDetermine)。"""
    H, W = bgr.shape[:2]
    if W < 300 or H < 300:
        return None
    cs = timeline(find_circles(bgr))
    if not cs:
        return None
    D = float(np.median([c[2] for c in cs]))
    cl = int(np.median([c[0] for c in cs]))
    c0 = cs[0]
    r = {"path": path, "W": W, "H": H, "n": len(cs), "D": D, "circle_left": cl, "circle_top": c0[1]}
    if D < MIN_DIAMETER:
        r.update(verdict="CannotDetermine", why=f"圆点直径小于 {MIN_DIAMETER} 像素, 图被缩小过",
                 ios=int((W, H) in IOS_RES), label=0)
        return r

    # 取值列: 第一个圆点上方的几行
    wx0, wx1 = max(0, int(cl - 1.5 * D)), min(W, int(cl + 4 * D))
    wy0, wy1 = max(0, int(c0[1] - 6 * D)), max(0, int(c0[1] - 0.3 * D))
    lefts = []
    if wy1 > wy0:
        dark, _ = _dark_grey(bgr[wy0:wy1, wx0:wx1])
        rows = _text_rows(_runs(dark.sum(1) >= 2), D)
        for a, b in reversed(rows):                   # 从下往上, 离时间轴最近的先
            cols = np.flatnonzero(dark[a:b + 1].any(0))
            if cols[0] <= 1:                          # 贴着窗口左边: 是从标签栏伸过来的, 不要
                continue
            lefts.append(int(cols[0]) + wx0)
    r["lefts"] = " ".join(map(str, lefts))

    # "处理进度"标签: 第一个圆点那一行, 取值列左边的灰字(只记录, 不参与判定)
    ly0, ly1 = max(0, int(c0[1] - 0.3 * D)), min(H, int(c0[1] + c0[3] + 0.3 * D))
    lx0 = int(0.03 * W)
    lx1 = max(lx0 + 1, int(cl - 1.5 * D))
    _, grey = _dark_grey(bgr[ly0:ly1, lx0:lx1])
    r["label_ink"] = round(int(grey.sum()) / (D * D), 3)
    r["label"] = int(int(grey.sum()) >= LABEL_INK * D * D)

    # 第一个圆点右边的步骤文字左缘(只记录, 不参与判定)
    sx0, sx1 = min(W - 1, c0[0] + c0[2] + 1), min(W, int(c0[0] + c0[2] + 3 * D))
    sd, _ = _dark_grey(bgr[ly0:ly1, sx0:sx1])
    sc = np.flatnonzero(sd.any(0))
    r["step_gap"] = round((int(sc[0]) + sx0 - (cl + D)) / D, 3) if len(sc) else ""

    r.update(decide(W, H, D, cl, lefts))
    return r


def decide(W, H, D, cl, lefts):
    """量好之后的判定。lefts = 圆点上方各行的左缘, 离时间轴最近的在前。
    单独拿出来是为了能拿扫描结果 csv 重新判(--redecide), 不用重扫图; C# 的 Run() 后半段和这里一一对应。"""
    out = {"verdict": "CannotDetermine", "value_left": "", "offset_px": "", "offset": "", "ios": int((W, H) in IOS_RES)}
    if D < MIN_DIAMETER:
        out["why"] = f"圆点直径小于 {MIN_DIAMETER} 像素, 图被缩小过"
        return out
    if not lefts:
        out["why"] = "圆点上方找不到取值列文字"
        return out
    edge = cl + STEP_TEXT * D
    if lefts[0] >= edge:
        out["why"] = "紧挨着上方的是步骤文字, 第一个圆点没找到"
        return out
    vals = [x for x in lefts if x < edge][:2]       # 再往上从圆点右边开始的行(居中的金额之类)跳过
    vl = min(vals)
    off = (cl - vl) / D
    out.update(value_left=vl, offset_px=cl - vl, offset=round(off, 4))
    if off < GUARD or off > GUARD_HIGH:
        out["why"] = "偏移差得太多, 版面没认对"
        return out
    sus = off < IOS_THRESHOLD if out["ios"] else off <= THRESHOLD
    out["verdict"] = "Suspicious" if sus else "Ok"
    out["why"] = ("苹果原尺寸截图, 圆点缩进不够" if out["ios"] else "圆点左缘不在取值列文字右边") if sus else ""
    return out


def measure(p):
    try:
        buf = np.fromfile(p, np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None
        if bgr is None and buf.size > 2 and buf[0] == 0xFF and buf[1] == 0xD8:
            # 截断的 JPEG(结尾少了 FFD9 结束标记, 本地实测有这种上传): 补上再解一次, 能解出绝大部分
            bgr = cv2.imdecode(np.concatenate([buf, np.array([0xFF, 0xD9], np.uint8)]), cv2.IMREAD_COLOR)
        if bgr is None:
            return {"path": p, "verdict": "unreadable"}
        return measure_img(bgr, p)
    except Exception as e:  # noqa: BLE001
        return {"path": p, "verdict": "error", "why": repr(e)[:200]}


KEYS = ["path", "W", "H", "verdict", "offset", "offset_px", "circle_left", "value_left", "D", "n", "circle_top",
        "lefts", "label", "label_ink", "step_gap", "ios", "why"]


def tile(r, width=300):
    """时间轴那一块放大(只裁圆点左右一小段, 放大后几个像素的差看得见), 红线 = 量到的取值列左缘。
    就是经理那条线: 红线从文字左缘往下, 真图从圆点左边擦过, 假图切进圆点里。"""
    bgr = cv2.imdecode(np.fromfile(r["path"], np.uint8), cv2.IMREAD_COLOR)
    D, cl, ct = float(r["D"]), int(r["circle_left"]), int(r["circle_top"])
    vl = int(r["value_left"]) if str(r.get("value_left", "")) != "" else cl
    H, W = bgr.shape[:2]
    y0, y1 = max(0, int(ct - 3.4 * D)), min(H, int(ct + 1.6 * D))
    x0, x1 = max(0, int(cl - 1.0 * D)), min(W, int(cl + 3.0 * D))
    c = bgr[y0:y1, x0:x1].copy()
    z = width / c.shape[1]
    c = cv2.resize(c, (width, max(1, int(c.shape[0] * z))), interpolation=cv2.INTER_NEAREST)
    X = int((vl - x0) * z)
    cv2.line(c, (X, 0), (X, c.shape[0] - 1), (0, 0, 255), 1)
    return c


def write_sheets(rows, outdir, tag, cols=4, per=24):
    """对照图: 每张 24 格, 格子上写 序号 / 偏移 / 宽 / 有无标签(L1 有 L0 无); 序号对应 tag_index.csv。"""
    if not rows:
        return
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, f"{tag}_index.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sheet", "no", "offset", "offset_px", "W", "H", "label", "path"])
        for s in range(0, len(rows), per):
            part = rows[s:s + per]
            tiles = []
            for j, r in enumerate(part):
                try:
                    t = tile(r)
                except Exception:  # noqa: BLE001
                    t = np.full((200, 300, 3), 200, np.uint8)
                bar = np.full((22, t.shape[1], 3), 255, np.uint8)
                cv2.putText(bar, f"{s + j + 1}  {float(r['offset']):+.3f}  {r['W']}  L{r['label']}", (3, 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 180), 1)
                tiles.append(np.vstack([bar, t]))
                w.writerow([f"{tag}_{s // per + 1:02d}.png", s + j + 1, r["offset"], r["offset_px"], r["W"], r["H"],
                            r["label"], r["path"]])
            hh = max(t.shape[0] for t in tiles)
            tiles = [cv2.copyMakeBorder(t, 0, hh - t.shape[0] + 4, 0, 4, cv2.BORDER_CONSTANT, value=(150, 150, 150))
                     for t in tiles]
            while len(tiles) % cols:
                tiles.append(np.full_like(tiles[0], 255))
            grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
            cv2.imencode(".png", grid)[1].tofile(os.path.join(outdir, f"{tag}_{s // per + 1:02d}.png"))


def main():
    ap = argparse.ArgumentParser(description="处理进度时间轴位置检查, 批量扫描")
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--out", required=True, help="结果 csv(只记有时间轴的图)")
    ap.add_argument("--sheets", default="", help="出对照图的目录; 不给就不出")
    ap.add_argument("--every", type=int, default=1, help="每 N 张取 1 张")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--from-csv", action="store_true", help="不扫图, 直接读 --out 那个 csv 出统计和对照图")
    ap.add_argument("--redecide", default="", help="配合 --from-csv: 按现在的判定规则重新判一遍, 结果写到这个 csv")
    a = ap.parse_args()

    t0 = time.time()
    if a.from_csv:
        with open(a.out, encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        if a.redecide:
            for r in rows:
                if r.get("verdict") == "error" or r.get("D", "") == "":
                    continue
                lefts = [int(x) for x in (r.get("lefts") or "").split()]
                r.update(decide(int(r["W"]), int(r["H"]), float(r["D"]), int(r["circle_left"]), lefts))
            with open(a.redecide, "w", encoding="utf-8-sig", newline="") as f:
                w = csv.DictWriter(f, fieldnames=KEYS, extrasaction="ignore")
                w.writeheader()
                w.writerows(rows)
        for r in rows:
            if r.get("verdict") in ("Ok", "Suspicious"):
                r["offset"], r["label"] = float(r["offset"]), int(r["label"])
        report(rows, None, a)
        return
    files = []
    for root in a.roots:
        if not os.path.isdir(root):
            raise SystemExit(f"not a folder: {root}")
        for dp, _d, fs in os.walk(root):
            files += [os.path.join(dp, f) for f in fs if os.path.splitext(f)[1].lower() in EXTS]
    files.sort()
    files = files[::max(1, a.every)]
    print(f"images to scan: {len(files):,}  (every {a.every})  listing took {time.time() - t0:.0f}s", flush=True)

    rows, step = [], max(1000, len(files) // 50)
    bad = {"unreadable": [], "error": []}          # 读不了的、出错的单独记, 不算进时间轴页
    with ProcessPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(measure, files, chunksize=32), 1):
            if r and r.get("verdict") in bad:
                bad[r["verdict"]].append(r)
            elif r:
                rows.append(r)
            if i % step == 0 or i == len(files):
                el = time.time() - t0
                print(f"  {i:,}/{len(files):,}  timeline pages {len(rows):,}  {i / max(el, 1e-9):.0f} img/s", flush=True)
    print(f"unreadable files {len(bad['unreadable']):,}   errors {len(bad['error']):,}", flush=True)
    for r in bad["error"][:5]:
        print("  error:", r["path"], r["why"], flush=True)
    with open(a.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=KEYS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    report(rows, len(files), a)
    print(f"done in {time.time() - t0:.0f}s  -> {a.out}")


def report(rows, n_files, a):
    ok = [r for r in rows if r.get("verdict") in ("Ok", "Suspicious")]
    sus = sorted([r for r in ok if r["verdict"] == "Suspicious"], key=lambda r: r["offset"])
    near = sorted([r for r in ok if r["verdict"] == "Ok" and r["offset"] < 0.10], key=lambda r: r["offset"])
    print(f"\nscanned {n_files if n_files is not None else '?'}   timeline pages {len(rows):,}   measured {len(ok):,}")
    print(f"  Suspicious (offset <= {THRESHOLD}, iPhone sizes < {IOS_THRESHOLD})   {len(sus):,}")
    print(f"  Ok but offset < 0.10 (close)         {len(near):,}")
    print(f"  Ok offset >= 0.10                    {len(ok) - len(sus) - len(near):,}")
    other = [r for r in rows if r.get("verdict") not in ("Ok", "Suspicious")]
    for why in sorted({r.get("why", "") for r in other}):
        print(f"  not measured: {why}  {sum(1 for r in other if r.get('why', '') == why):,}")
    if ok:
        v = np.array([r["offset"] for r in ok])
        print("  offset percentiles  p0.1 {:+.3f}  p1 {:+.3f}  p5 {:+.3f}  p50 {:+.3f}  p99 {:+.3f}".format(
            *np.percentile(v, [0.1, 1, 5, 50, 99])))
        print("  label missing: among Ok {:,}   among Suspicious {:,}".format(
            sum(1 for r in ok if r["verdict"] == "Ok" and not r["label"]), sum(1 for r in sus if not r["label"])))
    if a.sheets:
        write_sheets(sus[:240], a.sheets, "suspicious")
        write_sheets(near[:96], a.sheets, "close")
        rest = [r for r in ok if r["offset"] >= 0.10]
        write_sheets(rest[:: max(1, len(rest) // 24)][:24], a.sheets, "normal")
        print("sheets ->", a.sheets)


if __name__ == "__main__":
    main()
