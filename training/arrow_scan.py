r"""账单详情页右边"灰字 + > 箭头"的间距检查(经理说的 111) —— Python 参考实现 + 批量扫描。

现象
----
账单管理那几行(账单分类 转账红包 >、标签 请选择 >、备注 添加 >), 右对齐的灰字后面跟一个 > 箭头。
经理: "假图都很靠近后面的箭头, 真图是有距离的"。
本地看下来假图有两处不对, 而且同时出现:
    箭头离文字近    9 月的真图 30~36 像素(1080~1320 宽), 假图 12~20
    箭头大一号      9 月的真图箭头高 20~23 像素, 假图 24~35
所以量 R = 间距 / 箭头高, 两处都算进去了。用箭头高做单位, 不用文字高:
拼音页的文字行带着拼音, 文字高不准; 箭头不受影响。

★ 支付宝 8 月到 9 月之间改过这一块(箭头变小、离远了), 所以真图有新老两种:
    9 月以后的真图(新版)       R 1.4~1.8
    7~8 月的真图(老版)         R 0.86~0.94   (负号检查那批对照图 CTRL 全是这样)
    拼音页(箭头本来就大一号)   R 0.95 起
    假图                       R 0.62~0.70
  经理说的"真图有距离"是对新版说的; 没升级的用户截出来是老版的样子, 不能当假图。
  所以阈值放在假图和老版真图之间, 不放在假图和新版真图之间。

量法
----
1. 页面右半边(宽度 45% 往右)按行切开, 每行最右边一段如果是 > 形的窄条, 而且离页面右边 3%~14% 宽,
   就当它是箭头; 它左边那一段是文字的结尾。
2. 只要: 文字是灰的(灰度中位数 115~205), 箭头是灰的(每行笔画最深点的中位数 >= 100, 排除"我的消费图鉴"那种
   蓝底里的黑箭头; 不用整个箭头最深的一个点, 那个点 JPEG 一压就变深, 见 find_rows), 行的左边有黑字标签。
3. 一页里箭头离右边距离最常见的那个(+-2 像素)就是账单管理那一列, 只取这一列的行, 间距和箭头高各取中位数。

用法
----
    python -u training/arrow_scan.py --out E:/x/ar.csv --sheets E:/x/ar_sheets D:/download2/OtherImages
    --every 8   每 8 张取 1 张(和时间轴扫描同一批)
"""
from __future__ import annotations

import os
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")      # 必须在 import numpy 之前, 见 pick_pinyin.py

import argparse  # noqa: E402
import collections  # noqa: E402
import csv  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

cv2.setNumThreads(1)
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# R = 间距 / 箭头高 < 这个值判可疑(暂定, 等服务器数据定)。
# ★ 不能放在假图和新版真图中间(~1.1): 支付宝 8 月到 9 月之间改过这一块,
#   7~8 月的真图(老版本)R 0.86~0.94, 9 月以后的真图 1.4~1.8, 假图 0.62~0.70。
#   还没升级的用户截出来就是老版本的样子, 所以线要放在假图和老版本真图之间。
THRESHOLD = 0.78
MIN_ARROW_H = 14         # 箭头高小于这个像素数不判: 图被缩小过(贴线的另外有"差一个像素"那条管)

IOS_RES = {(1179, 2556), (1290, 2796), (1170, 2532), (1320, 2868), (1206, 2622), (1284, 2778),
           (1125, 2436), (1242, 2688), (828, 1792), (750, 1334), (1242, 2208)}


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


def _is_chevron(sub):
    """'>' 形: 每一行只有一笔(一段), 笔画细; 每行那一笔的中心从上往中间往右走, 再往下往左回来。
    ★ 只比"最右点"不够: 数字 9、5, 往下的 ∨ 都能混过去(服务器上真把时间戳最后一位当成了箭头)。
      9 的上半是个圈, 一行两段; ∨ 的上半两条腿, 一行两段; > 每行都只有一段。"""
    ys = np.flatnonzero(sub.any(1))
    if len(ys) < 6:
        return False
    top, bot = ys[0], ys[-1]
    h = bot - top + 1
    w = sub.shape[1]
    cx = []
    for y in range(top, bot + 1):
        segs = _runs(sub[y])
        if len(segs) != 1:
            if len(segs) > 1:
                return False
            cx.append(None)
            continue
        a, b = segs[0]
        if b - a + 1 > 0.7 * w and not (h * 0.35 <= y - top <= h * 0.65):   # 除了尖上, 每行的笔画要细
            return False
        cx.append((a + b) / 2)
    q = max(1, h // 5)

    def mean(lo, hi):
        v = [c for c in cx[lo:hi] if c is not None]
        return sum(v) / len(v) if v else None
    t, m, btm = mean(0, q), mean(h // 2 - q // 2, h // 2 + q // 2 + 1), mean(h - q, h)
    return None not in (t, m, btm) and m - t >= 0.25 * w and m - btm >= 0.25 * w


def find_rows(bgr):
    """所有"灰字 + 灰色 > 箭头"的行。"""
    H, W = bgr.shape[:2]
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    s = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 1]
    ink = (g < 215) & (s < 60)
    dark = (g < 110) & (s < 60)
    x0 = int(0.45 * W)
    reg = ink[:, x0:]
    out = []
    for a, b in _runs(reg.sum(1) >= 2):
        h = b - a + 1
        if h < 0.015 * W or h > 0.05 * W:
            continue
        cr = _runs(reg[a:b + 1].any(0))
        if len(cr) < 2:
            continue
        ca, cb = cr[-1]
        cw = cb - ca + 1
        sub = reg[a:b + 1, ca:cb + 1]
        ys = np.flatnonzero(sub.any(1))
        ch = int(ys[-1] - ys[0] + 1)
        margin = W - (x0 + cb) - 1
        if not (2 <= cw <= 0.75 * ch and ch >= 0.4 * h and 0.03 * W <= margin <= 0.14 * W):
            continue
        if not _is_chevron(sub):
            continue
        # 箭头有多深: 每一行笔画最深的那个点, 取各行的中位数。
        # ★ 不能取整个箭头最深的一个点: JPEG 压一下(质量 75 以下)或者放大, 笔画边上会冒出几个更深的噪点,
        #   造假工具的箭头(最深 119)就被压到 100 以下, 整列被当成黑箭头扔掉, 结论变成判不了(2026-10-05 经理的两张假图)。
        ga = g[a:b + 1, x0 + ca:x0 + cb + 1]
        arrow_gray = int(np.median([ga[y][sub[y]].min() for y in range(sub.shape[0]) if sub[y].any()]))
        arrow_min = int(ga[sub].min())            # 老办法(整个箭头最深一个点), 只记录, 用来数这次改动影响了哪些图
        if arrow_gray < 100:                      # 黑箭头: 蓝底胶囊里的那种, 不是账单管理的行
            continue
        ta, tb = cr[-2]
        tsub = reg[a:b + 1, :tb + 1]
        text_gray = int(np.median(g[a:b + 1, x0:x0 + tb + 1][tsub]))
        if not (115 <= text_gray <= 205):         # 取值是灰字
            continue
        if not dark[a:b + 1, int(0.03 * W):int(0.25 * W)].any():   # 左边要有黑字标签
            continue
        # 取值文字的起点: 从结尾往左, 字和字之间的空小于一个行高就还算同一段
        j = len(cr) - 2
        while j > 0 and cr[j][0] - cr[j - 1][1] - 1 < h:
            j -= 1
        # 账单管理的取值是右对齐的短字, 左边到标签之间是一大片空白。
        # 起点左边 1.5 个行高以内有字, 说明是左对齐的长取值(比如"付款方式 中国农业银行(xxxx) >"), 不要
        tl = x0 + cr[j][0]
        if ink[a:b + 1, max(0, tl - int(1.5 * h)):tl].any():
            continue
        out.append({"y": a, "h": h, "text_left": x0 + cr[j][0], "text_right": x0 + tb, "arrow_left": x0 + ca,
                    "arrow_w": cw, "arrow_h": ch, "margin": margin, "gap": ca - tb - 1, "arrow_gray": arrow_gray, "arrow_min": arrow_min,
                    "text_gray": text_gray})
    return out


def _is_pinyin_page(bgr):
    """整页带不带拼音标注, 用现成的 pinyin_probe(C# 那边是 PinyinCheck, 两边对应, 26 万张实测过)。
    ★ 不自己按行判: 拼音有时自成一行、有时和汉字粘成一块, 按行判漏得多。"""
    import pinyin_probe as pp
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    g = gray
    if max(g.shape) > 1400:
        sc = 1400 / max(g.shape)
        g = cv2.resize(g, (int(g.shape[1] * sc), int(g.shape[0] * sc)), interpolation=cv2.INTER_NEAREST)
    flat = float(np.bincount(g.ravel(), minlength=256).max()) / g.size
    cs = pp._components(pp._text_mask(gray), H, W)
    if len(cs) < 60:
        return False
    ratio, n_small, stacked, _big_h = pp._stacking(cs)
    if n_small == 0:
        return False
    return pp.has_pinyin({"pinyin_ratio": ratio, "stacked": stacked, "flat": flat})


def measure_img(bgr, path=""):
    """没有这种行返回 None; 否则返回一行结果。"""
    H, W = bgr.shape[:2]
    if W < 500 or H < 800:
        return None
    rows = find_rows(bgr)
    if not rows:
        return None
    # 账单管理那一列: 箭头离右边距离相同(+-2 像素)、箭头大小相近的行, 至少 2 行。
    # 只有 1 行的多半是别的东西(蓝底转账页的"预约转账 >"、促销倒计时条), 不判。
    mc = collections.Counter(r["margin"] for r in rows).most_common(1)[0][0]
    col = [r for r in rows if abs(r["margin"] - mc) <= 2]
    ah = float(np.median([r["arrow_h"] for r in col]))
    col = [r for r in col if abs(r["arrow_h"] - ah) <= 0.2 * ah]
    if len(col) < 2:
        return None
    gap = float(np.median([r["gap"] for r in col]))
    aw = float(np.median([r["arrow_w"] for r in col]))
    r = {"path": path, "W": W, "H": H, "n": len(col), "margin": mc, "gap": gap, "arrow_h": ah, "arrow_w": aw,
         "arrow_gray": int(np.median([x["arrow_gray"] for x in col])), "ys": " ".join(str(x["y"]) for x in col),
         "hs": " ".join(str(x["h"]) for x in col), "ios": int((W, H) in IOS_RES),
         "rescued": sum(1 for x in rows if x["arrow_min"] < 100)}   # 老办法会扔掉的行数; 0 = 结论和改之前一模一样
    # 拼音检查比较贵, 只在要判可疑的时候才跑
    pinyin = int(_is_pinyin_page(bgr)) if ah >= MIN_ARROW_H and (gap + 1) / ah < THRESHOLD else ""
    r["pinyin"] = pinyin
    r.update(decide(gap, ah, pinyin))
    return r


def decide(gap, ah, pinyin):
    """pinyin 只在 R < THRESHOLD 时才有意义(其余情况没测, 传什么都不影响结果)。"""
    out = {"ratio": round(gap / ah, 4) if ah else ""}
    if ah < MIN_ARROW_H:
        out.update(verdict="CannotDetermine", why=f"箭头高小于 {MIN_ARROW_H} 像素, 图被缩小过")
        return out
    if gap / ah >= THRESHOLD:
        out.update(verdict="Ok", why="")
        return out
    if (gap + 1) / ah >= THRESHOLD:
        # 间距再量长一个像素就过线了: 差在测量误差以内, 不判。
        # 小图(箭头 16 像素)一个像素就是 0.06, 大图(箭头 34 像素)是 0.03, 这样小图自动更保守。
        out.update(verdict="CannotDetermine", why="离线不到一个像素, 不判")
        return out
    if pinyin:
        # 拼音模式下支付宝把箭头画大一号、间距也小一点, 真图 R 在 1.0 上下, 离假图太近, 不判
        out.update(verdict="CannotDetermine", why="拼音页, 箭头本来就大, 不判")
        return out
    out.update(verdict="Suspicious", why="箭头离文字太近")
    return out


def measure(p):
    try:
        buf = np.fromfile(p, np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None
        if bgr is None and buf.size > 2 and buf[0] == 0xFF and buf[1] == 0xD8:
            bgr = cv2.imdecode(np.concatenate([buf, np.array([0xFF, 0xD9], np.uint8)]), cv2.IMREAD_COLOR)
        if bgr is None:
            return {"path": p, "verdict": "unreadable"}
        return measure_img(bgr, p)
    except Exception as e:  # noqa: BLE001
        return {"path": p, "verdict": "error", "why": repr(e)[:200]}


KEYS = ["path", "W", "H", "verdict", "ratio", "gap", "arrow_h", "arrow_w", "arrow_gray", "margin", "n", "ys", "hs", "pinyin", "rescued",
        "ios", "why"]


def tile(r, width=420):
    """账单管理那几行的右半边, 放大。"""
    bgr = cv2.imdecode(np.fromfile(r["path"], np.uint8), cv2.IMREAD_COLOR)
    H, W = bgr.shape[:2]
    ys = [int(v) for v in str(r["ys"]).split()][:2]
    hs = [int(v) for v in str(r["hs"]).split()][:2]
    parts = []
    for y, h in zip(ys, hs):
        c = bgr[max(0, y - 6):min(H, y + h + 6), int(0.55 * W):W]
        z = width / c.shape[1]
        parts.append(cv2.resize(c, (width, max(1, int(c.shape[0] * z))), interpolation=cv2.INTER_AREA))
    return np.vstack(parts)


def write_sheets(rows, outdir, tag, cols=3, per=24):
    if not rows:
        return
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, f"{tag}_index.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sheet", "no", "ratio", "gap", "arrow_h", "W", "H", "path"])
        for s in range(0, len(rows), per):
            tiles = []
            for j, r in enumerate(rows[s:s + per]):
                try:
                    t = tile(r)
                except Exception:  # noqa: BLE001
                    t = np.full((60, 420, 3), 200, np.uint8)
                bar = np.full((20, t.shape[1], 3), 255, np.uint8)
                cv2.putText(bar, f"{s + j + 1}  R{float(r['ratio']):.2f}  g{float(r['gap']):.0f} a{float(r['arrow_h']):.0f}  {r['W']}",
                            (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 180), 1)
                tiles.append(np.vstack([bar, t]))
                w.writerow([f"{tag}_{s // per + 1:02d}.png", s + j + 1, r["ratio"], r["gap"], r["arrow_h"], r["W"], r["H"], r["path"]])
            hh = max(t.shape[0] for t in tiles)
            tiles = [cv2.copyMakeBorder(t, 0, hh - t.shape[0] + 4, 0, 4, cv2.BORDER_CONSTANT, value=(150, 150, 150))
                     for t in tiles]
            while len(tiles) % cols:
                tiles.append(np.full_like(tiles[0], 255))
            grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
            cv2.imencode(".png", grid)[1].tofile(os.path.join(outdir, f"{tag}_{s // per + 1:02d}.png"))


def report(rows, a):
    ok = [r for r in rows if r.get("verdict") in ("Ok", "Suspicious")]
    for r in ok:
        r["ratio"] = float(r["ratio"])
    sus = sorted([r for r in ok if r["verdict"] == "Suspicious"], key=lambda r: r["ratio"])
    print(f"\npages with a measured arrow row: {len(ok):,}   Suspicious (R < {THRESHOLD}) {len(sus):,}")
    other = [r for r in rows if r.get("verdict") not in ("Ok", "Suspicious")]
    for why in sorted({r.get("why", "") for r in other}):
        print(f"  not measured: {why}  {sum(1 for r in other if r.get('why', '') == why):,}")
    if not ok:
        return
    # 2026-10-05 箭头深浅改成按行取中位数: 只有 rescued > 0 的图结论可能和改之前不一样
    res = [r for r in rows if int(r.get("rescued") or 0) > 0]
    print(f"  pages where the arrow-darkness change kept extra rows: {len(res):,}   " + "  ".join(
        f"{k} {v:,}" for k, v in collections.Counter(r.get("verdict") for r in res).most_common()))
    v = np.array([r["ratio"] for r in ok])
    print("  R percentiles  p0.1 {:.2f}  p1 {:.2f}  p5 {:.2f}  p50 {:.2f}  p99 {:.2f}".format(
        *np.percentile(v, [0.1, 1, 5, 50, 99])))
    print("  R histogram:", "  ".join(f"<{b:.1f}:{int(((v >= b - 0.1) & (v < b)).sum())}"
                                      for b in np.arange(0.4, 2.31, 0.1)))
    # 按上传日期(文件名里的时间)看新老版本各占多少: 老版本 R 0.78~1.2, 新版本 >= 1.2
    import re
    days = collections.defaultdict(list)
    for r in ok:
        m = re.search(r"_(20\d{6})\d{6}\.", os.path.basename(r["path"]))
        days[m.group(1) if m else "?"].append(r["ratio"])
    print(f"\n  by upload day: pages   R<{THRESHOLD} (suspicious)   {THRESHOLD}~1.2 (old layout or pinyin)   >=1.2 (new layout)")
    for d in sorted(days):
        x = np.array(days[d])
        print(f"    {d}  {len(x):>6,}   {int((x < THRESHOLD).sum()):>5}   {int(((x >= THRESHOLD) & (x < 1.2)).sum()):>6}   {int((x >= 1.2).sum()):>6}")
    by = collections.defaultdict(list)
    for r in ok:
        by[f"{r['W']}x{r['H']}"].append(r["ratio"])
    print(f"\n  by resolution (pages, R p1 / p50, pages with R < {THRESHOLD}, < 1.0):")
    for k, vv in sorted(by.items(), key=lambda kv: -len(kv[1]))[:25]:
        x = np.array(vv)
        print(f"    {k:<12}{len(x):>7,}   {np.percentile(x, 1):.2f} / {np.median(x):.2f}   {int((x < THRESHOLD).sum()):>4} {int((x < 1.0).sum()):>4}")
    if a.sheets:
        write_sheets(sus[:240], a.sheets, "suspicious")
        near = sorted([r for r in ok if r["verdict"] == "Ok" and r["ratio"] < 1.0], key=lambda r: r["ratio"])
        write_sheets(near[:96], a.sheets, "close")
        rest = [r for r in ok if r["ratio"] >= 1.2]
        write_sheets(rest[:: max(1, len(rest) // 24)][:24], a.sheets, "normal")
        print("sheets ->", a.sheets)
    if getattr(a, "copy_to", ""):
        # 整页原图拷出来人眼看: S_ 全部可疑, C_ 老版本那一段(阈值 ~ 1.0)抽 60 张, O_ 新版本抽 12 张
        import shutil
        os.makedirs(a.copy_to, exist_ok=True)
        mid = sorted([r for r in ok if r["verdict"] == "Ok" and r["ratio"] < 1.0], key=lambda r: r["path"])
        new = sorted([r for r in ok if r["ratio"] >= 1.2], key=lambda r: r["path"])
        picks = [(f"S_{i:03d}_R{r['ratio']:.2f}_{r['W']}x{r['H']}_", r) for i, r in enumerate(sus, 1)]
        # N_ 箭头深浅改动以后才判可疑的(改之前多半是判不了), 全部拷出来看
        picks += [(f"N_{i:03d}_R{r['ratio']:.2f}_{r['W']}x{r['H']}_", r)
                  for i, r in enumerate([r for r in sus if int(r.get("rescued") or 0) > 0], 1)]
        picks += [(f"C_{i:03d}_R{r['ratio']:.2f}_{r['W']}x{r['H']}_", r)
                  for i, r in enumerate(mid[:: max(1, len(mid) // 60)][:60], 1)]
        picks += [(f"O_{i:02d}_R{r['ratio']:.2f}_{r['W']}x{r['H']}_", r)
                  for i, r in enumerate(new[:: max(1, len(new) // 12)][:12], 1)]
        n = 0
        for pre, r in picks:
            try:
                shutil.copy2(r["path"], os.path.join(a.copy_to, pre + os.path.basename(r["path"])))
                n += 1
            except OSError:
                pass
        print(f"copied {n} images -> {a.copy_to}")


def main():
    ap = argparse.ArgumentParser(description="右边灰字和 > 箭头的间距检查, 批量扫描")
    ap.add_argument("roots", nargs="*")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sheets", default="")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--from-csv", action="store_true", help="不扫图, 读 --out 的 csv 出统计和对照图")
    ap.add_argument("--copy-to", default="", help="把可疑的、老版本那一段、新版本样本的整页原图拷到这里")
    a = ap.parse_args()

    t0 = time.time()
    if a.from_csv:
        with open(a.out, encoding="utf-8-sig", newline="") as f:
            report(list(csv.DictReader(f)), a)
        return
    # 先确认结果文件写得进去, 免得扫完十几分钟才发现目录不在或者文件被占用
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "a", encoding="utf-8"):
        pass
    files = []
    for root in a.roots:
        if not os.path.isdir(root):
            raise SystemExit(f"not a folder: {root}")
        for dp, _d, fs in os.walk(root):
            files += [os.path.join(dp, f) for f in fs if os.path.splitext(f)[1].lower() in EXTS]
    files.sort()
    files = files[::max(1, a.every)]
    print(f"images to scan: {len(files):,}  (every {a.every})  listing took {time.time() - t0:.0f}s", flush=True)

    rows, bad, step = [], collections.Counter(), max(1000, len(files) // 50)
    with ProcessPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(measure, files, chunksize=32), 1):
            if r and r.get("verdict") in ("unreadable", "error"):
                bad[r["verdict"]] += 1
            elif r:
                rows.append(r)
            if i % step == 0 or i == len(files):
                el = time.time() - t0
                print(f"  {i:,}/{len(files):,}  pages with arrow rows {len(rows):,}  {i / max(el, 1e-9):.0f} img/s", flush=True)
    print(f"unreadable files {bad['unreadable']:,}   errors {bad['error']:,}", flush=True)
    with open(a.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=KEYS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    report(rows, a)
    print(f"done in {time.time() - t0:.0f}s  -> {a.out}")


if __name__ == "__main__":
    main()
