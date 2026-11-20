r"""金额上下位置检查 —— AmountYCheck.cs 的 Python 参考实现 + 服务器批量量(2026-11-20, 经理: 假图金额Y坐标不对)。

页面
----
白底 账单详情 页: 卡片里居中一摞  logo / 名字 / 金额(-800.00) / 交易成功, 下面左对齐一行行 创建时间 ...。
经理给的 3 张假图(1179x2556 苹果截图)把 金额 + 交易成功 整块贴低了:
名字底到金额顶 73~80 像素, 真图同一种手机(3 倍图)只有 56~61 像素; 交易成功 也跟着往下, 但没金额挪得多。

量什么(只看图, 不用 OCR)
------------------------
比例 = (金额顶 - 名字底) / (交易成功底 - 名字底)
  名字底、交易成功底 都是系统字体中文字的下沿(拼音字体的拼音在字上面, 不影响下沿); 金额是支付宝自己的数字字体,
  数字的上沿很稳(逗号、小数点只往下)。三个都是页面排版的位置, 手机大小、显示缩放都按比例变, 比例不变。
  比例 * 100 >= 门槛 判可疑(整数比较)。门槛分两种:
    图的尺寸正好是 iPhone 原生截图尺寸: 0.30。苹果不能换系统字体, 排版是死的: 同一尺寸的真图位置差不过 2 像素,
      服务器 11 月 1~4 日 iPhone 尺寸的 账单详情 真图最高约 0.28(1284x2778 那一档 0.2785)。
    别的尺寸: 0.32。安卓可以换手写体、拼音字体, 中文字的下沿会往上抬, 真图最高到 0.307。

实测(2026-11-20)
    经理的 3 张假图(1179x2556): 0.326 / 0.341 / 0.348; JPEG 95~40、缩 0.75 / 0.5、TG 1280、长边 2000: 0.323~0.350。
    服务器 11 月白图全部 195,078 张(11-01 ~ 11-04), 量到 164,647 张, p50 0.261, p99 0.276:
      苹果尺寸真图一档一档很整齐, 例如 1170x2532 一万二千多张 名字底到金额顶 56~58 像素、到交易成功底 214~218 像素,
      金额数字高都是 71(带逗号 82)。
      第一版(没有下面第 5 步)按 0.31 判可疑 65 张, 看了小图:
        别的页面 41 张: 银行 app 的交易详情(-¥500.00 / 余额¥... / 交易卡号)14 张、支付成功 + 银行立减金 25 张、
                        转钱包 2 张 —— 加第 5 步的 名字 / 交易成功 要是深灰字 挡掉;
        账单详情 上面盖了弹窗 1 张 —— 第 5 步的 名字和金额之间要干净 挡掉;
        剩下 23 张都是 账单详情, 金额都比同尺寸真图低 14~25 像素, 苹果尺寸的数字还都比真图小一号(1170 宽 62~69, 真图 71),
        里面有以前 111 箭头检查抓到的假图 —— 都是改过的图。
      定版(本文件)同样 195,078 张再跑一遍: 量到 163,810 张, 可疑 16 张(iPhone 尺寸 12、别的 4), 一张张看过都是 账单详情:
        金额比同尺寸真图的常见位置低 6~22 像素, 14 张数字比同尺寸真图小一号(例如 1080 宽 50~57, 真图 66;
        1320x2868 59~68, 真图 79), 另 2 张金额没有小数(-20000)、数字纯黑(苹果真图是 #333)。
        第 5 步挡掉 837 张, 比例 >= 0.30 的 42 张全是别的页面(银行余额页、支付成功 + 立减金、转钱包)和 1 张弹窗;
        其余是 收款金额 / 付款成功 / 转账成功 一类别的页面, 少数是状态不是深灰字的 账单详情(处理中、等待对方领取)。
        iPhone 尺寸量到 76,405 张: 99.9% 在 0.2785 以下; 0.30~0.32 之间的安卓图是以前的假图和手写体 / 拼音字体真图(最高 0.307)。
    服务器 11 月蓝图随机 2 万张: 0 张量到(不是这种页面)。

做法(和 C# 一步一步对应)
------------------------
1. 整屏竖截图才判: W >= 300, H >= 1.6W。
2. 墨 = min(B, G, R) < 200。只看 0.04W~0.96W、上面 0.6H。每行 >= 2 个墨点算有字, 连续有字的行是一条行带;
   行带之间的缝 <= max(2, W/250) 的合并(logo 中间断开)。每条行带记左右墨的范围 x0 / x1。
3. 从上往下找金额: 居中(x0 > 0.12W 且 |x0 + x1 - W| <= 0.06W)、高 0.040W~0.095W、宽 >= 1.5 倍高、
   深色中性(min < 160 的像素里 max - min < 40 的 >= 80%: logo 是彩色的);
   上一条(名字)居中, 下一条(交易成功)居中、宽 <= 0.5W, 再下一条从 0.12W 以内起(第一行 创建时间 / 服务费 ...);
   金额高 >= 1.3 倍 交易成功 高; 名字高在 交易成功 高的 0.9~1.3 倍(名字是标点、字母、~ 一类时下沿不是中文字的下沿, 不判)。
   第一个全对上的就是。
4. D = 交易成功底 - 名字底, 太小(< 60 像素)不判。
5. 是不是 账单详情 那一摞(宁可不判):
   名字、交易成功 两行里 深灰中性像素(min < 110 且 max - min < 40)都要 >= 行高平方 / 16
     (银行页面的 余额 是浅灰字、支付成功页的 支付成功 是蓝字、划掉的原价是浅灰字, 都是 0);
   名字底和金额顶之间(上下各让 max(2, 间距/8), 左右 0.12W~0.88W)和底色差 > 40 的像素 < 千分之五
     (底色 = 这块每个通道的下中位数; 弹窗、水印、阴影会超)。
6. 比例 = (金额顶 - 名字底) / D, 和门槛比。

返回: v 0 判不了 / 1 正常 / 2 可疑(和别的检查一样 Ok=1 Suspicious=2 CannotDetermine=0)。

用法
----
    python -I training/amount_y_check.py 图 [图 ...]
    python -u training/amount_y_check.py --scan D:/download2/OtherImages --since 20261101 --workers 16 --max-minutes 45 ^
        --out E:/_move_logs/amty_1120/white.csv --crops E:/_move_logs/amty_1120/crops_white
和 training/demo/AmountYCheck.cs 逐步对应, 改一边要同步改另一边。
"""
from __future__ import annotations

import os
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")      # 必须在 import numpy 之前

import argparse  # noqa: E402
import csv  # noqa: E402
import random  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

RR_MAX_PCT_IPHONE = 30  # iPhone 原生截图尺寸: (金额顶 - 名字底) * 100 >= 30 * (交易成功底 - 名字底) 判可疑
RR_MAX_PCT = 32         # 别的尺寸(安卓能换字体, 真图最高 0.307)
INK = 200               # min(B, G, R) < 200 算墨
DARK = 160              # 金额要深色: min < 160
NEUTRAL = 40            # 中性(灰黑): max - min < 40
MIN_D = 60              # 名字底到交易成功底不到 60 像素: 图太小, 不判
TEXT_DARK = 110         # 名字 / 交易成功 的深灰字(#333): min < 110 且中性; 浅灰字(#999)、蓝字都不算
GAP_DIST = 40           # 名字和金额之间, 和底色差(三通道差的绝对值之和) > 40 的像素算"有东西"
GAP_MAX_PERMILLE = 5    # 有东西的像素 >= 千分之五: 不判(弹窗、水印)

# iPhone 原生截图尺寸(竖屏), 和 YenCheck.cs 的表一样
IPHONE_SIZES = {
    (640, 1136), (750, 1334), (1242, 2208), (828, 1792), (1125, 2436), (1242, 2688), (1170, 2532),
    (1284, 2778), (1179, 2556), (1290, 2796), (1206, 2622), (1320, 2868), (1260, 2736),
}

UPLOAD_DAY = re.compile(r"_([0-9]{8})[0-9]{6}\.")


def _runs(on):
    """一维 bool -> [(起, 止)), 连续为真的段"""
    out = []
    x, n = 0, len(on)
    while x < n:
        if on[x]:
            x0 = x
            while x < n and on[x]:
                x += 1
            out.append((x0, x))
        else:
            x += 1
    return out


def check_bgr(img):
    """img: 原图解码成的 BGR(uint8)。返回 dict, v / reason 一定有, 量到的话还有位置和 ratio"""
    H, W = img.shape[:2]
    r = {"v": 0, "W": W, "H": H}
    if W < 300 or H * 10 < 16 * W:
        r["reason"] = "不是整屏竖截图"
        return r
    yz = H * 60 // 100
    xa, xz = W * 4 // 100, W * 96 // 100
    sub = img[:yz]
    mn = sub.min(axis=2)
    mx = sub.max(axis=2)
    ink = mn[:, xa:xz] < INK
    # 2. 行带
    raw = _runs(ink.sum(axis=1) >= 2)
    gapmax = max(2, W // 250)
    bands = []
    for a, b in raw:
        if bands and a - bands[-1][1] <= gapmax:
            bands[-1] = (bands[-1][0], b)
        else:
            bands.append((a, b))
    B = []
    for a, b in bands:
        cols = np.nonzero(ink[a:b].any(axis=0))[0]
        B.append((a, b, int(cols[0]) + xa, int(cols[-1]) + xa + 1))     # (上, 下, 左, 右) 右/下不含

    def centered(t):
        return t[2] * 100 > 12 * W and abs(t[2] + t[3] - W) * 100 <= 6 * W

    # 3. 找金额
    amt = -1
    name_bad = False
    for i, t in enumerate(B):
        a, b, x0, x1 = t
        h = b - a
        if not centered(t) or not (40 * W <= h * 1000 <= 95 * W) or (x1 - x0) * 2 < 3 * h:
            continue
        m = mn[a:b, x0:x1] < DARK
        dark = int(m.sum())
        neu = int((m & ((mx[a:b, x0:x1].astype(np.int16) - mn[a:b, x0:x1]) < NEUTRAL)).sum())
        if dark == 0 or neu * 5 < dark * 4:
            continue
        if i < 1 or i + 2 >= len(B):
            continue
        nm, st, row = B[i - 1], B[i + 1], B[i + 2]
        if not (centered(nm) and centered(st)) or (st[3] - st[2]) * 2 > W or row[2] * 100 > 12 * W:
            continue
        sh, nh = st[1] - st[0], nm[1] - nm[0]
        if h * 10 < 13 * sh:
            continue
        if not (9 * sh <= nh * 10 <= 13 * sh):
            name_bad = True
            continue
        amt = i
        break
    if amt < 0:
        r["reason"] = ("名字太矮或太高(标点、字母、符号一类), 量不准" if name_bad
                       else "没找到 名字 / 金额 / 交易成功 / 第一行 这样的一摞(不是账单详情这种页面)")
        return r
    nm, am, st, row = B[amt - 1], B[amt], B[amt + 1], B[amt + 2]
    r.update(nm_a=nm[0], nm_b=nm[1], am_a=am[0], am_b=am[1], am_x0=am[2], am_x1=am[3],
             st_a=st[0], st_b=st[1], st_x0=st[2], st_x1=st[3], row_a=row[0], row_b=row[1])
    # 4. 间距
    d = st[1] - nm[1]
    up = am[0] - nm[1]
    r["d"], r["up"] = d, up
    if d < MIN_D:
        r["reason"] = "字太小"
        return r
    r["ratio_raw"] = round(up / d, 4)         # 不管第 5 步过没过都记(服务器批量时看被挡掉的是什么)

    # 5. 是不是 账单详情 那一摞
    def dark_text(t):
        a, b, x0, x1 = t
        s_mn = mn[a:b, x0:x1]
        return int(((s_mn < TEXT_DARK) & ((mx[a:b, x0:x1].astype(np.int16) - s_mn) < NEUTRAL)).sum())

    nh, sh = nm[1] - nm[0], st[1] - st[0]
    r["n_dark"], r["s_dark"] = dark_text(nm), dark_text(st)
    if r["n_dark"] * 16 < nh * nh or r["s_dark"] * 16 < sh * sh:
        r["reason"] = "名字或交易成功不是深灰色的字(不像账单详情, 比如银行页面的余额、支付成功页), 不判"
        return r
    mg = max(2, up // 8)
    ga, gb = nm[1] + mg, am[0] - mg
    gx0, gx1 = W * 12 // 100, W * 88 // 100
    if gb > ga:
        reg = sub[ga:gb, gx0:gx1]
        bg = [_lowmed(reg[..., c]) for c in range(3)]
        reg = reg.astype(np.int32)
        dist = np.abs(reg[..., 0] - bg[0]) + np.abs(reg[..., 1] - bg[1]) + np.abs(reg[..., 2] - bg[2])
        r["gap_odd"] = int((dist > GAP_DIST).sum())
        if r["gap_odd"] * 1000 >= GAP_MAX_PERMILLE * dist.size:
            r["reason"] = "名字和金额之间有别的东西(弹窗、水印、阴影一类), 量不准"
            return r

    # 6. 比例
    iphone = (W, H) in IPHONE_SIZES
    thr = RR_MAX_PCT_IPHONE if iphone else RR_MAX_PCT
    r["iphone"], r["thr"] = int(iphone), thr
    r["ratio"] = r["ratio_raw"]
    r["v"] = 2 if up * 100 >= thr * d else 1
    q = (up * 2000 + d) // (2 * d)            # 比例的千分位, 四舍五入用整数算(C# 一样), 理由里两边逐字相同
    txt = f"{q // 1000}.{q % 1000:03d}"
    kind = "iPhone 原图尺寸, 真图最高约 0.28" if iphone else "真图最高约 0.31"
    if r["v"] == 2:
        r["reason"] = (f"可疑: 金额偏下, 名字底到金额顶 {up} 像素, 名字底到交易成功底 {d} 像素, "
                       f"比例 {txt} >= 0.{thr:02d}({kind})")
    else:
        r["reason"] = f"正常: 名字底到金额顶 {up} 像素, 名字底到交易成功底 {d} 像素, 比例 {txt}(门槛 0.{thr:02d})"
    return r


def _lowmed(v):
    """uint8 数组的下中位数(第 (n+1)//2 小的值), 用直方图算, C# 一样"""
    c = np.cumsum(np.bincount(v.ravel(), minlength=256))
    return int(np.searchsorted(c, (v.size + 1) // 2))


def read_bgr(path):
    import cv2
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)


# ---------------------------------------------------------------- 批量

COLS = ["path", "err", "W", "H", "v", "reason", "nm_a", "nm_b", "am_a", "am_b", "am_x0", "am_x1", "st_a", "st_b",
        "st_x0", "st_x1", "row_a", "row_b", "d", "up", "ratio_raw", "n_dark", "s_dark", "gap_odd", "iphone", "thr",
        "ratio", "sec"]


def _init_worker():
    import cv2
    cv2.setNumThreads(1)


_CROPS = ""


def _init_crops(crops):
    global _CROPS
    _init_worker()
    _CROPS = crops


def _bin(r):
    """小图放哪一档: SUSP 可疑(原尺寸) / GUARD 第 5 步挡掉但比例 >= 0.30 / R028 / LOW / MID"""
    if r["v"] == 2:
        return "SUSP"
    rr = r.get("ratio_raw")
    if rr is None:
        return None
    if r["v"] == 0:
        return "GUARD" if rr >= 0.30 else "CANNOT"
    if rr >= 0.28:
        return "R028"
    if rr < 0.21:
        return "LOW"
    return "MID"


def score_one(path):
    t0 = time.time()
    try:
        img = read_bgr(path)
        if img is None:
            return {"path": path, "err": "unreadable"}
        r = check_bgr(img)
        out = {"path": path, "err": ""}
        out.update({k: r.get(k, "") for k in COLS if k not in ("path", "err", "sec")})
        out["sec"] = round(time.time() - t0, 3)
        if _CROPS and "nm_a" in r:
            # 名字上面一点 到 第一行下面一点 的小图(给人看): 可疑的留原尺寸, 别的缩到 360 宽;
            # 可疑 / 挡掉的高比例 / 0.28 以上 / 0.21 以下全留, 中间的和判不了的各留 1/300
            import cv2
            import zlib
            W = r["W"]
            b = _bin(r)
            hsh = zlib.crc32(os.path.basename(path).encode("utf-8"))
            if b in ("MID", "CANNOT") and hsh % 300:
                b = None
            if b:
                c = img[max(0, r["nm_a"] - W * 3 // 100):min(img.shape[0], r["row_b"] + W // 100)]
                if b != "SUSP":
                    c = cv2.resize(c, (360, max(1, c.shape[0] * 360 // W)), interpolation=cv2.INTER_AREA)
                d = os.path.join(_CROPS, b)
                os.makedirs(d, exist_ok=True)
                cv2.imencode(".png", c)[1].tofile(os.path.join(d, f"{r.get('ratio_raw', 0):.3f}_" +
                                                              os.path.splitext(os.path.basename(path))[0] + ".png"))
        return out
    except Exception as e:  # noqa: BLE001
        return {"path": path, "err": repr(e)[:200], "sec": round(time.time() - t0, 3)}


def _inputs(args):
    paths = []
    for a in args:
        if a.startswith("@"):
            with open(a[1:], encoding="utf-8") as f:
                paths += [x.strip() for x in f if x.strip()]
        elif os.path.isdir(a):
            for root, _dirs, files in os.walk(a):
                paths += [os.path.join(root, x) for x in files if x.lower().endswith((".jpg", ".jpeg", ".png"))]
        else:
            paths.append(a)
    return paths


def scan(a):
    paths = _inputs(a.inputs)
    n_all = len(paths)
    if a.since:
        paths = [p for p in paths if (UPLOAD_DAY.search(os.path.basename(p)) or [None, ""])[1] >= a.since]
    rng = random.Random(a.seed)
    rng.shuffle(paths)
    if a.sample:
        paths = paths[:a.sample]
    print(f"images found {n_all}, since {a.since or '-'}: {len(paths)} to check", flush=True)
    from multiprocessing import Pool
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    if a.crops:
        os.makedirs(a.crops, exist_ok=True)
    t0 = time.time()
    n = err = 0
    vs = {0: 0, 1: 0, 2: 0}
    reasons = {}
    ratios = []
    by_kind = {"iphone": [], "other": []}
    guard_hi = 0
    stopped = False
    block = max(256, a.workers * 64)
    with Pool(a.workers, initializer=_init_crops, initargs=(a.crops,)) as pool, \
            open(a.out, "w", encoding="utf-8-sig", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        for i in range(0, len(paths), block):
            if a.max_minutes and (time.time() - t0) / 60 >= a.max_minutes:
                stopped = True
                break
            for r in pool.imap_unordered(score_one, paths[i:i + block], chunksize=8):
                w.writerow(r)
                n += 1
                if r.get("err"):
                    err += 1
                    continue
                vs[r["v"]] += 1
                key = r["reason"].split(":")[0] if r["v"] else r["reason"]
                reasons[key] = reasons.get(key, 0) + 1
                if r["ratio"] != "":
                    ratios.append(float(r["ratio"]))
                    by_kind["iphone" if r["iphone"] == 1 else "other"].append(r["v"])
                elif r["ratio_raw"] != "" and float(r["ratio_raw"]) >= 0.30:
                    guard_hi += 1
            fo.flush()
            el = time.time() - t0
            print(f"{n} / {len(paths)}  {n / max(el, 1e-9):.2f} pages/s  measured {vs[1] + vs[2]}  suspicious {vs[2]}  err {err}",
                  flush=True)
    print()
    print(f"pages {n}  errors {err}  ok {vs[1]}  suspicious {vs[2]}  cannot {vs[0]}"
          + ("  STOPPED BY --max-minutes" if stopped else ""))
    for k, c in sorted(reasons.items(), key=lambda t: -t[1]):
        print(f"  {c:7d}  {k}")
    if ratios:
        q = np.percentile(np.array(ratios), [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9])
        print("ratio percentiles 0.1/1/5/25/50/75/95/99/99.9:", " ".join(f"{x:.3f}" for x in q))
        for t in (0.28, 0.29, 0.30, 0.31, 0.32, 0.33):
            k = sum(x >= t for x in ratios)
            print(f"  >= {t:.2f}: {k}  ({100.0 * k / len(ratios):.3f} % of measured)")
        for t in (0.15, 0.18, 0.20, 0.21):
            k = sum(x < t for x in ratios)
            print(f"  <  {t:.2f}: {k}  ({100.0 * k / len(ratios):.3f} % of measured)")
        for kk, v in by_kind.items():
            if v:
                thr = RR_MAX_PCT_IPHONE if kk == "iphone" else RR_MAX_PCT
                print(f"  {kk}: measured {len(v)}, suspicious (>= 0.{thr:02d}): {sum(x == 2 for x in v)}")
    print(f"  stopped by step 5 with ratio >= 0.30: {guard_hi}")
    print(f"AMTY_SCAN_DONE pages={n} minutes={(time.time() - t0) / 60:.1f}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--since", default="")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1120)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-minutes", type=float, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--crops", default="")
    a = ap.parse_args()
    if a.scan:
        if not a.out:
            ap.error("--scan 要 --out")
        scan(a)
        return
    for p in a.inputs:
        img = read_bgr(p)
        if img is None:
            print("UNREADABLE", p)
            continue
        r = check_bgr(img)
        print(r["W"], r["H"], r["v"], r.get("ratio", ""), r["reason"], p)


if __name__ == "__main__":
    main()
