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
  真图: 苹果 0.254~0.259, 安卓 0.22~0.28, 拼音字体最高 0.291; 假图 0.326 / 0.341 / 0.348。
  比例 * 100 >= RR_MAX_PCT 判可疑(整数比较)。

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
4. D = 交易成功底 - 名字底, 太小(< 60 像素)不判; 比例 = (金额顶 - 名字底) / D。

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

RR_MAX_PCT = 31         # (金额顶 - 名字底) * 100 >= 31 * (交易成功底 - 名字底) 判可疑(暂定, 等服务器结果)
INK = 200               # min(B, G, R) < 200 算墨
DARK = 160              # 金额要深色: min < 160
NEUTRAL = 40            # 中性(灰黑): max - min < 40
MIN_D = 60              # 名字底到交易成功底不到 60 像素: 图太小, 不判

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
    # 4. 比例
    d = st[1] - nm[1]
    up = am[0] - nm[1]
    r["d"], r["up"] = d, up
    if d < MIN_D:
        r["reason"] = "字太小"
        return r
    r["ratio"] = round(up / d, 4)
    r["v"] = 2 if up * 100 >= RR_MAX_PCT * d else 1
    q = (up * 2000 + d) // (2 * d)            # 比例的千分位, 四舍五入用整数算(C# 一样), 理由里两边逐字相同
    txt = f"{q // 1000}.{q % 1000:03d}"
    if r["v"] == 2:
        r["reason"] = (f"可疑: 金额偏下, 名字底到金额顶 {up} 像素, 名字底到交易成功底 {d} 像素, "
                       f"比例 {txt} >= 0.{RR_MAX_PCT:02d}(真图一般 0.22~0.29)")
    else:
        r["reason"] = f"正常: 名字底到金额顶 {up} 像素, 名字底到交易成功底 {d} 像素, 比例 {txt}"
    return r


def read_bgr(path):
    import cv2
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)


# ---------------------------------------------------------------- 批量

COLS = ["path", "err", "W", "H", "v", "reason", "nm_a", "nm_b", "am_a", "am_b", "am_x0", "am_x1", "st_a", "st_b",
        "st_x0", "st_x1", "row_a", "row_b", "d", "up", "ratio", "sec"]


def _init_worker():
    import cv2
    cv2.setNumThreads(1)


_CROPS = ""


def _init_crops(crops):
    global _CROPS
    _init_worker()
    _CROPS = crops


def _bin(ratio):
    if ratio is None:
        return None
    if ratio >= 0.30:
        return "R030"
    if ratio >= 0.28:
        return "R028"
    if ratio < 0.21:
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
            # 名字上面一点 到 第一行下面一点 的小图(给人看), 缩到 360 宽; 0.28 以上和 0.21 以下全留, 中间的留 1/300
            import cv2
            import zlib
            W = r["W"]
            b = _bin(r.get("ratio"))
            hsh = zlib.crc32(os.path.basename(path).encode("utf-8"))
            if b == "MID" and hsh % 300:
                b = None
            if b is None and r["v"] == 0 and hsh % 300 == 0:
                b = "CANNOT"
            if b:
                c = img[max(0, r["nm_a"] - W * 3 // 100):min(img.shape[0], r["row_b"] + W // 100)]
                c = cv2.resize(c, (360, max(1, c.shape[0] * 360 // W)), interpolation=cv2.INTER_AREA)
                d = os.path.join(_CROPS, b)
                os.makedirs(d, exist_ok=True)
                cv2.imencode(".png", c)[1].tofile(os.path.join(d, f"{r.get('ratio', 0):.3f}_" +
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
