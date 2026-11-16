r"""凭证模糊检查 —— BlurCheck.cs 的 Python 参考实现 + 服务器批量打分(2026-11-16, 经理: 模糊图片的单独处理就好)。

思路
----
截图是手机直接渲染的, 字、图标、按钮的边都很"硬": 从底色到字色 1~2 个像素就过去了。
图被放大过、拍屏、整张糊过一遍, 边就变"软", 要好几个像素才过去。
所以量页面上干净的强边有多宽, 取中位数:
  - 沿行(竖边)和沿列(横边)找相邻像素差的局部最大处, 取边两侧 x-2 .. x+3 共 6 个像素;
  - 两头落差 C >= MIN_CONTRAST(深字白底差 200 左右, 蓝底白字 150 左右), 中间单调(反向回落 <= 8% 落差);
  - 这条边的宽 E = C / 最大一步(像素); 硬边 1~2, 软边变大;
  - 整页分数 = 所有边 E 的中位数(偶数个取中间两个的平均)。
干净的强边少于 MIN_EDGES 条判不了。只看这一张图自己, 不用知道是苹果还是安卓, 白图蓝图一样算。

门槛按尺寸分三类(page_kind, 只看原图宽高)
------------------------------------------
    截图        BLUR_MIN 2.4
    转账凭证图  VOUCHER_BLUR_MIN 3.1(支付宝生成的分享图, 本来就虚, 用 2.4 会把它们全判模糊)
    照片尺寸    BLUR_MIN 2.4(只做标记, 经理要不要照片单独处理自己定)

服务器实测(2026-11-16, 11 月真图白图 / 蓝图各随机 3 万张, 量到 29,939 / 29,932 张)
-------------------------------------------------------------------------------
    截图        白 中位 1.41 / 99% 1.94, >= 2.4 的 30 张(0.10%); 蓝 中位 1.40 / 99% 1.65, 10 张(0.03%)
                这些是被别的程序放大 / 缩放过的截图(平台收银台 1080x2400 2.96、954x2094 等), 看得清但不是原图
    照片尺寸    白 212 张 / 蓝 160 张, 中位 2.57 / 2.60, >= 2.4 的 143 / 125 张(拿另一部手机拍屏幕)
    转账凭证图  白 565 张(1.9%, 1304x2038 / 1304x2122 / 1252x1954 ...)真图最高 2.88, 0 张 >= 3.1; 蓝图只有 14 张
    合计判模糊  白 0.58%(照片 143 + 截图 30), 蓝 0.45%(照片 125 + 截图 10)
    (只用一个门槛 2.4 的话白图 2.39%, 多出来的全是转账凭证图)
本机变体(最终代码, 截图 1,869 张 / 转账凭证图 30 张 / 照片 7 张, 判模糊的比例)
    截图        JPEG 60 / 40 重存 1 / 4 张(最高 2.40 / 2.48); 缩小一半、TG 1280 0 张;
                高斯 0.8 92%, 高斯 1.2 100%, 先缩 0.7 再放回 97%, 缩 0.5 再放回 99.9%
    转账凭证图  原图最高 2.85, JPEG 60 / 40 最高 2.97 / 2.98 -> 0 张;
                高斯 0.8 80%, 高斯 1.2 97%, 先缩 0.7 再放回 63%, 缩 0.5 再放回 13%(本来就是放大出来的, 再放大变化小)
已知漏洞
    宽 1100~1310、高宽比 1.45~1.66 的清楚截图(裁过的, 服务器 6 万张里 19 张)也用 3.1, 轻微的糊抓不到。
    转账凭证图再缩小到宽 1100 以下(比如 TG 1280 -> 819x1280)就按截图算, 本机 30 张里 20 张判模糊(2.40~2.60);
    服务器上传的图里这种尺寸 >= 2.4 的 6 万张里只有 1 张, 所以没放宽。

用法
----
    python -I training/blur_check.py 图 [图 ...]                              (打分)
    python -u training/blur_check.py --scan D:/download2/OtherImages --since 20261101 --sample 20000 ^
        --workers 16 --max-minutes 60 --out E:/_move_logs/blur_1116/white.csv --copy-to E:/_move_logs/blur_1116/look_white
和 training/demo/BlurCheck.cs 逐步对应(解码成彩色 -> 转灰 -> 量边 -> 中位数), 改一边要同步改另一边。
"""
from __future__ import annotations

import os
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")      # 必须在 import numpy 之前

import argparse  # noqa: E402
import csv  # noqa: E402
import random  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

MIN_CONTRAST = 100      # 边两头至少差这么多灰度
MONO_TOL = 0.08         # 中间反向回落最多落差的 8%
MIN_STEP = 12           # 局部最大的那一步至少这么大(太小的不是边)
MIN_EDGES = 200         # 干净的强边少于这么多条判不了
BLUR_MIN = 2.4          # 截图 / 照片: 中位边宽 >= 这个判模糊
VOUCHER_BLUR_MIN = 3.1  # 支付宝生成的转账凭证图: 本身就虚(服务器真图 2.4~2.9, JPEG 40 重存最高 2.98), 单独用这个门槛

KIND_SCREEN, KIND_VOUCHER, KIND_PHOTO = 0, 1, 2
KIND_NAMES = {KIND_SCREEN: "截图", KIND_VOUCHER: "转账凭证图", KIND_PHOTO: "照片尺寸"}

UPLOAD_DAY = re.compile(r"_([0-9]{8})[0-9]{6}\.")


def page_kind(W, H):
    """只按原图尺寸分(和 C# PageKind 一样):
    1 转账凭证图 = 支付宝生成的转账凭证图: 宽 1304、高 1950~2450(page_type.py 的 W2 尺寸),
                  或存下来又缩小 / 裁过的: 宽 1100~1310、高宽比 1.45~1.66(手机截图都在 1.77 以上;
                  宽 1320 的是苹果 Pro Max 截图裁过的, 是清楚的, 不算);
    2 照片尺寸 = 长边 >= 2600 且长短边比 < 1.45(拿相机拍的), 只做标记, 门槛和截图一样;
    0 截图 = 其它。"""
    if W <= 0 or H <= 0:
        return KIND_SCREEN
    r = H / W
    if (W == 1304 and 1950 <= H <= 2450) or (1100 <= W <= 1310 and 1.45 <= r <= 1.66):
        return KIND_VOUCHER
    if max(W, H) >= 2600 and max(W, H) / min(W, H) < 1.45:
        return KIND_PHOTO
    return KIND_SCREEN


def _edges_1d(a):
    """a: 二维 int32 灰度, 沿 axis=1 找边, 返回每条边的宽 E(float64)"""
    d = a[:, 1:] - a[:, :-1]                 # d[:, x] = a[x+1] - a[x]
    ad = np.abs(d)
    W = a.shape[1]
    lm = np.zeros(ad.shape, dtype=bool)
    lm[:, 1:-1] = (ad[:, 1:-1] >= ad[:, :-2]) & (ad[:, 1:-1] > ad[:, 2:]) & (ad[:, 1:-1] >= MIN_STEP)
    ys, xs = np.nonzero(lm)
    ok = (xs >= 2) & (xs + 3 < W)
    ys, xs = ys[ok], xs[ok]
    if len(ys) == 0:
        return np.zeros(0, np.float64)
    p = a[ys[:, None], xs[:, None] + np.arange(-2, 4)[None, :]]    # x-2 .. x+3
    C = p[:, 5] - p[:, 0]
    Ca = np.abs(C)
    keep = Ca >= MIN_CONTRAST
    steps = (p[:, 1:] - p[:, :-1]) * np.sign(C)[:, None]          # 同方向为正
    keep &= steps.min(axis=1).astype(np.float64) >= -MONO_TOL * Ca
    g = ad[ys, xs]
    return Ca[keep].astype(np.float64) / g[keep].astype(np.float64)


def edge_widths(gray):
    a = gray.astype(np.int32)
    return np.concatenate([_edges_1d(a), _edges_1d(np.ascontiguousarray(a.T))])


def _median(v):
    s = np.sort(v)
    n = len(s)
    return float(s[n // 2]) if n % 2 else float((s[n // 2 - 1] + s[n // 2]) / 2.0)


def check_gray(gray):
    """gray 是原图(不要先缩放)转的灰度。
    返回 dict: v (0 判不了 / 1 清楚 / 2 模糊), edges, width(中位边宽), crisp(E <= 1.5 的比例),
    kind(page_kind), min(这一类用的门槛)"""
    H, W = gray.shape[:2]
    kind = page_kind(W, H)
    t = VOUCHER_BLUR_MIN if kind == KIND_VOUCHER else BLUR_MIN
    e = edge_widths(gray)
    n = int(e.size)
    if n < MIN_EDGES:
        return {"v": 0, "edges": n, "width": None, "crisp": None, "kind": kind, "min": t}
    w = _median(e)
    return {"v": 2 if w >= t else 1, "edges": n, "width": w, "crisp": float((e <= 1.5).mean()), "kind": kind, "min": t}


def read_gray(path):
    """和 C# 一样: 先解码成彩色(BGR), 再转灰"""
    import cv2
    img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# ---------------------------------------------------------------- 批量打分

COLS = ["path", "err", "W", "H", "kind", "v", "edges", "width", "crisp", "sec"]


def _init_worker():
    import cv2
    cv2.setNumThreads(1)


def score_one(path):
    t0 = time.time()
    try:
        g = read_gray(path)
        if g is None:
            return {"path": path, "err": "unreadable"}
        r = check_gray(g)
        return {"path": path, "err": "", "W": g.shape[1], "H": g.shape[0], "kind": r["kind"], "v": r["v"], "edges": r["edges"],
                "width": "" if r["width"] is None else round(r["width"], 4),
                "crisp": "" if r["crisp"] is None else round(r["crisp"], 4), "sec": round(time.time() - t0, 2)}
    except Exception as e:  # noqa: BLE001
        return {"path": path, "err": repr(e)[:200], "sec": round(time.time() - t0, 2)}


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


def _bin(w):
    """抽图看用的分档"""
    if w is None or w == "":
        return None
    w = float(w)
    for lo in (3.0, 2.8, 2.6, 2.4, 2.2, 2.0):
        if w >= lo:
            return f"B{int(lo * 10)}"
    return None


def scan(a):
    paths = _inputs(a.inputs)
    n_all = len(paths)
    if a.since:
        paths = [p for p in paths if (UPLOAD_DAY.search(os.path.basename(p)) or [None, ""])[1] >= a.since]
    rng = random.Random(a.seed)
    rng.shuffle(paths)
    if a.sample:
        paths = paths[:a.sample]
    print(f"images found {n_all}, since {a.since or '-'}: {len(paths)} to score", flush=True)
    from multiprocessing import Pool
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    if a.copy_to:
        os.makedirs(a.copy_to, exist_ok=True)
    t0 = time.time()
    n = err = 0
    copied = {}
    vs = {0: 0, 1: 0, 2: 0}
    kinds = {}
    widths = []
    stopped = False
    block = max(256, a.workers * 64)
    with Pool(a.workers, initializer=_init_worker) as pool, open(a.out, "w", encoding="utf-8-sig", newline="") as fo:
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
                kinds[(r["kind"], r["v"])] = kinds.get((r["kind"], r["v"]), 0) + 1
                if r["width"] != "":
                    widths.append(float(r["width"]))
                b = _bin(r["width"])
                if a.copy_to and b and copied.get(b, 0) < a.copy_per_bin:
                    shutil.copy2(r["path"], os.path.join(a.copy_to, f"{b}_{float(r['width']):.2f}_" + os.path.basename(r["path"])))
                    copied[b] = copied.get(b, 0) + 1
            fo.flush()
            el = time.time() - t0
            print(f"{n} / {len(paths)}  {n / max(el, 1e-9):.2f} pages/s  blurry {vs[2]}  cannot {vs[0]}  err {err}", flush=True)
    print()
    print(f"pages {n}  errors {err}  ok {vs[1]}  blurry {vs[2]}  cannot {vs[0]}"
          + ("  STOPPED BY --max-minutes" if stopped else ""))
    for k in (KIND_SCREEN, KIND_VOUCHER, KIND_PHOTO):
        t = VOUCHER_BLUR_MIN if k == KIND_VOUCHER else BLUR_MIN
        print(f"  kind {k} (min {t}): ok {kinds.get((k, 1), 0)}  blurry {kinds.get((k, 2), 0)}  cannot {kinds.get((k, 0), 0)}")
    if widths:
        q = np.percentile(np.array(widths), [1, 5, 25, 50, 75, 95, 99, 99.5, 99.9])
        print("width percentiles 1/5/25/50/75/95/99/99.5/99.9:", " ".join(f"{x:.2f}" for x in q))
        for t in (2.0, 2.2, 2.4, 2.6, 2.8, 3.0):
            k = sum(x >= t for x in widths)
            print(f"  >= {t:.1f}: {k}  ({100.0 * k / len(widths):.2f} %)")
    print("copied", dict(sorted(copied.items())))
    print(f"BLUR_SCAN_DONE pages={n} minutes={(time.time() - t0) / 60:.1f}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--since", default="")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1116)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-minutes", type=float, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--copy-to", default="")
    ap.add_argument("--copy-per-bin", type=int, default=40)
    a = ap.parse_args()
    if a.scan:
        if not a.out:
            ap.error("--scan 要 --out")
        scan(a)
        return
    for p in a.inputs:
        g = read_gray(p)
        if g is None:
            print("UNREADABLE", p)
            continue
        r = check_gray(g)
        print(g.shape[1], g.shape[0], r, p)


if __name__ == "__main__":
    main()
