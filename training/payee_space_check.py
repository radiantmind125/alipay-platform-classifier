r"""收款方空白检查 —— PayeeSpaceCheck.cs 的 Python 参考实现 + 服务器批量量(2026-11-17, 经理: 针对这个写个算法 检测这个空白位置)。

页面
----
蓝色页头的 转账成功 页: 转账成功 / ¥金额 / 收款方 ... 名字(**某) / 付款方式 ... / 下面白卡片 立即通知收款人。
经理给的 3 张假图(1320x2868)页头 收款方 那一行是 名字（**某）: 名字和左括号之间空一大块(34~35 像素, 字高 42),
真图同尺寸只空 12~14 像素(字高 40)。OCR 读出来有时带空格有时不带, 所以按像素量。

做法(只看图, 和 C# 一步一步对应)
--------------------------------
1. 页头: 从 0.12H 往下找第一行"蓝底像素不到一半"的行, 那以上是蓝色页头(最多到 0.55H)。
   蓝底像素: B >= 200, R <= 130, B - R >= 100。
2. 页头里的字: R >= 150 的像素(蓝底 R 只有 50~80, 白字 200 以上)。按行数字像素, 连续有字的行是一条行带。
3. 最高的行带是金额; 金额下面第一条行带是 收款方 那一行(拼音页: 比下一条矮一半以上的是拼音, 跳过)。
4. 这一行按列切成一段段(连续有字的列), 最大的空把左边标签和右边取值分开;
   标签要从 0.25W 以内起、宽 2.0~4.0 个字高(收款方 三个字, 拼音字体窄一些), 取值要到 0.75W 以后才完。
5. 取值最后一段要像右括号; 找左括号紧跟的第一个星号: 竖着只占一小截(<= 0.45 行高)、不在最底下、宽 <= 0.75 行高、宽 >= 0.4 倍自己的高。
   星号前一段是左括号(窄: 宽 <= 0.4 行高、宽 <= 0.6 倍自己的高、高 >= 0.4 行高), 再前一段是名字的最后一个字。
6. 名字最后一个字判不了的几种(宁可不判):
     太窄(宽 < 0.3 行高: 1 l 川 的一竖、括号)、太矮(高 < 0.5 行高: 。，、);
     ？ ！ 收尾(竖着两截以上、最下面一截是小点: 高 <= 0.25 行高、宽 <= 0.4 行高), 全角 ？ ！ 右边本来就空;
     那个字到左括号之间 B < 120 的像素 >= 行高平方 / 40(深色 / 彩色表情);
     名字和括号之间(两边各让 max(2, 行高/10) 像素)和底色差 > 40 的像素 >= 行高平方 / 40(蓝色、灰色表情 💤 👤),
     底色 = 标签和取值之间那一大段空白每个通道的下中位数。
7. 空白 = 左括号最左的墨 - 名字最后一个字最右的墨(像素); 比例 = 空白 / 行高。空白 * 100 >= GAP_MIN_PCT * 行高 判可疑。

实测(2026-11-17)
    经理的 3 张假图 0.81~0.83; 经理截的图(780x1280 压过)0.82; JPEG 90~40、缩 0.75 / 0.5、TG 1280、BGRA 0.81~0.90
    服务器 11 月蓝图全部 95,810 张, 量到约 2 万张(其余不是这个页面、或者是上面判不了的几种):
        苹果原生截图 0.29~0.34 居多(名字和括号之间本来就空半个字), 安卓 0.02~0.25,
        最后一个字是 月 日 一类(右边留白多)、拉丁字母、＆ 0.45~0.50, 方块字(□)0.50;
        >= 0.65 只有 1 张: 昵称自己带了几个空格(晨SIR   (**英), 1.19), 这种分不出来。
    服务器 11 月白图随机 3 万张: 0 张量到(没有蓝色页头)。
    本机真图 3,302 张整页(7~11 月): 0 张可疑。

返回: v 0 判不了 / 1 正常 / 2 可疑(和别的检查一样 Ok=1 Suspicious=2 CannotDetermine=0)。

用法
----
    python -I training/payee_space_check.py 图 [图 ...]
    python -u training/payee_space_check.py --scan E:/BlueImages --since 20261101 --workers 16 --max-minutes 60 ^
        --out E:/_move_logs/space_1117/blue.csv --crops E:/_move_logs/space_1117/crops
和 training/demo/PayeeSpaceCheck.cs 逐步对应, 改一边要同步改另一边。
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

GAP_MIN_PCT = 65        # 空白 / 行高 >= 65% 判可疑; 整数比较, C# 和这里不会差在小数上
#                         服务器 11 月蓝图真图(排除下面几种判不了的)最高 0.50, 假图 0.81 起, 取中间
INK_R = 150             # 页头白字: R >= 这个
FOREIGN_B = 120         # 页头里 B < 这个的像素既不是白字也不是蓝底(深色表情、头像、浮窗);
#                         JPEG 在白字边上把蓝底的 B 压到 140~160(色度抽样), 所以不能用 200
NONBG_DIST = 40         # 空白里和底色差(三通道差的绝对值之和) > 40 的像素算"有东西"(蓝色 / 灰色表情 💤 👤 等)

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
    """img: 原图解码成的 BGR(uint8)。返回 dict, v / reason 一定有, 量到的话还有 gap / h / ratio 和位置"""
    H, W = img.shape[:2]
    r = {"v": 0, "W": W, "H": H}
    if H < 400 or W < 300:
        r["reason"] = "图太小"
        return r
    B = img[..., 0].astype(np.int32)
    R = img[..., 2].astype(np.int32)
    lim = int(0.55 * H)
    # 1. 蓝色页头的底
    blue = (B[:lim] >= 200) & (R[:lim] <= 130) & (B[:lim] - R[:lim] >= 100)
    bcnt = blue.sum(axis=1)
    yb = lim
    for y in range(int(0.12 * H), lim):
        if bcnt[y] * 2 < W:
            yb = y
            break
    r["yb"] = yb
    if yb < int(0.15 * H):
        r["reason"] = "没有蓝色页头"
        return r
    # 2. 行带
    xa, xz = int(0.02 * W), int(0.98 * W)
    ink = R[:yb] >= INK_R
    rowcnt = ink[:, xa:xz].sum(axis=1)
    bands = _runs(rowcnt >= 2)
    if len(bands) < 3:
        r["reason"] = "页头字行不到 3 条"
        return r
    hs = [b - a for a, b in bands]
    ia = int(np.argmax(hs))
    # 3. 金额下面第一条(拼音行跳过)
    pay = -1
    for i in range(ia + 1, len(bands)):
        if i + 1 < len(bands) and hs[i] * 10 < 6 * hs[i + 1]:
            continue
        pay = i
        break
    if pay < 0:
        r["reason"] = "金额下面没有字行"
        return r
    y0, y1 = bands[pay]
    return measure_line(img, y0, y1, r)


def _lowmed(v):
    """uint8 数组的下中位数(第 (n+1)//2 小的值), 用直方图算, C# 一样"""
    c = np.cumsum(np.bincount(v.ravel(), minlength=256))
    return int(np.searchsorted(c, (v.size + 1) // 2))


def measure_line(img, y0, y1, r):
    """第 4~7 步: 已经知道 收款方 那一行是 [y0, y1) 行。服务器小图复核也直接调这个(左边补底色到原来的 x)。"""
    H, W = img.shape[:2]
    B = img[y0:y1, :, 0].astype(np.int32)
    R = img[y0:y1, :, 2].astype(np.int32)
    h = y1 - y0
    r["y0"], r["y1"], r["h"] = y0, y1, h
    if h < 12:
        r["reason"] = "字太小"
        return r
    band = R >= INK_R
    seg = _runs(band.sum(axis=0) > 0)
    if len(seg) < 5:
        r["reason"] = "收款方那一行字太少"
        return r
    # 4. 标签 / 取值
    gaps = [seg[i + 1][0] - seg[i][1] for i in range(len(seg) - 1)]
    k = int(np.argmax(gaps))
    lab, val = seg[:k + 1], seg[k + 1:]
    lw = lab[-1][1] - lab[0][0]
    r["label_x0"], r["label_w"] = lab[0][0], round(lw / h, 3)
    if lab[0][0] > 0.25 * W or not (2.0 * h <= lw <= 4.0 * h) or val[-1][1] < 0.75 * W:
        r["reason"] = "这一行不像 收款方 ... 名字"
        return r
    # 5. 每一段的宽、上沿、下沿
    feats = []
    for a, b in val:
        rows = np.nonzero(band[:, a:b].any(axis=1))[0]
        feats.append((a, b, b - a, int(rows[0]), int(rows[-1]) + 1))
    # 星号 = 竖着只占一小截、不在最底下、不太宽也不太扁; 左括号 = 窄、比自己宽高得多、至少 0.4 行高。
    # 要"左括号紧跟星号"才算: 小写字母(o a u ...)也像星号, 但前面不是左括号
    def is_star(f):
        a, b, w, t, bt = f
        e = bt - t
        return e * 100 <= 45 * h and t * 100 < 60 * h and w * 100 <= 75 * h and w * 10 >= 4 * e

    def is_bracket(f):
        a, b, w, t, bt = f
        e = bt - t
        return w * 10 <= 4 * h and w * 10 <= 6 * e and e * 10 >= 4 * h

    if not is_bracket(feats[-1]):
        r["reason"] = "取值最后不是右括号(不是 名字(**某) 的样子)"
        return r
    any_star = False
    star = -1
    for i in range(len(feats)):
        if is_star(feats[i]):
            any_star = True
            if i >= 1 and is_bracket(feats[i - 1]):
                star = i
                break
    if star < 0:
        r["reason"] = ("星号前面不是左括号" if any_star else "取值里没有星号") + "(不是 名字(**某) 的样子)"
        return r
    if star < 2:
        r["reason"] = "左括号前面没有名字"
        return r
    ba, bb, bw, bt0, bt1 = feats[star - 1]
    pa, pb, pw, pt0, pt1 = feats[star - 2]
    r["br_x0"], r["prev_x1"] = ba, pb
    # 6. 名字最后一个字
    if pw * 10 < 3 * h:
        r["reason"] = "名字最后一个字太窄(1 l 川 或括号一类), 量不准"
        return r
    if (pt1 - pt0) * 10 < 5 * h:
        r["reason"] = "名字最后一个是标点(。，、一类), 量不准"
        return r
    # ？ ！ 收尾: 竖着分成两截以上, 最下面一截是个小点(高 <= 0.25 行高、宽 <= 0.4 行高); 全角的 ？ ！ 右边本来就空一大块
    rr = _runs(band[:, pa:pb].any(axis=1))
    if len(rr) >= 2:
        la, lb = rr[-1]
        lc = np.nonzero(band[la:lb, pa:pb].any(axis=0))[0]
        if (lb - la) * 4 <= h and (int(lc[-1]) - int(lc[0]) + 1) * 10 <= 4 * h:
            r["reason"] = "名字最后一个是 ？ ！ 一类(下面一个点), 量不准"
            return r
    if int((B[:, pa:ba] < FOREIGN_B).sum()) * 40 >= h * h:
        r["reason"] = "名字后面有表情图标一类的彩色东西, 量不准"
        return r
    # 空白里有没有别的东西(蓝色 / 灰色的表情, R 不够白字、B 又不低): 和底色比; 底色 = 标签和取值之间那一大段空白的下中位数
    m = max(2, h // 10)
    if ba - m > pb + m:
        ga, gb = seg[k][1], seg[k + 1][0]
        bg = [_lowmed(img[y0:y1, ga:gb, c]) for c in range(3)]
        reg = img[y0:y1, pb + m:ba - m].astype(np.int32)
        dist = np.abs(reg[..., 0] - bg[0]) + np.abs(reg[..., 1] - bg[1]) + np.abs(reg[..., 2] - bg[2])
        if int((dist > NONBG_DIST).sum()) * 40 >= h * h:
            r["reason"] = "名字和括号之间有表情一类的东西, 量不准"
            return r
    # 7. 空白
    gap = ba - pb
    r["gap"] = gap
    r["ratio"] = round(gap / h, 4)
    r["v"] = 2 if gap * 100 >= GAP_MIN_PCT * h else 1
    q = (gap * 200 + h) // (2 * h)            # 比例的百分位, 四舍五入用整数算(C# 一样), 理由里两边逐字相同
    txt = f"{q // 100}.{q % 100:02d}"
    r["reason"] = (f"可疑: 名字和左括号之间空 {gap} 像素 = {txt} 个字高 >= 0.{GAP_MIN_PCT:02d}(真图一般 0.1~0.5)"
                   if r["v"] == 2 else f"正常: 名字和左括号之间空 {gap} 像素 = {txt} 个字高")
    return r


def read_bgr(path):
    import cv2
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)


# ---------------------------------------------------------------- 批量

COLS = ["path", "err", "W", "H", "v", "reason", "yb", "y0", "y1", "h", "label_x0", "label_w", "prev_x1", "br_x0", "gap",
        "ratio", "sec"]


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
    for lo in (0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3):
        if ratio >= lo:
            return f"R{int(round(lo * 100)):03d}"
    return "R000"


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
        if _CROPS and "y0" in r and "label_x0" in r:
            # 收款方那一行的小图(给人看), 上下各多 1 个行高; 0.3 以下的只留 1/50, 判不了的留 1/100
            import cv2
            import zlib
            h = r["h"]
            c = img[max(0, r["y0"] - h):min(img.shape[0], r["y1"] + h), max(0, r["label_x0"] - 10):]
            b = _bin(r.get("ratio"))
            hsh = zlib.crc32(os.path.basename(path).encode("utf-8"))
            if b == "R000" and hsh % 50:
                b = None
            if b is None and r["v"] == 0 and hsh % 100 == 0:
                b = "CANNOT"
            if b:
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
        q = np.percentile(np.array(ratios), [1, 5, 25, 50, 75, 95, 99, 99.5, 99.9])
        print("ratio percentiles 1/5/25/50/75/95/99/99.5/99.9:", " ".join(f"{x:.3f}" for x in q))
        for t in (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80):
            k = sum(x >= t for x in ratios)
            print(f"  >= {t:.2f}: {k}  ({100.0 * k / len(ratios):.3f} % of measured)")
    print(f"SPACE_SCAN_DONE pages={n} minutes={(time.time() - t0) / 60:.1f}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--since", default="")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1117)
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
