r"""蓝图上 FieldTimeCheck 两条规则会不会误判(2026-11-16)—— 只读图, 不改规则。

蓝图(转账成功 / 到账成功 / 支付成功 蓝色结果页)上一般没有 支付时间 / 创建时间 / 订单号, 到账时间不带秒,
所以规则大多判不了; 但对方 OCR 可能把页面上别的时间、长数字传进来, 这里按"最坏情况"试:
  - 整页找字框(RapidOCR 默认的找框), 只认长条框(宽 >= 页宽 10% 且 >= 4 倍高: 带秒时间、长数字都是长条),
    不做方向分类; 有时间或长数字的那几行, 再认它左边同一行的标签框;
    (本地 20 张蓝图和整页全认比: 找到的带秒时间、长数字完全一样, 单进程每页 2.1 秒 -> 1.3 秒以内)
  - 任何一个带秒的 年-月-日 时:分:秒 都当页面时间, 任何一个 >= 14 位的数字串(时间本身的数字不算)都当订单号,
    两两套规则(规则、窗口、跳过表全部用 field_time_scan.py 里的, 和 FieldTimeCheck.cs 一样), 有一个犯了就记下;
  - 另外照标签(支付时间 / 创建时间 / 订单号 那几行)跑一遍 check(), 和白图的用法一样。
有带秒时间、长数字或者犯了规则的页复制到 --copy-to, 文件名前缀 W1 / W2 / L2(照标签可疑) / TS(只是有带秒时间)。

用法
----
    python -u training/field_time_blue.py D:/download2/BlueImages --since 20261101 --sample 0 --seed 1116 ^
        --workers 16 --max-minutes 100 --out E:/_move_logs/blue_1116/fb.csv --copy-to E:/_move_logs/blue_1116/look
    python -I training/field_time_blue.py 目录或@列表 [...] --out fb.csv          (本地)
--since: 只要文件名上传日期 >= 这天的图; --sample 0 = 全部(打乱顺序, 到 --max-minutes 就不再开新页)。
"""
from __future__ import annotations

import os
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")      # 必须在 import numpy 之前

import argparse  # noqa: E402
import collections  # noqa: E402
import csv  # noqa: E402
import random  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import field_time_scan as F  # noqa: E402

TIME_M = re.compile(r"(20[0-9]{2})[-/.]([0-9]{1,2})[-/.]([0-9]{1,2})[ \t\u00a0\u3000]*([0-9]{1,2})[:\uff1a]([0-9]{2})(?![:\uff1a0-9])")
LABEL_KEYS = ["支付时间", "创建时间", "到账时间", "转账时间", "付款时间", "交易时间", "订单号", "商家订单号", "交易单号", "流水号"]
UPLOAD_DAY = re.compile(r"_([0-9]{8})[0-9]{6}\.")

_ENG = None


def _init_worker():
    global _ENG
    import cv2
    cv2.setNumThreads(1)    # OpenCV 默认按 CPU 数开线程池(本机 16), 多进程时每个进程 25~30 个线程互相抢, 慢 3 倍
    from rapidocr_onnxruntime import RapidOCR
    try:
        _ENG = RapidOCR(intra_op_num_threads=1, inter_op_num_threads=1)
    except Exception as e:  # noqa: BLE001
        print("WARNING: RapidOCR 不认线程参数, 用默认:", repr(e)[:120], flush=True)
        _ENG = RapidOCR()


MIN_W, MIN_ASPECT = 0.10, 4.0      # 长条框: 宽 >= 页宽 10% 且 >= 4 倍高


def _rect(q):
    return float(q[:, 0].min()), float(q[:, 1].min()), float(q[:, 0].max()), float(q[:, 1].max())


def _rec(img, quads):
    if not quads:
        return []
    res, _ = _ENG.text_rec([F._crop(img, q) for q in quads])
    return [str(r[0]) if float(r[1]) >= 0.5 else "" for r in res]      # 和 RapidOCR 的 text_score 0.5 一样


def _left_label(r, rects):
    """同一行、在它左边最近的那个框的下标(中心上下差 < 0.6 个框高), 没有返回 None"""
    x0, y0, x1, y1 = r
    cy, h = (y0 + y1) / 2, y1 - y0
    best = None
    for i, (a0, b0, a1, b1) in enumerate(rects):
        if a1 > x0 + 2 or (a0, b0, a1, b1) == r:
            continue
        if abs((b0 + b1) / 2 - cy) < 0.6 * max(h, b1 - b0):
            if best is None or a1 > rects[best][2]:
                best = i
    return best


def _no_time_digits(t):
    """去掉时间(带秒、不带秒)以后剩下的数字: 时间本身的 14 位数字不能当订单号"""
    t = F.TIME_RE.sub(" ", t)
    t = TIME_M.sub(" ", t)
    return F.digits_of(t)


def scan_one(path):
    t0 = time.time()
    try:
        import numpy as np
        import cv2
        img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return {"path": path, "err": "unreadable"}
        H, W = img.shape[:2]
        boxes, _ = _ENG.text_det(np.ascontiguousarray(img))
        quads = [np.asarray(b, np.float32) for b in ([] if boxes is None else boxes)]
        rects = [_rect(q) for q in quads]
        longs = [i for i, (x0, y0, x1, y1) in enumerate(rects) if (x1 - x0) >= MIN_W * W and (x1 - x0) >= MIN_ASPECT * max(y1 - y0, 1)]
        texts = dict(zip(longs, _rec(img, [quads[i] for i in longs])))
        # 有时间或长数字的行: 认它左边的标签框
        want = {}
        for i in longs:
            t = texts[i]
            if F.TIME_RE.search(t) or TIME_M.search(t) or len(_no_time_digits(t)) >= 14:
                j = _left_label(rects[i], rects)
                if j is not None:
                    want[i] = j
        need = sorted(set(j for j in want.values() if j not in texts))
        texts.update(zip(need, _rec(img, [quads[j] for j in need])))
        up = F.upload_time_from_name(path)
        times_s, times_m, nums, labels, lab_val = [], 0, [], set(), {}
        for i in longs:
            t = texts[i]
            lab = texts.get(want[i], "") if i in want else ""
            for k in LABEL_KEYS:
                if (k in t or k in lab) and not (k == "订单号" and ("商家订单号" in t or "商家订单号" in lab)):
                    labels.add(k)
            for m in F.TIME_RE.finditer(t):
                times_s.append((m.group(0), lab, F.parse_page_time(m.group(0))))
            times_m += len(TIME_M.findall(t))
            d = _no_time_digits(t)
            if len(d) >= 14:
                nums.append((lab, d))
            for k in ("支付时间", "创建时间", "订单号"):
                if k in lab_val:
                    continue
                if k in lab and not (k == "订单号" and "商家" in lab):
                    lab_val[k] = t
                elif t.startswith(k) and not (k == "订单号" and "商家" in t) and len(F.digits_of(t)) >= 8:
                    lab_val[k] = t
        lines = longs
        w1, w2 = [], []
        for s, lab, pt in times_s:
            if pt is None:
                continue
            if up is not None:
                d1 = (pt - up).total_seconds()
                if F.MIN_AHEAD <= d1 <= F.MAX_AHEAD:
                    w1.append(f"{s}<{lab}>d1={int(d1)}")
            for nlab, d in nums:
                ot, _fmt = F.order_no_time(d)
                if ot is not None:
                    d2 = (ot - pt).total_seconds()
                    if F.ORDER_MIN <= d2 <= F.ORDER_MAX:
                        w2.append(f"{d[:20]}<{nlab}>{s} d2={int(d2)}")
        r = F.check(lab_val.get("支付时间"), lab_val.get("创建时间"), lab_val.get("订单号"), up)
        return {"path": path, "err": "", "W": W, "H": H, "lines": len(lines), "up": F._ts(up),
                "labels": "/".join(sorted(labels)), "n_ts": len(times_s), "n_tm": times_m, "n_num": len(nums),
                "ts": " ; ".join(f"{s}<{lab}>" for s, lab, _ in times_s)[:300],
                "nums": " ; ".join(f"{d[:24]}<{lab}>" for lab, d in nums)[:300],
                "worst_r1": " ; ".join(w1)[:300], "worst_r2": " ; ".join(w2)[:300],
                "v": r["v"], "r1": int(r["r1"]), "r2": int(r["r2"]), "field": r["field"],
                "d1": "" if r["d1"] is None else int(r["d1"]), "d2": "" if r["d2"] is None else int(r["d2"]),
                "pay": (lab_val.get("支付时间") or "")[:40], "create": (lab_val.get("创建时间") or "")[:40],
                "order": (lab_val.get("订单号") or "")[:40], "sec": round(time.time() - t0, 2)}
    except Exception as e:  # noqa: BLE001
        return {"path": path, "err": repr(e)[:200], "sec": round(time.time() - t0, 2)}


COLS = ["path", "err", "W", "H", "lines", "up", "labels", "n_ts", "n_tm", "n_num", "ts", "nums", "worst_r1", "worst_r2",
        "v", "r1", "r2", "field", "d1", "d2", "pay", "create", "order", "sec"]


def _copy_tag(r):
    if r.get("err"):
        return None
    if r["worst_r1"]:
        return "W1"
    if r["worst_r2"]:
        return "W2"
    if r["v"] == 2:
        return "L2"
    if r["n_ts"]:
        return "TS"
    return None


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


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--since", default="")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1116)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-minutes", type=float, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--copy-to", default="")
    ap.add_argument("--max-copy", type=int, default=400)
    a = ap.parse_args()

    paths = _inputs(a.inputs)
    n_all = len(paths)
    if a.since:
        def day(p):
            m = UPLOAD_DAY.search(os.path.basename(p))
            return m.group(1) if m else ""
        paths = [p for p in paths if day(p) >= a.since]
    rng = random.Random(a.seed)
    rng.shuffle(paths)
    if a.sample:
        paths = paths[:a.sample]
    print(f"images found {n_all}, since {a.since or '-'}: {len(paths)} to scan", flush=True)
    try:
        import importlib.metadata as md
        print("rapidocr_onnxruntime", md.version("rapidocr_onnxruntime"), flush=True)
    except Exception:  # noqa: BLE001
        pass

    from multiprocessing import Pool
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    if a.copy_to:
        os.makedirs(a.copy_to, exist_ok=True)
    t0 = time.time()
    n = 0
    copied = 0
    stopped = False
    c = collections.Counter()
    block = max(64, a.workers * 32)
    with Pool(a.workers, initializer=_init_worker) as pool, open(a.out, "w", encoding="utf-8-sig", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        for i in range(0, len(paths), block):
            if a.max_minutes and (time.time() - t0) / 60 >= a.max_minutes:
                stopped = True
                break
            for r in pool.imap_unordered(scan_one, paths[i:i + block], chunksize=2):
                w.writerow(r)
                n += 1
                if r.get("err"):
                    c["err"] += 1
                    continue
                c["ts"] += r["n_ts"] > 0
                c["tm"] += r["n_tm"] > 0
                c["num"] += r["n_num"] > 0
                c["W1"] += bool(r["worst_r1"])
                c["W2"] += bool(r["worst_r2"])
                c["L2"] += r["v"] == 2
                c["v" + str(r["v"])] += 1
                for k in filter(None, r["labels"].split("/")):
                    c["label " + k] += 1
                tag = _copy_tag(r)
                if a.copy_to and tag and copied < a.max_copy:
                    shutil.copy2(r["path"], os.path.join(a.copy_to, tag + "_" + os.path.basename(r["path"])))
                    copied += 1
            fo.flush()
            el = time.time() - t0
            print(f"{n} / {len(paths)}  {n / max(el, 1e-9):.2f} pages/s  ts {c['ts']}  W1 {c['W1']}  W2 {c['W2']}  L2 {c['L2']}"
                  f"  err {c['err']}", flush=True)
    print()
    print(f"pages {n}  errors {c['err']}  copied {copied}" + ("  STOPPED BY --max-minutes" if stopped else ""))
    for k, v in sorted(c.items()):
        print(f"  {k}: {v}")
    print(f"FIELD_BLUE_DONE pages={n} minutes={(time.time() - t0) / 60:.1f}")


if __name__ == "__main__":
    main()
