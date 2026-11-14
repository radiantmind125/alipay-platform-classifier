r"""账单详情页两个字段时间的检查 —— FieldTimeCheck.cs 的 Python 参考实现 + 服务器批量扫描(2026-11-14)。

两条规则(只看字段, 不看图)
--------------------------
第一条: 页面时间(支付时间, 读不出来再用创建时间)比凭证上传时间还晚 1~60 秒 -> 可疑。
    真图先付款、再截图、再上传, 页面时间一定在上传之前(本地 11 月真图 313 张, 上传最少晚 12 秒, 中位 40 秒)。
    假图: ArrowGapCheck 在服务器上抓到的 420 张里 112 张页面比上传晚 1~32 秒; 11-14 放过去的那张晚 25 秒。
    晚几分钟以上的多半是认错字(日期 / 小时)或时区不对, 不判。
第二条: 订单号前 14 位就是页面上的支付时间(年月日时分秒), 晚 0~15 秒 -> 可疑。
    支付宝自己的订单号是 日期 + 固定几位(收钱码 yyyyMMdd2300 14.. 28 位, 转账 yyyyMMdd2000 4001 1100.. 32 位,
    还有 2200 14..), 第 9~12 位是 2000 / 2100 / 2200 / 2300 的不看: 按时间读正好是整点, 整点付款的真图会撞上。
    窗口从 0 秒开始: 商家订单号常常是时间开头但都在支付之前, 字段取错也不会判。
上传时间按服务器图片文件名最后的 _yyyyMMddHHmmss 算。

和 training/demo/FieldTimeCheck.cs 逐步对应(check / parse_page_time / upload_time_from_name / order_no_time),
改一边要同步改另一边。--rules-jsonl 模式和 demo/fieldtimetest 读同一个 jsonl, 用来逐条对拍。

扫描时怎么读字段(只为了在服务器上验证规则, 不交出去)
------------------------------------------------------
RapidOCR 先找字框(宽过 1600 的整页缩到 1600 宽再找, 框再放大回原图), 只认左边 30% 宽里起头的框(标签列),
找到 支付时间 / 创建时间 / 订单号 / 商家订单号 的标签后, 再认同一行右边的框和正下方折行的数字。
值的拼法和 v1114 的 ana_orders.value_after 一样。一页只认二三十个框, 比整页全认快几倍。

用法
----
    python -u training/field_time_scan.py --from-ar E:/_move_logs/arrow_1114/ar.csv --sample 60000 --seed 1114 ^
        --workers 20 --max-minutes 100 --out E:/_move_logs/field_1114/ft.csv --copy-to E:/_move_logs/field_1114/look
    python -u training/field_time_scan.py 图或目录 [...] --out ft.csv            (本地试)
    python -I training/field_time_scan.py --rules-jsonl in.jsonl out.jsonl       (只跑规则, 和 C# 对拍)
    python -I training/field_time_scan.py --selftest
"""
from __future__ import annotations

import os
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")      # 必须在 import numpy 之前

import argparse  # noqa: E402
import calendar  # noqa: E402
import collections  # noqa: E402
import csv  # noqa: E402
import datetime  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

# ---------------------------------------------------------------- 规则(和 FieldTimeCheck.cs 一样)

MIN_AHEAD, MAX_AHEAD = 1, 60            # 第一条: 页面时间 - 上传时间 在这个范围(秒, 含两端)判可疑
ORDER_MIN, ORDER_MAX = 0, 15            # 第二条: 订单号时间 - 页面时间 在这个范围(秒, 含两端)判可疑
ALIPAY_FORMAT_CODES = ("2000", "2100", "2200", "2300")   # 订单号第 9~12 位: 支付宝自己的格式, 第二条不看

TIME_RE = re.compile(r"(20[0-9]{2})[-/.]([0-9]{1,2})[-/.]([0-9]{1,2})[ \t\u00a0\u3000]*"
                     r"([0-9]{1,2})[:\uff1a]([0-9]{2})[:\uff1a]([0-9]{2})")
UPLOAD_RE = re.compile(r"_([0-9]{14})\.")


def _make(y, mo, d, h, mi, s):
    if y < 2000 or y > 2099 or mo < 1 or mo > 12 or h > 23 or mi > 59 or s > 59:
        return None
    if d < 1 or d > calendar.monthrange(y, mo)[1]:
        return None
    return datetime.datetime(y, mo, d, h, mi, s)


def parse_page_time(text):
    """字段值里第一个 年-月-日 时:分:秒; 找不到、或者第一个不是合法时间, 返回 None"""
    if not text:
        return None
    m = TIME_RE.search(text)
    if not m:
        return None
    return _make(*(int(g) for g in m.groups()))


def _from14(d):
    return _make(int(d[0:4]), int(d[4:6]), int(d[6:8]), int(d[8:10]), int(d[10:12]), int(d[12:14]))


def upload_time_from_name(name):
    """文件名最后的 _yyyyMMddHHmmss. (带不带目录都行)"""
    if not name:
        return None
    base = name.replace("\\", "/").rsplit("/", 1)[-1]
    m = UPLOAD_RE.search(base)
    return _from14(m.group(1)) if m else None


def digits_of(s):
    return re.sub(r"[^0-9]", "", s or "")


def order_no_time(digits):
    """(订单号前 14 位的时间 或 None, 是不是支付宝自己的格式)"""
    if len(digits) < 14:
        return None, False
    if digits[8:12] in ALIPAY_FORMAT_CODES:
        return None, True
    return _from14(digits[:14]), False


def check(pay, create, order, upload):
    """upload: datetime(到秒)或 None。返回 dict, v = 0 判不了 / 1 没犯 / 2 可疑"""
    r = {"v": 0, "field": "", "pt": None, "up": upload, "d1": None, "digits": digits_of(order), "fmt": False,
         "ot": None, "d2": None, "r1": False, "r2": False}
    pt = parse_page_time(pay)
    if pt is not None:
        r["field"] = "支付时间"
    else:
        pt = parse_page_time(create)
        if pt is not None:
            r["field"] = "创建时间"
    r["pt"] = pt
    r["ot"], r["fmt"] = order_no_time(r["digits"])
    if pt is None:
        return r
    if upload is not None:
        r["d1"] = (pt - upload).total_seconds()
        r["r1"] = MIN_AHEAD <= r["d1"] <= MAX_AHEAD
    if r["ot"] is not None:
        r["d2"] = (r["ot"] - pt).total_seconds()
        r["r2"] = ORDER_MIN <= r["d2"] <= ORDER_MAX
    if r["r1"] or r["r2"]:
        r["v"] = 2
    elif upload is None and len(r["digits"]) < 14:
        r["v"] = 0
    else:
        r["v"] = 1
    return r


def _ts(t):
    return t.strftime("%Y%m%d%H%M%S") if t is not None else ""


def rules_jsonl(src, dst):
    """对拍: 每行 {id, pay, create, order, upload(14 位或空), name}; upload 空就从 name 取"""
    n = 0
    with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8", newline="\n") as g:
        for line in f:
            if not line.strip():
                continue
            x = json.loads(line)
            up = _from14(x["upload"]) if x.get("upload") else upload_time_from_name(x.get("name") or "")
            r = check(x.get("pay"), x.get("create"), x.get("order"), up)
            g.write(json.dumps({"id": x["id"], "v": r["v"], "r1": r["r1"], "r2": r["r2"], "field": r["field"],
                                "pt": _ts(r["pt"]), "up": _ts(r["up"]),
                                "d1": None if r["d1"] is None else int(r["d1"]), "digits": r["digits"],
                                "fmt": r["fmt"], "ot": _ts(r["ot"]),
                                "d2": None if r["d2"] is None else int(r["d2"])}, ensure_ascii=False) + "\n")
            n += 1
    print("RULES_JSONL_DONE", n)


# 固定用例: 和 fieldtimetest --selftest 一样; (支付时间, 创建时间, 订单号, 上传 14 位或文件名, 期望 v, 说明)
SELFTEST = [
    (None, "2026-11-14 04:35:45", "20261114200040011100059255558788", "20261114043520", 2, "11-14 假图: 创建时间比上传晚 25 秒"),
    ("2026-11-01 04:45:24", None, "20261101044526584798398267229281", "x_20261101044657.jpg", 2, "11 月箭头判不出的假图: 订单号 = 支付时间 + 2 秒"),
    ("2026-11-04 07:28:01", None, "20261104072802950852632894448512", "20261104072932", 2, "同上 + 1 秒"),
    ("2026-11-03 22:45:06", None, "2026110323001442801444266253", "20261103224543", 1, "真图收钱码: 2300 格式不看, 上传晚 37 秒"),
    ("2026-11-03 23:00:10", None, "2026110323001442801444266253", "20261103230100", 1, "整点撞上 23:00:14 的真图, 2300 格式不看"),
    ("2026-11-01 20:00:38", None, "20261101200040011100390084455147", "20261101200110", 1, "转账 2000 格式撞 20:00:40, 不看"),
    ("2026-11-03 22:20:16", None, "20261103221950013887", "20261103222100", 1, "商家订单号误传: 支付前 26 秒, 不判"),
    ("2026-11-03 22:20:16", None, "", "20261103222016", 1, "上传和页面同一秒: 不判"),
    ("2026-11-03 22:20:17", None, "", "20261103222016", 2, "页面比上传晚 1 秒"),
    ("2026-11-03 22:21:16", None, "", "20261103222016", 2, "晚 60 秒"),
    ("2026-11-03 22:21:17", None, "", "20261103222016", 1, "晚 61 秒: 不判(多半认错字)"),
    ("2026-11-03 22:20:16", None, "", "20261103142016", 1, "晚 8 小时(上传传成 UTC): 不判"),
    ("2026-11-03 22:20:16", None, "20261103222031999", None, 2, "订单号晚 15 秒"),
    ("2026-11-03 22:20:16", None, "20261103222032999", None, 1, "晚 16 秒: 不判"),
    ("2026-11-03 22:20:16", None, "20261103222015999", None, 1, "早 1 秒: 不判"),
    ("2026-11-03 22:20", None, "20261103222016999", "20261103222000", 0, "页面时间没有秒: 判不了"),
    (None, None, "20261103222016999", "20261103222000", 0, "没有页面时间: 判不了"),
    ("2026-11-03 22:20:16", None, "123", None, 0, "没上传时间, 订单号太短: 判不了"),
    ("支付时间2026-11-0322:20:16", None, "订单号 2026 1103 2220 18 77", None, 2, "带标签、丢空格、订单号带空格"),
    ("2026/11/03 22：20：16", None, "", "20261103222020", 1, "斜杠日期 全角冒号, 上传晚 4 秒"),
    ("2026-02-30 22:20:16", "2026-11-03 22:20:16", "", "20261103222010", 2, "支付时间不合法 -> 用创建时间"),
    ("2026-11-03 22:20:16", None, "２０２６１１０３２２２０１８", None, 0, "全角数字不认"),
    ("2026-11-03 24:00:00", None, "", "20261103222010", 0, "24 点不合法"),
    ("2026-11-03 22:20:16", None, "", "a/b_20261103222016.jpg", 1, "从文件名取上传时间"),
]


def selftest():
    bad = 0
    for pay, create, order, upl, want, why in SELFTEST:
        up = None
        if upl:
            up = _from14(upl) if re.fullmatch(r"[0-9]{14}", upl) else upload_time_from_name(upl)
        r = check(pay, create, order, up)
        ok = r["v"] == want
        bad += not ok
        print(("PASS " if ok else "FAIL ") + f"v={r['v']} want={want} d1={r['d1']} d2={r['d2']}  {why}")
    print("FIELD_SELFTEST_OK" if bad == 0 else f"FIELD_SELFTEST_FAILED {bad}")
    return 0 if bad == 0 else 1


# ---------------------------------------------------------------- 读字段(扫描用)

DET_W = 1600         # 找字框时整页缩到这个宽(736 在拼音页上把拼音和标签连成一个框, 页面时间丢一半)
LEFT_X = 0.30        # 标签列: 框的左边在页面宽 30% 以内
LABELS = (("pay", "支付时间", None), ("create", "创建时间", None),
          ("order", "订单号", "商家"), ("merchant", "商家订单号", None))
_ENG = None


def _engine():
    global _ENG
    if _ENG is None:
        from rapidocr_onnxruntime import RapidOCR
        try:        # 只直接调 text_det / text_rec, 方向分类不用; 单线程(多进程并行)
            _ENG = RapidOCR(intra_op_num_threads=1, inter_op_num_threads=1)
        except Exception as e:  # noqa: BLE001  老版本不认这两个参数
            print("RapidOCR thread options not accepted, using defaults:", repr(e)[:120], flush=True)
            _ENG = RapidOCR()
        if not (hasattr(_ENG, "text_det") and hasattr(_ENG, "text_rec")):
            raise RuntimeError("rapidocr_onnxruntime 版本不对: 没有 text_det / text_rec")
    return _ENG


def _crop(img, q):
    import cv2
    import numpy as np
    w = int(max(np.linalg.norm(q[0] - q[1]), np.linalg.norm(q[2] - q[3])))
    h = int(max(np.linalg.norm(q[0] - q[3]), np.linalg.norm(q[1] - q[2])))
    w, h = max(w, 1), max(h, 1)
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    M = cv2.getPerspectiveTransform(q.astype(np.float32), dst)
    out = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)
    if out.shape[0] / out.shape[1] >= 1.5:
        out = np.rot90(out)
    return out


def read_fields(img):
    """返回 {pay, create, order, merchant: 值的文字(没找到是 ""), boxes: 框数, recs: 认了几个框}"""
    import cv2
    import numpy as np
    eng = _engine()
    H, W = img.shape[:2]
    out = {k: "" for k, _l, _e in LABELS}
    out["boxes"] = out["recs"] = 0
    s = DET_W / W if W > DET_W else 1.0
    small = cv2.resize(img, (max(1, round(W * s)), max(1, round(H * s))), interpolation=cv2.INTER_AREA) if s < 1.0 else img
    boxes, _ = eng.text_det(small)
    if boxes is None or len(boxes) == 0:
        return out
    quads = [np.asarray(b, np.float32) / s for b in boxes]
    rects = [(float(q[:, 0].min()), float(q[:, 1].min()), float(q[:, 0].max()), float(q[:, 1].max())) for q in quads]
    order = sorted(range(len(quads)), key=lambda i: (rects[i][1], rects[i][0]))
    out["boxes"] = len(quads)
    texts = {}

    def rec(idxs):
        idxs = [i for i in dict.fromkeys(idxs) if i not in texts]
        if not idxs:
            return
        res, _ = eng.text_rec([_crop(img, quads[i]) for i in idxs])
        for i, r in zip(idxs, res):
            texts[i] = str(r[0]) if float(r[1]) >= 0.5 else ""      # 和 RapidOCR 的 text_score 0.5 一样
        out["recs"] += len(idxs)

    rec([i for i in order if rects[i][0] < LEFT_X * W])

    found = {}
    for key, lab, excl in LABELS:
        for i in order:
            t = texts.get(i, "").replace(" ", "")
            if lab in t and not (excl and excl in t):
                found[key] = i
                break

    plan = {}
    need = []
    for key, li in found.items():
        x0, y0, x1, y1 = rects[li]
        h = y1 - y0
        cy = (y0 + y1) / 2
        row = sorted((j for j in order if j != li and abs((rects[j][1] + rects[j][3]) / 2 - cy) < 0.6 * h
                      and rects[j][0] > x0), key=lambda j: rects[j][0])
        wrap = []
        if row:
            xv = rects[row[0]][0]
            yb = max(rects[j][3] for j in row)
            wrap = [j for j in order if j not in row and j != li and 0 < rects[j][1] - yb < 1.2 * h
                    and abs(rects[j][0] - xv) < 3 * h]
        plan[key] = (li, row, wrap)
        need += row + wrap
    rec(need)

    for key, lab, _e in LABELS:
        if key not in plan:
            continue
        li, row, wrap = plan[key]
        t = texts.get(li, "").replace(" ", "")
        txt = t.split(lab, 1)[1] + "".join(texts.get(j, "").replace(" ", "") for j in row)
        for j in wrap:
            w = texts.get(j, "").replace(" ", "")
            if re.fullmatch(r"\d{1,6}", w):
                txt += w
        out[key] = txt
    return out


def _imread(path):
    import cv2
    import numpy as np
    buf = np.fromfile(path, np.uint8)
    im = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if im is None and buf.size > 2 and buf[0] == 0xFF and buf[1] == 0xD8:      # 截断的 JPEG 补上结尾再解
        im = cv2.imdecode(np.concatenate([buf, np.array([0xFF, 0xD9], np.uint8)]), cv2.IMREAD_COLOR)
    return im


def _init_worker():
    import cv2
    cv2.setNumThreads(1)
    _engine()


def scan_one(task):
    path, extra = task
    t0 = time.time()
    row = {"path": path, **extra}
    try:
        im = _imread(path)
        if im is None:
            row["err"] = "unreadable"
            return row
        row["W"], row["H"] = int(im.shape[1]), int(im.shape[0])
        f = read_fields(im)
        row.update(f)
        r = check(f["pay"], f["create"], f["order"], upload_time_from_name(path))
        row.update({"v": r["v"], "r1": int(r["r1"]), "r2": int(r["r2"]), "field": r["field"], "pt": _ts(r["pt"]),
                    "up": _ts(r["up"]), "d1": "" if r["d1"] is None else int(r["d1"]),
                    "o14": r["digits"][:14], "olen": len(r["digits"]), "fmt": int(r["fmt"]),
                    "d2": "" if r["d2"] is None else int(r["d2"])})
    except Exception as e:  # noqa: BLE001
        row["err"] = repr(e)[:200]
    row["sec"] = round(time.time() - t0, 2)
    return row


COLS = ["path", "arrow", "ratio", "gap_w", "W", "H", "v", "r1", "r2", "field", "pt", "up", "d1", "o14", "olen", "fmt",
        "d2", "pay", "create", "order", "merchant", "boxes", "recs", "sec", "err"]


def _copy_tag(row):
    """要拷出来看的页: 两条犯了的全拷; 贴线的(上传只比页面晚 0~11 秒、页面晚 61~600 秒、订单号差 -60~-1 / 16~120 秒)也拷"""
    if row.get("v") == 2:
        return ("R1" if row["r1"] else "") + ("R2" if row["r2"] else "")
    d1, d2 = row.get("d1"), row.get("d2")
    if d1 != "" and d1 is not None and -11 <= d1 <= 0:
        return "NEAR1"
    if d1 != "" and d1 is not None and 61 <= d1 <= 600:
        return "WIDE1"
    if d2 != "" and d2 is not None and (-60 <= d2 <= -1 or 16 <= d2 <= 120):
        return "WIDE2"
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="*")
    ap.add_argument("--from-ar", help="arrow_scan.py 的 ar.csv: 只扫里面的页(都是有账单管理那几行的账单详情)")
    ap.add_argument("--sample", type=int, default=0, help="随机抽这么多页(ar 里判可疑的全要), 0 = 全部")
    ap.add_argument("--seed", type=int, default=1114)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-minutes", type=float, default=0, help="跑到这么多分钟就不再派新页, 已派的跑完照常收尾")
    ap.add_argument("--out", help="结果 csv")
    ap.add_argument("--copy-to", help="两条犯了的页和贴线的页拷到这里")
    ap.add_argument("--max-copy", type=int, default=1500)
    ap.add_argument("--rules-jsonl", nargs=2, metavar=("IN", "OUT"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if a.selftest:
        return selftest()
    if a.rules_jsonl:
        rules_jsonl(*a.rules_jsonl)
        return 0
    if not a.out:
        ap.error("要给 --out")
    if selftest() != 0:
        return 3

    tasks = []
    if a.from_ar:
        with open(a.from_ar, encoding="utf-8-sig") as f:
            ar = list(csv.DictReader(f))
        susp = [r for r in ar if r["verdict"] == "Suspicious"]
        rest = [r for r in ar if r["verdict"] != "Suspicious"]
        rng = random.Random(a.seed)
        if a.sample and a.sample < len(ar):
            rest = rng.sample(rest, max(0, a.sample - len(susp)))
        rng.shuffle(susp)
        rng.shuffle(rest)
        pick = susp + rest      # 判可疑的先跑(几百张), 其余打乱: 到时间停下来也是随机样本
        tasks = [(r["path"], {"arrow": r["verdict"], "ratio": r["ratio"], "gap_w": r["gap_w"]}) for r in pick]
        print(f"ar.csv {len(ar)} pages, Suspicious {len(susp)}, scanning {len(tasks)}", flush=True)
    for p in a.inputs:
        if p.startswith("@"):
            with open(p[1:], encoding="utf-8-sig") as f:
                tasks += [(x.strip(), {}) for x in f if x.strip()]
        elif os.path.isfile(p):
            tasks.append((p, {}))
        else:
            for dp, _d, fs in os.walk(p):
                tasks += [(os.path.join(dp, x), {}) for x in sorted(fs) if x.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not tasks:
        print("NO INPUT")
        return 2
    try:
        _engine()
    except Exception as e:  # noqa: BLE001
        print("OCR ENGINE FAILED", repr(e)[:300])
        return 3
    try:
        import importlib.metadata as _md
        print("rapidocr_onnxruntime", _md.version("rapidocr-onnxruntime"), flush=True)
    except Exception:  # noqa: BLE001  只是打印版本, 拿不到不要紧
        pass

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    if a.copy_to:
        os.makedirs(a.copy_to, exist_ok=True)
    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    done = 0
    copied = 0
    stats = collections.Counter()
    d1s, d2s = collections.Counter(), collections.Counter()
    stopped = False
    block = max(64, a.workers * 32)
    with open(a.out, "w", encoding="utf-8-sig", newline="") as fo, \
            ProcessPoolExecutor(a.workers, initializer=_init_worker) as ex:
        w = csv.DictWriter(fo, COLS, extrasaction="ignore")
        w.writeheader()
        for b0 in range(0, len(tasks), block):
            if a.max_minutes and (time.time() - t0) / 60 > a.max_minutes:
                stopped = True
                break
            for row in ex.map(scan_one, tasks[b0:b0 + block], chunksize=4):
                w.writerow(row)
                done += 1
                if row.get("err"):
                    stats["err"] += 1
                    continue
                arrow = row.get("arrow", "-")
                stats["pages"] += 1
                stats["pt"] += bool(row.get("pt"))
                stats["olen14"] += row.get("olen", 0) >= 14
                stats["fmt"] += row.get("fmt", 0)
                stats[f"v{row['v']}"] += 1
                stats[f"v{row['v']}|{arrow}"] += 1
                if row["r1"]:
                    stats[f"r1|{arrow}"] += 1
                if row["r2"]:
                    stats[f"r2|{arrow}"] += 1
                if row["d1"] != "" and row["d1"] >= -30:
                    d1s[row["d1"] if row["d1"] <= 120 else (600 if row["d1"] <= 600 else 99999)] += 1
                if row["d2"] != "" and -120 <= row["d2"] <= 120:
                    d2s[row["d2"]] += 1
                tag = _copy_tag(row)
                if tag and a.copy_to and copied < a.max_copy:
                    try:
                        shutil.copy2(row["path"], os.path.join(a.copy_to, f"{tag}_{arrow}_{os.path.basename(row['path'])}"))
                        copied += 1
                    except OSError as e:
                        print("COPY FAILED", row["path"], e)
            fo.flush()
            el = time.time() - t0
            rate = done / el if el else 0
            print(f"{done} / {len(tasks)}  {rate:.1f} pages/s  ETA {(len(tasks) - done) / rate / 60 if rate else 0:.0f} min  "
                  f"r1 {sum(v for k, v in stats.items() if k.startswith('r1|'))}  "
                  f"r2 {sum(v for k, v in stats.items() if k.startswith('r2|'))}  err {stats['err']}", flush=True)

    print()
    print(f"pages {stats['pages']}  errors {stats['err']}  with page time {stats['pt']}  order no >= 14 digits "
          f"{stats['olen14']} (Alipay format {stats['fmt']})  copied {copied}" + ("  STOPPED BY --max-minutes" if stopped else ""))
    for k in sorted(stats):
        if "|" in k or k.startswith("v"):
            print(f"  {k}: {stats[k]}")
    print("page minus upload (s), -30..120 each, 600 = 121..600, 99999 = later:", sorted(d1s.items()))
    print("order-no time minus page time (s), -120..120:", sorted(d2s.items()))
    print(f"FIELD_SCAN_DONE pages={done} minutes={(time.time() - t0) / 60:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
