"""整页分类小模型(只看图, 不做 OCR)的数据: 解码 -> 两种缩略图 -> 缓存 + 名单(manifest.jsonl)。

标签用 page_type.py(规则 + OCR)跑出来、人眼核过的结果(census / 拼音库的 page_type.csv)。
解码和缩放跟 page_type.py 一模一样: cv2.imdecode(截断的 JPEG 补 FFD9 再解), cv2.resize INTER_AREA ——
C# 用 OpenCvSharp 的 ImDecode + Resize(Area) 能做出逐像素一样的图(scratchpad csprobe 实测), 以后移植不会差。
  page 视图: 整页拉伸到 PAGE_W x PAGE_H(宽 x 高)
  head 视图: 上面 HEAD_FRAC 那一段拉伸到 HEAD_W x HEAD_H(蓝图的标题、收款方 / 银行卡那一行都在这一段)
缓存存 JPEG q95(训练时本来就加 JPEG 压缩的扰动); ★ 以后推理和 C# 对拍要用原图现做的缩略图, 不能用缓存。

分组(防止同一张图训练和测试两边都有):
  先按 GWCZ 编号去重(census 优先); 再把缩略图一模一样的(同一张图换了编号重复上传, 池里约 1.9%)并成一组,
  按组里第一个编号的 md5 分:
    census 行(从凭证池随机抽的, 和整池比例一样): 30% test、20% val、其余 train
    其它名单(拼音库等, 不代表整池): 10% test_py、10% val、其余 train
rule 列: 只看图的规则会给的类别(凭证原图 W2、翻拍 P、浮窗 X、页头异常图标 B4、账单详情卡片 / 时间轴 W1), 空 = 交给模型。
  快速分类 = 规则先出手, 剩下的交给模型; 评估主要看 rule 为空的那些, 再看规则 + 模型合起来。
--gold(可选, 只在本地用): 人眼核过的名单(列 path, truth), 标签换成人眼定的。★ 名单里是真实凭证编号, 不能进仓库。
用法:
  python training/page_cnn_data.py --csv census.csv:census pinyin.csv:pinyin --out 缓存目录 [--workers 20] [--limit N]
输出: 缓存目录\\page\\<id>.jpg, head\\<id>.jpg, manifest.jsonl, build_summary.txt
"""
from __future__ import annotations

import os

for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")      # 服务器 24 核: 每个进程只开一个线程, 不然吃满提交内存

import argparse  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

PAGE_W, PAGE_H = 288, 640
HEAD_FRAC = 0.40
HEAD_W, HEAD_H = 448, 384
JPEG_Q = 95
GW = re.compile(r"GWCZ\d+")
VOUCHER_W, VOUCHER_H = 1304, (1950, 2450)       # 和 page_type.py 一样
BLUE_ABNORMAL = ("到账失败", "提交成功", "等待对方领取")


def decode(path: str):
    """和 page_type.decode 一样: 截断的 JPEG(少了结尾的 FFD9)补上再解一次。"""
    buf = np.fromfile(path, np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None
    if bgr is None and buf.size > 2 and buf[0] == 0xFF and buf[1] == 0xD8:
        bgr = cv2.imdecode(np.concatenate([buf, np.array([0xFF, 0xD9], np.uint8)]), cv2.IMREAD_COLOR)
    return bgr


def head_view(bgr):
    """head 视图: 上面 round(HEAD_FRAC * H) 行拉伸到 HEAD_W x HEAD_H, INTER_AREA。训练、快速分类、C# 都必须这样做。"""
    hh = max(1, int(round(HEAD_FRAC * bgr.shape[0])))
    return cv2.resize(bgr[:hh], (HEAD_W, HEAD_H), interpolation=cv2.INTER_AREA)


def views(bgr):
    """一张原图 -> (page, head) 两个 uint8 BGR 缩略图。以后推理(Python / C#)必须用同一个做法。"""
    page = cv2.resize(bgr, (PAGE_W, PAGE_H), interpolation=cv2.INTER_AREA)
    return page, head_view(bgr)


def gid(path: str) -> str:
    m = GW.search(path)
    return m.group() if m else os.path.splitext(os.path.basename(path))[0]


def bucket(key: str) -> int:
    return int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16) % 100


def split_of(key: str, set_name: str) -> str:
    b = bucket(key)
    if set_name == "census":
        return "test" if b < 30 else "val" if b < 50 else "train"
    return "test_py" if b < 10 else "val" if b < 20 else "train"


def rule_label(r: dict) -> str:
    """只看图的规则(page_type 里不用 OCR 的那几步)会给的类别, 不管对不对; 空 = 规则定不下来, 交给模型。
    顺序和 page_type 一样: 凭证原图尺寸 -> 翻拍照片 -> 浮窗 -> 页头异常图标 -> 账单详情卡片 / 时间轴。
    ★ 按"规则会不会出手"算, 不按标签: 卡片 A 的 支付消息 列表、嵌了账单的平台页, 规则照样判 W1, 那是规则的错。"""
    card = r.get("card", "") or ""
    if r.get("W") == VOUCHER_W and VOUCHER_H[0] <= (r.get("H") or 0) <= VOUCHER_H[1]:
        return "W2"
    if r["cls"] == "P":                       # P 只从只看图的翻拍规则来
        return "P"
    if r.get("sub") == "浮窗":                 # X/浮窗 只从只看图的浮窗规则来
        return "X"
    if r.get("state") in BLUE_ABNORMAL:
        return "B4"
    if card.startswith("A") or "T" in card:   # card 列: A / A0 / O + T(时间轴) + R<n>(箭头行)
        return "W1"
    return ""


def _init() -> None:
    cv2.setNumThreads(1)


def _work(task):
    i, path, out = task
    try:
        bgr = decode(path)
        if bgr is None:
            return i, "读不出", 0, 0, ""
        page, head = views(bgr)
        pj = cv2.imencode(".jpg", page, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])[1]
        pj.tofile(os.path.join(out, "page", f"{i}.jpg"))
        cv2.imencode(".jpg", head, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])[1].tofile(os.path.join(out, "head", f"{i}.jpg"))
        return i, "", bgr.shape[1], bgr.shape[0], hashlib.md5(page.tobytes()).hexdigest()
    except Exception as e:  # noqa: BLE001
        return i, repr(e)[:80], 0, 0, ""


def parse_spec(spec: str) -> tuple:
    """'路径.csv:集合名' -> (路径, 集合名); 没写集合名的用文件所在目录名(路径里的盘符冒号不算)。"""
    fn, sep, name = spec.rpartition(":")
    if not sep or not fn or "\\" in name or "/" in name or name.lower().endswith(".csv"):
        return spec, Path(spec).parent.name
    return fn, name


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="整页分类小模型的缩略图缓存 + 名单")
    ap.add_argument("--csv", nargs="+", required=True, help="page_type.csv:集合名(census 当整池的随机样本)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gold", type=Path, default=None, help="(本地用)人眼核过的名单 csv(列 path, truth)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--limit", type=int, default=0, help="只做前 N 张(本地试跑)")
    a = ap.parse_args()

    gold = {}
    if a.gold:
        with a.gold.open(encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                gold[gid(r["path"])] = r["truth"]
    rows, seen, dup = [], set(), 0
    for spec in a.csv:
        fn, name = parse_spec(spec)
        with open(fn, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                k = gid(r["path"])
                if k in seen:
                    dup += 1
                    continue
                seen.add(k)
                rows.append({"id": len(rows), "gid": k, "path": r["path"], "set": name, "cls": r["cls"], "sub": r.get("sub", ""),
                             "state": r.get("state", ""), "pinyin": r.get("pinyin", ""), "card": r.get("card", ""),
                             "header": r.get("header", ""), "dark": r.get("dark", ""), "reason": (r.get("reason") or "")[:80],
                             "truth": gold.get(k, "")})
    if a.limit:
        rows = rows[:a.limit]
    for d in ("page", "head"):
        (a.out / d).mkdir(parents=True, exist_ok=True)
    print(f"  名单 {len(rows):,} 张(同一 GWCZ 重复跳过 {dup:,})  进程 {a.workers}", flush=True)
    t0 = time.time()
    res = {}
    # ★ 用 ProcessPoolExecutor: 子进程死了会报 BrokenProcessPool、退出码非 0; multiprocessing.Pool 会一直等(10-02 拼音扫描卡死过)
    with ProcessPoolExecutor(a.workers, initializer=_init) as ex:
        for n, (i, err, w, h, md5) in enumerate(ex.map(_work, [(r["id"], r["path"], str(a.out)) for r in rows], chunksize=8), 1):
            res[i] = (err, w, h, md5)
            if n % 2000 == 0 or n == len(rows):
                print(f"    {n:,}/{len(rows):,}  {n / max(time.time() - t0, 1e-9):.1f} 张/秒", flush=True)
    # 缩略图一模一样的并成一组(同一张图换了编号重复上传), 整组跟着组里第一张分
    first = {}
    for r in rows:
        err, w, h, md5 = res[r["id"]]
        r["W"], r["H"], r["err"] = w, h, err
        if not err:
            first.setdefault(md5, r)
            g = first[md5]
            r["group"] = g["gid"]
            r["split"] = split_of(g["gid"], g["set"])
        else:
            r["group"], r["split"] = r["gid"], "none"
        r["rule"] = rule_label(r)
    bad = sum(1 for r in rows if r["err"])
    with (a.out / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    good = [r for r in rows if not r["err"]]
    ngrp = Counter(r["group"] for r in good)
    lab = lambda r: r["truth"].split("/")[0] if r["truth"] else r["cls"]  # noqa: E731
    lines = [f"缓存 {a.out}  {len(rows):,} 张, 读不出 / 出错 {bad:,}, 用时 {time.time() - t0:.0f} 秒",
             f"视图 page {PAGE_W}x{PAGE_H}, head 上面 {HEAD_FRAC:.0%} -> {HEAD_W}x{HEAD_H}, JPEG q{JPEG_Q}",
             f"缩略图一模一样、编号不同的: {sum(v - 1 for v in ngrp.values() if v > 1):,} 张并进了别的组",
             f"只看图的规则会出手的 {sum(1 for r in good if r['rule']):,} 张(" + "  ".join(f"{k} {v:,}" for k, v in Counter(r['rule'] for r in good if r['rule']).most_common())
             + f"), 规则定不下来、交给模型的 {sum(1 for r in good if not r['rule']):,} 张", ""]
    for sp in ("train", "val", "test", "test_py"):
        for rl, nm in ((False, "交给模型"), (True, "规则出手")):
            c = Counter(lab(r) for r in good if r["split"] == sp and bool(r["rule"]) == rl)
            lines.append(f"  {sp:8} {nm} {sum(c.values()):>7,}   " + "  ".join(f"{k} {v:,}" for k, v in sorted(c.items(), key=lambda x: -x[1])))
    if gold:
        g = Counter(r["split"] for r in good if r["truth"])
        lines.append(f"  人眼核过的(gold) {sum(g.values()):,} 张: " + "  ".join(f"{k} {v:,}" for k, v in g.most_common()))
    (a.out / "build_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    if bad > 0.01 * len(rows):
        print("  ★ 读不出的超过 1%, 停下来看看", flush=True)
        sys.exit(3)


if __name__ == "__main__":
    main()
