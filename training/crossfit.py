r"""交叉读: 训练页按 GWCZ 编号分成两半, 各训一个模型, 让它去读**另一半**的裁图 —— 拿到"没背过这条标签"的读法。

为什么(2026-10-14)
------------------
老师 OCR 的标签里还有不少错(名字、商家、卡号尾号、只标了一半的), 规则改不到。两套标准答案上量过:
"单独重读(recheck_segments 的 _recheck.csv) 和 模型 读得一样、却和标签不一样" 的段, 把标签换成模型读的,
42 条里 39 条从错变对、0 条从对变错(model-n1009e 读的是验证集, 没训过这些段)。
可训练集上的段模型都背过标签了, 拿它自己读训练集, 读出来就是标签, 一致不了 —— 所以要交叉:
    A 半的页训模型 A, 让 A 读 B 半; B 半的页训模型 B, 让 B 读 A 半。每一段都是被"没见过它"的模型读的。
同一张原图的段一定在同一半(按 GWCZ 编号分), 不会从同一页别的段里学到它。

用法
----
    python training/crossfit.py split --data E:\alipay-ai-data\train-n1009e --out-a E:\alipay-ai-data\train-cf-A --out-b E:\alipay-ai-data\train-cf-B
        -> 两个目录各有 train.txt(这一半的页)、val.txt 和 charset.txt(原样拷, 两个模型字表一样)
    python training/train_rec.py --data E:\alipay-ai-data\train-cf-A --epochs 16 --export E:\alipay-ai-data\model-cf-A\rec_pinyin.onnx
    python training/train_rec.py --data E:\alipay-ai-data\train-cf-B --epochs 16 --export E:\alipay-ai-data\model-cf-B\rec_pinyin.onnx
    python training/crossfit.py predict --onnx E:\alipay-ai-data\model-cf-A\rec_pinyin.onnx --list E:\alipay-ai-data\train-cf-B\train.txt --out cf_pred_B.csv
    python training/crossfit.py predict --onnx E:\alipay-ai-data\model-cf-B\rec_pinyin.onnx --list E:\alipay-ai-data\train-cf-A\train.txt --out cf_pred_A.csv
        -> csv: path, label, pred    (path 就是训练清单里的裁图路径)
★ 读的时候每个进程只开 1 个线程(OPENBLAS / OMP / MKL / onnxruntime 都是), 进程数用 --workers 定。
"""
from __future__ import annotations

import os
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")      # 必须在 import numpy / cv2 之前

import argparse  # noqa: E402
import csv  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
import zlib  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

# 和 train_rec.py 一模一样(这里不 import train_rec, 免得每个进程都去加载 torch)
IMG_H, MAX_W, MIN_W = 48, 640, 16
GW = re.compile(r"GWCZ\d+")


def fold_of(path: str) -> str:
    """按原图的 GWCZ 编号分两半; 没有编号的按文件名(去掉 _rNN_sNN)分。"""
    m = GW.search(path)
    key = m.group(0) if m else re.sub(r"_r\d+_s\d+\.\w+$", "", Path(path).name)
    return "A" if zlib.crc32(key.encode("utf-8")) % 2 == 0 else "B"


def cmd_split(a) -> None:
    lines = [ln for ln in (a.data / "train.txt").read_text(encoding="utf-8").splitlines() if "\t" in ln]
    parts = {"A": [], "B": []}
    for ln in lines:
        parts[fold_of(ln.split("\t", 1)[0])].append(ln)
    for k, out in (("A", a.out_a), ("B", a.out_b)):
        out.mkdir(parents=True, exist_ok=True)
        (out / "train.txt").write_text("\n".join(parts[k]) + "\n", encoding="utf-8")
        for f in ("val.txt", "charset.txt"):
            shutil.copy2(a.data / f, out / f)
        pages = {GW.search(ln).group(0) for ln in parts[k] if GW.search(ln)}
        print(f"  {k}: {len(parts[k]):,} 段   {len(pages):,} 页 -> {out}")
    pa = {GW.search(ln).group(0) for ln in parts["A"] if GW.search(ln)}
    pb = {GW.search(ln).group(0) for ln in parts["B"] if GW.search(ln)}
    print(f"  两半共有的页: {len(pa & pb)} (必须是 0)")


_SESS = None
_CHARS = None


def _init(onnx: str) -> None:
    global _SESS, _CHARS
    import onnxruntime as ort
    cv2.setNumThreads(1)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    _SESS = ort.InferenceSession(onnx, sess_options=so, providers=["CPUExecutionProvider"])
    _CHARS = [c for c in (Path(onnx).parent / "charset.txt").read_text(encoding="utf-8").split("\n") if c]


def _prep(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    nw = max(MIN_W, min(MAX_W, int(round(w * IMG_H / max(1, h)))))
    im = cv2.resize(img, (nw, IMG_H), interpolation=cv2.INTER_LINEAR)
    im = im.astype(np.float32) / 127.5 - 1.0
    return im.transpose(2, 0, 1)


def _decode(ids) -> str:
    out, prev = [], -1
    for i in ids:
        i = int(i)
        if i != prev and i != 0 and i <= len(_CHARS):
            out.append(_CHARS[i - 1])
        prev = i
    return "".join(out)


def _read(path: str) -> str:
    try:
        img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return ""
        x = _prep(img)[None].astype(np.float32)
        logits = _SESS.run(None, {"x": x})[0]
        ids = logits.argmax(2)[0]
        return _decode(ids[: max(1, x.shape[3] // 4)])
    except Exception:                  # noqa: BLE001
        return ""


def cmd_predict(a) -> None:
    rows = []
    for ln in a.list.read_text(encoding="utf-8").splitlines():
        if "\t" in ln:
            p, t = ln.split("\t", 1)
            rows.append((p, t))
    print(f"  读 {len(rows):,} 段   模型 {a.onnx}   进程 {a.workers}", flush=True)
    t0 = time.time()
    preds = []
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(str(a.onnx),)) as pool:
        for i, pr in enumerate(pool.map(_read, [p for p, _ in rows], chunksize=256), 1):
            preds.append(pr)
            if i % 50000 == 0:
                print(f"    {i:,}/{len(rows):,}   {i / max(1e-9, time.time() - t0):.0f} 段/秒", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "pred"])
        for (p, t), pr in zip(rows, preds):
            w.writerow([p, t, pr])
    same = sum(1 for (_p, t), pr in zip(rows, preds) if pr == t)
    print(f"  完 {time.time() - t0:.0f} 秒   读的和标签一字不差 {same / max(1, len(rows)):.1%}   -> {a.out}")


def main() -> None:
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split")
    s.add_argument("--data", type=Path, required=True)
    s.add_argument("--out-a", type=Path, required=True)
    s.add_argument("--out-b", type=Path, required=True)
    s = sub.add_parser("predict")
    s.add_argument("--onnx", type=Path, required=True)
    s.add_argument("--list", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    a = ap.parse_args()
    {"split": cmd_split, "predict": cmd_predict}[a.cmd](a)


if __name__ == "__main__":
    main()
