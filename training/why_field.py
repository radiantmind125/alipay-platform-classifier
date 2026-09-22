"""某个字段忽然读不出来了, 查是**训练数据没了**还是**模型读错了**。

起因: `计入收支` 在整页验收里从 57% 掉到 37%(和天花板差 23 个点, 9.2 sigma),
而前四次都在 52~57 之间。白图的段级指标却几乎没动, 说明不是整体变差,
是这一个串出了事。

两种可能, 分开查:

    训练数据没了   这个词的段被 check_alignment 或 clean_labels 筛掉了
                  -> 看 _segments.csv 里还剩几条, bad 标了几条
    模型读错了     数据还在, 但模型把它读成了别的
                  -> 拿模型在这些段上跑一遍, 看输出成了什么

★ 先查第一种, 因为它不用跑模型, 一秒出结果。

用法
----
    python why_field.py --pairs D:\\alipay-ai-data\\pinyin-pairs --field 计入收支
    python why_field.py --pairs ... --field 计入收支 --onnx D:\\alipay-ai-data\\rec_pinyin.onnx
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--also", type=Path, nargs="*", default=[])
    ap.add_argument("--field", required=True)
    ap.add_argument("--onnx", type=Path, default=None,
                    help="给了就再拿模型在这些段上跑一遍, 看读成了什么")
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()

    print("=" * 58)
    print("  WHY FIELD  (ASCII only - safe to paste)")
    print("=" * 58)
    print(f"  field = {a.field}")
    print()

    hits = []
    for d in [a.pairs, *a.also]:
        man = d / "_segments.csv"
        if not man.exists():
            continue
        rows = list(csv.DictReader(man.open(encoding="utf-8-sig")))
        f = [r for r in rows if a.field in (r.get("text") or "")]
        bad = sum(1 for r in f if r.get("bad") == "1")
        exact = sum(1 for r in f if (r.get("text") or "").strip() == a.field)
        print(f"  {d.name}")
        print(f"    段总数        {len(rows):>8,}")
        print(f"    含这个词的     {len(f):>8,}")
        print(f"      其中整条就是它 {exact:>6,}")
        print(f"      被标成 bad 的 {bad:>6,}  ({bad/max(1,len(f)):.1%})"
              f"{'   <-- 训练时会被筛掉' if bad else ''}")
        print(f"    ★ 训练时实际留下 {len(f)-bad:>6,} 条")
        print()
        for r in f:
            if r.get("bad") != "1":
                r["_dir"] = str(d)
                hits.append(r)

    if not hits:
        print("  一条都没有 —— 训练集里根本没有这个词, 那当然读不出来")
        return

    # 这个词周围都跟着什么, 看有没有被别的串挤掉
    print("  含这个词的段, 完整文本长什么样(前 10 种):")
    for t, n in Counter(r["text"] for r in hits).most_common(10):
        print(f"    {n:>5}x  {t[:40]}")
    print()

    if not a.onnx:
        print("  (要看模型把它读成什么, 加 --onnx)")
        return

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import random

    import cv2
    import numpy as np
    from pinyin_ocr import Recognizer

    rec = Recognizer(None, a.onnx)
    random.seed(3)
    pick = random.sample(hits, min(a.n, len(hits)))
    crops, keep = [], []
    for r in pick:
        p = Path(r["_dir"]) / "seg_input" / r["file"]
        im = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
        if im is not None:
            crops.append(im)
            keep.append(r)
    if not crops:
        print("  图都读不到")
        return
    out = rec(crops)
    ok = sum(1 for r, t in zip(keep, out) if r["text"].strip() == t.strip())
    has = sum(1 for t in out if a.field in t)
    print(f"  拿模型在 {len(keep)} 条上跑:")
    print(f"    整条读对的        {ok:>4}  ({ok/len(keep):.0%})")
    print(f"    读出来含这个词的   {has:>4}  ({has/len(keep):.0%})"
          f"   <-- 整页验收数的就是这个")
    print()
    print("  读错的都读成了什么:")
    wrong = Counter(t for r, t in zip(keep, out) if a.field not in t)
    for t, n in wrong.most_common(12):
        print(f"    {n:>4}x  {t[:40]!r}")


if __name__ == "__main__":
    main()
