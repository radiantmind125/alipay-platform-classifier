"""拿真实的段, 一边一边地多留一点, 看哪一种能把漏掉的字读回来。

为什么要这么试
--------------
`订单号` 被读成 `单号` 这 6 例, 已经确认:

    框够宽(114 像素, 一个字 39.8, 装得下 3 个字)
    存图看过, 订 单 号 三个字**都在框里**

★★★★★ 所以不是边界没框住, 是**模型没把框里的字读出来**。
   但"为什么没读出来"有好几个候选, 猜是猜不准的 ——
   这一程已经猜错过三次(蓝图摊薄、标签噪声、边界太窄)。

   **直接试**: 同一个位置, 左边多留一点 / 下边多留一点 / 上边多留一点,
   哪一种能把 `订` 读回来, 哪一种就是真因。

候选
----
    左边没留白     段的 x0 紧贴第一个字, 模型对贴边的字容易吞
    下边压到笔画   行边界取的是成员中位, 有勾有捺的字底下会被切掉
    上边拼音残缺   拼音带声调时更高, 逃过了拼音检测(has_pinyin 报 False),
                   于是上边界用的是**猜的** fallback, 可能把拼音切成半截,
                   半截拼音反而把底下的字带偏

★ 这三种都便宜(改切段时的留白), **都不用重训**。先确定是哪一种。

用法
----
    python probe_crop.py --dump D:\\alipay-ai-data\\dump_white_b.jsonl ^
        --src D:\\download2\\pinyin_hits --onnx D:\\alipay-ai-data\\rec_pinyin.onnx ^
        --field 订单号
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pinyin_ocr import Recognizer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--field", required=True)
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()

    recs = []
    with a.dump.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))

    fd = a.field
    # 挑出"读成了字段名真子串"的那些段
    cases = []
    for r in recs:
        if fd in (r.get("ours") or "").replace(" ", ""):
            continue
        for s in r.get("segs") or []:
            t = (s.get("t") or "").strip()
            if t and t != fd and t in fd and len(t) >= len(fd) - 2:
                cases.append((r, s))
                break
    cases = cases[:a.n]

    print("=" * 74)
    print("  PROBE CROP  (ASCII only - safe to paste)")
    print("=" * 74)
    print(f"  field = {fd}   样本 {len(cases)} 例")
    print()
    if not cases:
        print("  没找到样本")
        return

    # ★ 每种变体: (名字, 左, 右, 上, 下) 各多留多少像素
    VARIANTS = [
        ("原样",        0, 0, 0, 0),
        ("左+8",        8, 0, 0, 0),
        ("左+16",      16, 0, 0, 0),
        ("下+8",        0, 0, 0, 8),
        ("下+16",       0, 0, 0, 16),
        ("上+16",       0, 0, 16, 0),
        ("上+32",       0, 0, 32, 0),
        ("四边+8",      8, 8, 8, 8),
        ("四边+16",    16, 16, 16, 16),
    ]

    ok = {v[0]: 0 for v in VARIANTS}
    exact = {v[0]: 0 for v in VARIANTS}
    rec = Recognizer(None, a.onnx)
    detail = []

    for r, s in cases:
        p = a.src / r["file"]
        im = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
        if im is None:
            continue
        H, W = im.shape[:2]
        crops, names = [], []
        for nm, dl, dr, dt, db in VARIANTS:
            x0 = max(0, s["x0"] - dl)
            x1 = min(W, s["x1"] + dr)
            y0 = max(0, s["y0"] - dt)
            y1 = min(H, s["y1"] + db)
            crops.append(im[y0:y1, x0:x1])
            names.append(nm)
        outs = rec(crops)
        row = {"file": r["file"]}
        for nm, t in zip(names, outs):
            t = (t or "").strip()
            row[nm] = t
            if fd in t:
                ok[nm] += 1
            if t == fd:
                exact[nm] += 1
        detail.append(row)

    n = len(detail)
    print(f"  {'变体':<10}{'读出含字段名':>14}{'整条正好是它':>14}")
    print("  " + "-" * 40)
    for nm, *_ in VARIANTS:
        print(f"  {nm:<10}{ok[nm]:>8}/{n:<5}{exact[nm]:>8}/{n:<5}")
    print()
    print("  逐例看读成了什么:")
    for row in detail:
        print(f"    {row['file'][:46]}")
        for nm, *_ in VARIANTS:
            mark = "  <== 读回来了" if fd in row.get(nm, "") else ""
            print(f"      {nm:<10}{row.get(nm,'')[:28]!r}{mark}")
    print()
    print("  ★★★★★ 哪一边多留能读回来, 哪一边就是真因。")
    print("     都读不回来 -> 不是留白的事, 得从识别那边想。")


if __name__ == "__main__":
    main()
