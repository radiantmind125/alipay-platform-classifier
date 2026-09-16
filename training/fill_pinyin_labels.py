"""对 label/ 里的裁图跑 OCR, 把标签文字填进 _pairs.csv。

和裁图分成两步, 是因为 OCR 很慢(7159 张约 12 万行, 好几个小时),
分开之后重跑 OCR 不用重裁, 调 OCR 参数也不用动裁图。

质量过滤
--------
读出来的东西不是每条都能当标签, 这几种直接标成不可用:

    空的                   什么都没读出来
    纯拉丁                 说明裁行没裁干净, 拼音漏进来了
    太短                   一两个字符, 多半是噪声

★ 剩下的才写进 text 列。`usable` 列记这一条能不能用。

★★ 跑完**一定要人工核几十条**再往下走。这里量的是"读出来了没有",
   不是"读对了没有" —— 那个只有人眼能判。
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import cv2
import numpy as np

# 纯拉丁(拼音漏进来了)。排除带 @ 和数字的 —— 回单上本来就有邮箱和卡号
LATIN_ONLY = re.compile(r"^[a-zA-ZÀ-ɏ\s'·]+$")

# ★★★★★ 只挡"整条都是拉丁"不够。实测漏过这两条:
#     黄某人（德）备注 tongzhi 到账
#     Sueuz uanxiangqing quan buzhangdan 立 ?
#   它们**混着**汉字, 所以 LATIN_ONLY 判不中, 照样被当成可用标签。
#   拿这种去训, 等于教模型把拼音也读出来 —— 正好是要它别做的事。
#
#   所以再加一条: 文本里只要有**连续 4 个以上拉丁字母**的片段就不要。
#   4 这个数是看着实测样本定的: `tongzhi` 7 个, `uanxiangqing` 12 个;
#   而回单上正常的英文片段(卡组织缩写之类)一般不超过 3 个字母,
#   邮箱另外用 @ 放行。
LATIN_RUN = re.compile(r"[a-zA-ZÀ-ɏ]{4,}")
MIN_LEN = 2


def classify(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return "空"
    if len(s) < MIN_LEN:
        return "太短"
    if LATIN_ONLY.match(s) and "@" not in s:
        return "纯拉丁(拼音漏进来了)"
    if "@" not in s and LATIN_RUN.search(s):
        return "混着拼音"
    return "可用"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, required=True,
                    help="make_pinyin_pairs.py 出的目录")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    man = a.pairs / "_pairs.csv"
    if not man.exists():
        print(f"清单不在: {man}")
        return
    rows = list(csv.DictReader(man.open(encoding="utf-8-sig")))
    if a.limit:
        rows = rows[: a.limit]
    if not rows:
        print("清单是空的")
        return

    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        print("没装 rapidocr_onnxruntime:  pip install rapidocr_onnxruntime")
        return
    ocr = RapidOCR()

    label_dir = a.pairs / "label"
    print(f"要读 {len(rows):,} 条")

    tally: dict[str, int] = {}
    for i, r in enumerate(rows, 1):
        p = label_dir / r["file"]
        if not p.exists():
            r["text"], r["usable"] = "", "0"
            tally["图不在"] = tally.get("图不在", 0) + 1
            continue
        img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            r["text"], r["usable"] = "", "0"
            tally["读不了"] = tally.get("读不了", 0) + 1
            continue
        res, _ = ocr(img)
        # 一行可能读出多段(左标签 + 右取值), 按 x 排好拼起来
        parts = []
        for box, t, conf in (res or []):
            s = (t or "").strip()
            if s:
                parts.append((min(pt[0] for pt in box), s))
        parts.sort()
        text = " ".join(s for _, s in parts)

        kind = classify(text)
        tally[kind] = tally.get(kind, 0) + 1
        r["text"] = text
        r["usable"] = "1" if kind == "可用" else "0"

        if i % 500 == 0:
            print(f"  {i:,}/{len(rows):,}")

    out = a.pairs / "_pairs_labeled.csv"
    fields = list(rows[0].keys())
    if "usable" not in fields:
        fields.append("usable")
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    usable = sum(1 for r in rows if r.get("usable") == "1")
    py_usable = sum(1 for r in rows
                    if r.get("usable") == "1" and r.get("has_pinyin") == "1")
    print()
    print("=" * 60)
    print(f"读完 {len(rows):,} 条")
    print("=" * 60)
    for k, v in sorted(tally.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<24} {v:6,d}  ({v/len(rows):5.1%})")
    print()
    print(f"★ 可用 {usable:,} 条 ({usable/len(rows):.1%})")
    print(f"  其中上面有拼音的 {py_usable:,} 条 —— 这些才是真正要学的样本")
    print()
    print(f"填好的清单 -> {out}")
    print()
    print("★★ 下一步**必须**人工核几十条: 打开 input/ 的图, 对着 text 看读对了没有。")
    print("   这里只知道 读出来了, 不知道 读对了。")


if __name__ == "__main__":
    main()
