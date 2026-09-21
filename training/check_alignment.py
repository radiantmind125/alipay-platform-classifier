"""查切段有没有**错配** —— 图是这一段, 文字却是另一段的。

为什么怀疑有错配
----------------
白蓝合训之后, 每一轮打出来的那三条样例都是这样:

    真: 22:24:10                      出: 免费
    真: 玉                            出: 立
    真: 开通记账本，自动生成标签统计报告   出: >

★★★★★ 这**不是"差一点"**, 是**毫不相干**。而且二十轮都没学会, 别的样本却到了 90%。
   模型学不会毫不相干的东西 —— 说明**图和文字根本不是一对**。

   还有一条旁证: `玉`(1 个字) 和 `开通记账本...`(18 个字) 出现在同一个
   按宽度排的批次里。两者宽度本该差很多, 挤在一起就说明至少有一个配错了。

错配是怎么来的
--------------
split_segments 是**按顺序**把几何切出来的段和文字段配对的:
段数一样就认为第 i 段对第 i 段。段数一样**不代表顺序一样** ——
图上有图标、徽章的时候, OCR 读出来的段和几何切出来的段可能错位。

怎么查 —— 拿宽度和字数对不上来判
--------------------------------
一段文字有几个字, 它的图就该有多宽, 两者大体成正比。
所以看 `宽度 / 字数` 这个比值: 绝大多数样本会挤在一个范围里,
**离谱的那些多半是配错了**。

★ 这个判据不需要人工标注, 也不需要再跑 OCR。

用法
----
    python check_alignment.py --pairs 配对目录
    python check_alignment.py --pairs 配对目录 --drop      # 把离谱的标出来
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np

HAN = re.compile(r"[一-鿿]")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--also", type=Path, nargs="*", default=[])
    ap.add_argument("--lo", type=float, default=0.25,
                    help="宽字比低于中位的这个倍数算离谱")
    ap.add_argument("--hi", type=float, default=4.0)
    ap.add_argument("--drop", action="store_true",
                    help="把离谱的行在 _segments.csv 里标成 bad=1, 训练时可筛掉")
    a = ap.parse_args()

    for d in [a.pairs, *a.also]:
        man = d / "_segments.csv"
        if not man.exists():
            print(f"  清单不在: {man}")
            continue
        rows = list(csv.DictReader(man.open(encoding="utf-8-sig")))
        # 段的宽度清单里就有(x0/x1), 不用打开图
        w, n, keep = [], [], []
        for r in rows:
            t = (r.get("text") or "").strip()
            if not t:
                continue
            width = int(r["x1"]) - int(r["x0"])
            # 汉字按 1 个宽度单位, 半角的按 0.5 —— 不然一串数字会被误判成太宽
            units = sum(1.0 if HAN.match(c) else 0.5 for c in t)
            if units <= 0:
                continue
            w.append(width / units)
            n.append(len(t))
            keep.append(r)
        arr = np.array(w)
        med = float(np.median(arr))
        lo, hi = med * a.lo, med * a.hi
        bad = (arr < lo) | (arr > hi)
        print("=" * 58)
        print(f"  {d.name}")
        print("=" * 58)
        print(f"  段 {len(arr):,}   宽/字 中位 {med:.1f}px   "
              f"正常区间 {lo:.1f} ~ {hi:.1f}")
        print(f"  ★ 离谱的 {int(bad.sum()):,}  ({bad.mean():.2%})")
        print(f"    太窄(图小字多) {int((arr < lo).sum()):,}   "
              f"太宽(图大字少) {int((arr > hi).sum()):,}")
        print()
        idx = np.argsort(arr)
        print("  最窄的几条(图小字多, 多半是图配错成了短段):")
        for i in idx[:5]:
            r = keep[i]
            print(f"    {arr[i]:>6.1f}px/字  宽 {int(r['x1'])-int(r['x0']):>4}  "
                  f"{r['text'][:34]}")
        print("  最宽的几条(图大字少):")
        for i in idx[-5:]:
            r = keep[i]
            print(f"    {arr[i]:>6.1f}px/字  宽 {int(r['x1'])-int(r['x0']):>4}  "
                  f"{r['text'][:34]}")
        print()

        if a.drop:
            fields = list(rows[0].keys())
            if "bad" not in fields:
                fields.append("bad")
            flag = {id(r): "1" for r, b in zip(keep, bad) if b}
            for r in rows:
                r["bad"] = flag.get(id(r), "0")
            bak = man.with_suffix(".csv.bak2")
            if not bak.exists():
                bak.write_bytes(man.read_bytes())
            with man.open("w", encoding="utf-8-sig", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                wr.writeheader()
                wr.writerows(rows)
            print(f"  标好了 bad 列 -> {man.name}  (原件备份 {bak.name})")


if __name__ == "__main__":
    main()
