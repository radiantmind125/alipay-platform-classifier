"""量**拼音检测**本身准不准 —— 白图蓝图各一个数。

和"识别准不准"不是一回事
------------------------
    识别准不准   给一张裁好的图, 字读对没有        -> eval_rec 那边量
    检测准不准   拼音那一带**找没找出来**          -> 这里量

★★★★★ 为什么要单独量: 存图时发现 `订单号` 那 6 例里 **5 例 has_pinyin 报 False**,
   而图上 `dìng dān hào` 三个拼音清清楚楚。带声调的拼音更高, 正好卡在
   `annotation_labels` 小桶(<=0.55*big_h)和大桶(>=0.8*big_h)中间的空档里,
   两桶都不算, 于是检测不到。

   但 6 个样本说明不了面上有多少。**得有个全量的数。**

怎么量 —— 拿"拼音漏进 label 里"当判据
--------------------------------------
label 裁图是**特意把拼音裁在框外**的, 所以:

```
label 文字里还有成串的拉丁字母  ->  拼音没被裁掉  ->  这一行的拼音带**没检测出来**
```

★ 这个判据不用重跑模型, 也不用人工标注, 清单里现成就有。

★★ 它是**下界**: 拼音恰好没被 OCR 读成字母的那些漏不进来。
   所以真实的检测失败率只会比这个数**更高**, 不会更低。

用法
----
    python pinyin_detect_rate.py --pairs D:\\alipay-ai-data\\pinyin-pairs ^
        --also D:\\alipay-ai-data\\pinyin-pairs-blue2
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

# 4 个以上连着的拉丁字母才算 —— 少于 4 个可能是正文里的英文缩写
LATIN_RUN = re.compile(r"[a-zA-ZÀ-ɏ]{4,}")
# 带声调的那些字母。拼音检测就是栽在这批上, 单独数一下占多少
TONE = re.compile(r"[āáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜÀ-ɏ]")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--also", type=Path, nargs="*", default=[])
    ap.add_argument("--show", type=int, default=12)
    a = ap.parse_args()

    print("=" * 70)
    print("  PINYIN DETECT RATE  (ASCII only - safe to paste)")
    print("=" * 70)
    print("  判据: label 裁图**本该把拼音裁在框外**,")
    print("        里面还剩成串拉丁字母 = 这一行的拼音带没检测出来")
    print()

    for d in [a.pairs, *a.also]:
        man = d / "_pairs_labeled.csv"
        if not man.exists():
            print(f"  清单不在: {man}")
            continue
        rows = list(csv.DictReader(man.open(encoding="utf-8-sig")))
        done = [r for r in rows if (r.get("text") or "").strip()]
        leak, tone_leak = [], 0
        for r in done:
            t = r["text"]
            if LATIN_RUN.search(t):
                leak.append(r)
                if TONE.search(t):
                    tone_leak += 1
        n = max(1, len(done))
        print("-" * 70)
        print(f"  {d.name}")
        print(f"    清单总行数        {len(rows):>8,}")
        print(f"    读出文字的        {len(done):>8,}")
        print(f"    ★ 拼音漏进来的     {len(leak):>8,}  ({len(leak)/n:.2%})"
              f"   <- 检测失败率(下界)")
        print(f"      其中带声调的    {tone_leak:>8,}  "
              f"({tone_leak/max(1,len(leak)):.0%} of 漏的)")
        print(f"    ★ 检测成功率      {1-len(leak)/n:>8.2%}  (上界)")
        print()
        if leak:
            print(f"    漏进来长什么样(前 {a.show} 条):")
            for t, c in Counter(r["text"] for r in leak).most_common(a.show):
                print(f"      {c:>5}x  {t[:52]}")
        print()

    print("-" * 70)
    print("  ★★ 这是**下界**: 拼音没被 OCR 读成字母的那些漏不进来,")
    print("     所以真实的检测失败率只会比这个数更高。")
    print("  ★★ 带声调的占比高, 就坐实了'声调让拼音变高、卡进两个桶中间'那个解释。")


if __name__ == "__main__":
    main()
