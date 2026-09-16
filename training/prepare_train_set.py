"""把 _pairs_labeled.csv 整理成能直接喂给识别器的训练集。

出三样东西:

    train.txt     每行  图路径 <TAB> 文字
    val.txt       同上
    charset.txt   字表, 一行一个字

★★★★★ 划分训练/验证**按原图划, 不按行划**。
   同一张截图裁出来的十几行, 版面、字体、底色、清晰度全都一样,
   按行随机划会把同一张图的行分到两边 —— 验证集里全是训练集见过的图,
   验证分数虚高, 拿不准模型到底会不会读**没见过的**截图。
   这是做数据集最容易踩、也最难发现的一个坑: 分数好看, 上线拉胯。

★★ 生僻字单独挑出来报。一个字在整个训练集里只出现一两次, 模型学不会它,
   还会把 CTC 的输出空间撑大。报出来让人决定要不要连带那几行一起丢。

用法
----
    python prepare_train_set.py --pairs D:\\alipay-ai-data\\pinyin-pairs
    python prepare_train_set.py --pairs ... --min-char-freq 5 --val-ratio 0.05
"""
from __future__ import annotations

import argparse
import csv
import random
import re
from collections import Counter
from pathlib import Path

HAN = re.compile(r"[\u4e00-\u9fff]")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None,
                    help="不给就写在 --pairs 底下")
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--min-char-freq", type=int, default=3,
                    help="出现次数少于这个的字算生僻, 只报不删")
    ap.add_argument("--pinyin-only", action="store_true",
                    help="只要上面有拼音的行。★ 默认**两种都要** —— "
                         "拼音页上本来就两种行都有, 只训带拼音的, 训出来只会读半页")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    man = a.pairs / "_pairs_labeled.csv"
    if not man.exists():
        print(f"清单不在: {man}")
        print("   先跑 fill_pinyin_labels.py")
        return
    rows = list(csv.DictReader(man.open(encoding="utf-8-sig")))
    if not rows:
        print("清单是空的")
        return

    out_dir = a.out or a.pairs
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 筛 ----------
    keep = [r for r in rows if r.get("usable") == "1" and (r.get("text") or "").strip()]
    if a.pinyin_only:
        keep = [r for r in keep if r.get("has_pinyin") == "1"]
    if not keep:
        print("筛完一条不剩")
        return

    n_py = sum(1 for r in keep if r.get("has_pinyin") == "1")
    print("=" * 60)
    print(f"  清单 {len(rows):,} 行  ->  能用的 {len(keep):,} 行 ({len(keep)/len(rows):.0%})")
    print(f"  其中上面有拼音的 {n_py:,} ({n_py/len(keep):.0%}), 没拼音的 {len(keep)-n_py:,}")
    print("=" * 60)

    # ---------- 按**原图**划分 ----------
    # ★★★★★ 关键在这。按行划会把同一张截图的行分到训练和验证两边,
    #   那验证集等于在考已经背过的题。
    srcs = sorted({r["source"] for r in keep})
    random.seed(a.seed)
    random.shuffle(srcs)
    n_val = max(1, int(len(srcs) * a.val_ratio))
    val_srcs = set(srcs[:n_val])
    train = [r for r in keep if r["source"] not in val_srcs]
    val = [r for r in keep if r["source"] in val_srcs]
    print(f"  原图 {len(srcs):,} 张  ->  训练 {len(srcs)-n_val:,} 张 / 验证 {n_val:,} 张")
    print(f"  行数              训练 {len(train):,} 行 / 验证 {len(val):,} 行")
    print("  ★ 按原图划的, 验证集里的截图训练时一张都没见过")
    print()

    # ---------- 字表 ----------
    freq = Counter()
    for r in keep:
        freq.update(r["text"])
    rare = [c for c, n in freq.items() if n < a.min_char_freq]
    chars = sorted(c for c in freq if not c.isspace())
    han_chars = [c for c in chars if HAN.match(c)]
    print(f"  字表 {len(chars):,} 个字  (其中汉字 {len(han_chars):,})")
    print(f"  出现少于 {a.min_char_freq} 次的生僻字 {len(rare):,} 个")
    if rare:
        hit = sum(1 for r in keep if any(c in rare for c in r["text"]))
        print(f"    带生僻字的行 {hit:,} ({hit/len(keep):.1%}) "
              f"—— 要不要连行一起丢, 自己定")
        print(f"    前 30 个: {''.join(sorted(rare)[:30])}")
    print()

    lens = sorted(len(r["text"]) for r in keep)
    print(f"  标签长度  中位 {lens[len(lens)//2]}  "
          f"90% 分位 {lens[int(len(lens)*0.9)]}  最长 {lens[-1]}")
    print()

    # ---------- 写盘 ----------
    img_dir = a.pairs / "input"     # ★ 输入用 input/ —— **带拼音**的那套
    def dump(rs, name):
        p = out_dir / name
        with p.open("w", encoding="utf-8", newline="\n") as f:
            for r in rs:
                f.write(f"{img_dir / r['file']}\t{r['text']}\n")
        return p

    pt, pv = dump(train, "train.txt"), dump(val, "val.txt")
    pc = out_dir / "charset.txt"
    pc.write_text("\n".join(chars) + "\n", encoding="utf-8")

    print(f"  {pt}")
    print(f"  {pv}")
    print(f"  {pc}")
    print()
    print("★★★ 图用的是 input/ —— **带拼音**那套。标签来自 label/(不含拼音)。")
    print("   反了就训成普通 OCR 了, 白训。")


if __name__ == "__main__":
    main()
