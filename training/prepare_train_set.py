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
    ap.add_argument("--both-views", action="store_true",
                    help="同一行出两条样本: input/(带拼音) 和 label/(不带拼音), "
                         "文字相同。★ 见下面注释, 这是个要用实验定的选择, 不是显然更好")
    ap.add_argument("--segments", action="store_true",
                    help="★★★★★ 用 split_segments.py 切好的**段**, 而不是整行。"
                         "识别器的单位是段不是行 —— 整行 26:1, 段 3.8:1")
    ap.add_argument("--also", type=Path, nargs="*", default=[],
                    help="★ 再并进来几个配对目录(比如蓝图那一份), 一起出一份训练集。"
                         "每一份的图路径按**它自己的**目录解析, 不会指错地方")
    ap.add_argument("--cap", type=int, default=0,
                    help="★ 每个配对目录最多取这么多段(按原图整张取, 不打散)。"
                         "拿来做对照实验: 把白图砍到和蓝图一样多, 再看白图能到多少 —— "
                         "这样才分得清蓝图弱是**数据少**还是**本来就难**")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    name = "_segments.csv" if a.segments else "_pairs_labeled.csv"
    rows = []
    for d in [a.pairs, *a.also]:
        man = d / name
        if not man.exists():
            print(f"清单不在: {man}")
            print("   先跑 " + ("split_segments.py" if a.segments else "fill_pinyin_labels.py"))
            return
        got = list(csv.DictReader(man.open(encoding="utf-8-sig")))
        # ★★ 每一行记住自己是从哪个目录来的。并多份的时候, 图路径必须按
        #    **各自的**目录解析 —— 统一按第一个目录拼, 后面那份的图就全指错了,
        #    而且不会报错, 只会训出一堆读不到的图。
        for r in got:
            r["_dir"] = str(d)
        if a.cap and len(got) > a.cap:
            # ★ 按**原图**整张取, 不按段随机取。按段取会把同一张图的段
            #   分到训练和验证两边, 验证集就等于在考背过的题。
            bysrc = {}
            for r in got:
                bysrc.setdefault(r["source"], []).append(r)
            srcs = sorted(bysrc)
            random.Random(a.seed).shuffle(srcs)
            picked, n = [], 0
            for sname in srcs:
                if n >= a.cap:
                    break
                picked.extend(bysrc[sname])
                n += len(bysrc[sname])
            n_src = len({r["source"] for r in picked})
            print(f"  {d.name}: {len(got):,} 行  ->  砍到 {len(picked):,} 行"
                  f"  (原图 {len(bysrc):,} 张里取了 {n_src:,} 张)")
            got = picked
        else:
            print(f"  {d.name}: {len(got):,} 行")
        rows.extend(got)
    if not rows:
        print("清单是空的")
        return

    out_dir = a.out or a.pairs
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 筛 ----------
    # ★ 段清单里没有 usable 列 —— 它本来就是从可用的行里切出来的
    keep = [r for r in rows
            if (a.segments or r.get("usable") == "1") and (r.get("text") or "").strip()]
    # ★★★ check_alignment.py --drop 标出来的错配行, 这里筛掉。
    #   错配是"图是这一段、文字是另一段的" —— 这种样本模型**永远学不会**,
    #   只会往梯度里灌噪声。实测白图上占 1.6%, 而且正是训练里那几条
    #   二十轮都学不会的(20 像素宽的图配了 18 个字的文字)。
    n_bad = sum(1 for r in keep if r.get("bad") == "1")
    if n_bad:
        keep = [r for r in keep if r.get("bad") != "1"]
        print(f"  筛掉错配的 {n_bad:,} 条 (check_alignment 标的)")
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
    #
    # ★★★★★ 为什么有 --both-views 这个选项 —— 训练和上线时看到的图**不一样**。
    #
    #   训练时我们喂的是 input/: 上面**整条**拼音都在。
    #   上线时模型拿到的是经理那条管线的检测器画的框 —— 实测那个检测器
    #   97% 的行都框得到, 也就是说它框的是**汉字那一行**, 顶多蹭到一点
    #   拼音的下伸笔画, 不会把整条拼音带框进去。
    #
    #   于是有个风险: 模型在训练里学会了"上面那一截不用看", 上线时拿到
    #   一个贴着汉字裁的紧框, 它可能把汉字的上半截也当成"不用看的那一截"。
    #
    #   --both-views 的办法: 同一行出两条样本, 文字一样, 图一张带拼音一张不带。
    #   等于告诉模型"有没有拼音你都得读对", 而不是"永远忽略上面那截"。
    #
    # ★★ 但这**不是显然更好**, 所以默认关着:
    #     label/ 那张图正是老师用来读出这条文字的图, 拿它当样本接近同义反复,
    #     容易让模型在简单样本上刷分, 冲淡真正要学的那一半。
    #   两种都训一遍, 在验证集上比字准确率, 用数说话。
    sub_i = "seg_input" if a.segments else "input"
    sub_l = "seg_label" if a.segments else "label"

    def dump(rs, name):
        p = out_dir / name
        with p.open("w", encoding="utf-8", newline="\n") as f:
            for r in rs:
                base = Path(r.get("_dir") or a.pairs)
                f.write(f"{base / sub_i / r['file']}\t{r['text']}\n")
                if a.both_views:
                    f.write(f"{base / sub_l / r['file']}\t{r['text']}\n")
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
