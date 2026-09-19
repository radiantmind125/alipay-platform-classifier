"""把训练标签里**老师自己读错的**那些字纠回去, 然后重训。

为什么要做
----------
400 张实测里唯一还明显丢字段的是 `计入收支`(92.1%), 读成了 `计人收支`。
本来以为是模型分不清 入 和 人。**查下来不是**:

    计入收支   标签里出现 4284 次
    计人收支   标签里出现  191 次      <- 老师读错的, 进了训练集

★★★★★ **模型不是自己认错, 是照着错的标签学的。** 老师(现成 OCR 读不含拼音的
   裁图)读错了 191 次, 这 191 条就成了"正确答案"。

再一扫, 这种成对的错有 **1135 组, 涉及 5870 条标签**:

    413x 支付奕励 -> 支付奖励      184x 金部账单 -> 全部账单
    191x 计人收支 -> 计入收支      159x 支付时简 -> 支付时间
     87x 账单洋情 -> 账单详情       86x 粽签     -> 标签

判据和那个**必须挡住的坑**
--------------------------
判据: 两个词**一样长、只差一个字**, 高频的比低频的多 N 倍 -> 低频那个多半是读错的。

★★★★★ 但光这样会**改坏真数据**。扫出来的里面有这么一组:

    70x 立即领取4积分  ->  1108x 立即领取1积分

   这两个**根本不是一回事** —— 4 积分和 1 积分是两笔不同的奖励。
   照着改就把真数据改没了。

★★ 所以加一条硬闸: **不一样的那个字必须是汉字**。数字、字母、标点不一样的一律不动。
   金额、积分、卡号、时间这些差一个数字都是天壤之别, 碰不得。

★ 还有一条: 低频那个本身不能太多。一个词出现几百次还说它是错的, 那多半是我判错了。

用法
----
    python clean_labels.py --pairs 配对目录                  # 只看要改什么
    python clean_labels.py --pairs 配对目录 --apply          # 真改
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

HAN = re.compile(r"[一-鿿]")
RATIO = 10          # 高频要比低频多这么多倍
MAX_RARE = 500      # 低频那个超过这么多次就不当它是错的
MIN_LEN = 2


def load_counts(pairs: Path) -> Counter:
    lex: Counter = Counter()
    man = pairs / "_pairs_labeled.csv"
    for r in csv.DictReader(man.open(encoding="utf-8-sig")):
        if r.get("usable") != "1":
            continue
        for t in (r.get("text") or "").split(" "):
            t = t.strip()
            if len(t) >= MIN_LEN and HAN.search(t):
                lex[t] += 1
    return lex


def find_pairs(lex: Counter, ratio: int, max_rare: int) -> dict:
    """返回 {读错的: 正确的}。"""
    buckets = defaultdict(list)
    for w, n in lex.items():
        if MIN_LEN <= len(w) <= 12:
            for i in range(len(w)):
                buckets[(len(w), i, w[:i] + "\0" + w[i + 1:])].append((w, n, i))
    fix = {}
    for _k, v in buckets.items():
        if len(v) < 2:
            continue
        v.sort(key=lambda t: -t[1])
        top, tn, pos = v[0]
        for w, n, _i in v[1:]:
            if n > max_rare or tn < ratio * max(1, n):
                continue
            # ★★★★★ 硬闸: 不一样的那个字必须**两边都是汉字**。
            #   数字不一样的绝对不能动 —— 立即领取4积分 和 立即领取1积分
            #   是两笔不同的奖励, 不是读错。
            if not (HAN.match(w[pos]) and HAN.match(top[pos])):
                continue
            # 同一个错词可能匹配到多个"正确词", 留最高频那个
            if w not in fix or lex[fix[w]] < tn:
                fix[w] = top
    return fix


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--ratio", type=int, default=RATIO)
    ap.add_argument("--max-rare", type=int, default=MAX_RARE)
    ap.add_argument("--apply", action="store_true",
                    help="真改 _segments.csv。不给就只看不动")
    ap.add_argument("--top", type=int, default=20)
    a = ap.parse_args()

    lex = load_counts(a.pairs)
    fix = find_pairs(lex, a.ratio, a.max_rare)
    inst = sum(lex[w] for w in fix)
    print("=" * 58)
    print("  CLEAN LABELS  (ASCII only - safe to paste)")
    print("=" * 58)
    print(f"  词表 {len(lex):,}")
    print(f"  判成老师读错的 {len(fix):,} 个词,  涉及标签实例 {inst:,} 条")
    print()
    print(f"  {'次数':>7}  {'读错的':<14} ->  {'正确的'}")
    for w in sorted(fix, key=lambda w: -lex[w])[: a.top]:
        print(f"  {lex[w]:>7}  {w:<14} ->  {fix[w]}")
    print()
    # 数字那条闸拦下了多少 —— 报出来让人看见它确实在干活
    loose = find_pairs(lex, a.ratio, a.max_rare)
    blocked = []
    buckets = defaultdict(list)
    for w, n in lex.items():
        if MIN_LEN <= len(w) <= 12:
            for i in range(len(w)):
                buckets[(len(w), i, w[:i] + "\0" + w[i + 1:])].append((w, n, i))
    for _k, v in buckets.items():
        if len(v) < 2:
            continue
        v.sort(key=lambda t: -t[1])
        top, tn, pos = v[0]
        for w, n, _i in v[1:]:
            if n > a.max_rare or tn < a.ratio * max(1, n):
                continue
            if not (HAN.match(w[pos]) and HAN.match(top[pos])):
                blocked.append((w, top, n))
    print(f"  ★ 被'必须是汉字'那条闸拦下的 {len(blocked):,} 组 —— 这些**不能改**:")
    for w, top, n in sorted(blocked, key=lambda t: -t[2])[:8]:
        print(f"      {n:>5}x  {w:<18} vs  {top}")
    print()

    seg = a.pairs / "_segments.csv"
    if not a.apply:
        print(f"  (只看没动。要真改加 --apply, 会改写 {seg.name})")
        return
    if not seg.exists():
        print(f"  {seg} 不在, 先跑 split_segments.py")
        return
    rows = list(csv.DictReader(seg.open(encoding="utf-8-sig")))
    n_ch = 0
    for r in rows:
        t = (r.get("text") or "").strip()
        if t in fix:
            r["text"] = fix[t]
            n_ch += 1
    bak = seg.with_suffix(".csv.bak")
    if not bak.exists():
        bak.write_bytes(seg.read_bytes())
        print(f"  原件备份 -> {bak.name}")
    with seg.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  改了 {n_ch:,} 条段标签  ({n_ch/max(1,len(rows)):.1%})")
    print("  ★ 接着重跑 prepare_train_set.py 和 train_rec.py")


if __name__ == "__main__":
    main()
