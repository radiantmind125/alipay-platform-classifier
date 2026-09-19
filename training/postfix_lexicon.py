"""拿训练标签里的词表给识别结果做**后处理纠错**。

为什么要这一步
--------------
400 张实测里, 真正丢掉的字段一共 28 个, 其中 19 个是同一个毛病:

    计入收支  ->  计人收支        入 和 人 长得太像

字段匹配是整串比对, **错一个字整条就没了**。这一项因此只有 92.1%。
重训未必能治(入和人本来就难分), 但这类错有个特点:

★★★★★ **错出来的东西不是词。** `计人收支` 在 12.7 万行标签里一次都没出现过,
   而 `计入收支` 出现了几千次。既然如此, 见到不是词的、又和某个高频词只差一个字的,
   就把它纠回去。

怎么做才不会帮倒忙
------------------
★★ 纠错最怕**过度纠正** —— 把本来对的改坏。所以三道闸:

    1. 只纠**不在词表里**的 token。在词表里的一律不动。
    2. 候选词必须**足够高频**(min-freq), 不然拿一个偶然出现过一次的怪词去纠, 更糟。
    3. 候选必须**唯一**。编辑距离 1 的候选有两个以上就放着不动 ——
       宁可不纠, 不可纠错。

★ 而且长度必须一样(只换字, 不增删) —— 增删字的错多半是切图切坏了, 不是认错字,
  那种纠不回来。

用法
----
    python postfix_lexicon.py --pairs 配对目录 --dump e2e 出的 jsonl   # 量效果
    python postfix_lexicon.py --pairs 配对目录 --save lexicon.json      # 存词表
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

HAN = re.compile(r"[一-鿿]")
MIN_LEN = 2          # 一个字的 token 不纠, 没有上下文
MIN_FREQ = 20        # 候选词至少出现这么多次


def build_lexicon(pairs: Path) -> Counter:
    """从标签里数词频。一行里空格分开的每一段算一个 token。"""
    man = pairs / "_pairs_labeled.csv"
    lex: Counter = Counter()
    for r in csv.DictReader(man.open(encoding="utf-8-sig")):
        if r.get("usable") != "1":
            continue
        for t in (r.get("text") or "").split(" "):
            t = t.strip()
            if len(t) >= MIN_LEN and HAN.search(t):
                lex[t] += 1
    return lex


def build_index(lex: Counter, min_freq: int) -> dict:
    """按 (长度, 挖掉第 i 个字) 建索引, 这样找编辑距离 1 的候选是 O(len)。"""
    idx: dict = {}
    for w, n in lex.items():
        if n < min_freq:
            continue
        for i in range(len(w)):
            idx.setdefault((len(w), i, w[:i] + "\0" + w[i + 1:]), []).append(w)
    return idx


def correct_token(tok: str, lex: Counter, idx: dict, min_freq: int):
    """返回 (纠正后, 是否动过)。不该动就原样返回。"""
    if len(tok) < MIN_LEN or not HAN.search(tok):
        return tok, False
    if tok in lex:                       # ★ 闸一: 本来就是词, 不动
        return tok, False
    cands = set()
    for i in range(len(tok)):
        for w in idx.get((len(tok), i, tok[:i] + "\0" + tok[i + 1:]), ()):
            if w != tok:
                cands.add(w)
    if len(cands) != 1:                  # ★ 闸三: 候选不唯一就不动
        return tok, False
    w = cands.pop()
    if lex[w] < min_freq:                # ★ 闸二: 候选不够高频不动
        return tok, False
    return w, True


def fix_text(text: str, lex, idx, min_freq) -> tuple[str, int]:
    out, n = [], 0
    for t in (text or "").split(" "):
        f, ch = correct_token(t, lex, idx, min_freq)
        out.append(f)
        n += ch
    return " ".join(out), n


FIELDS_15 = ["账单详情", "全部账单", "创建时间", "付款方式", "订单号", "转账备注",
             "收款方", "账户余额", "计入收支", "转账成功", "交易成功", "对方账户",
             "商家名称", "支付时间", "交易号"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--dump", type=Path, default=None, help="给了就量效果")
    ap.add_argument("--save", type=Path, default=None)
    ap.add_argument("--min-freq", type=int, default=MIN_FREQ)
    a = ap.parse_args()

    lex = build_lexicon(a.pairs)
    idx = build_index(lex, a.min_freq)
    print("=" * 58)
    print("  POSTFIX LEXICON  (ASCII only - safe to paste)")
    print("=" * 58)
    print(f"  词表 {len(lex):,} 个词, 其中出现 >= {a.min_freq} 次的 "
          f"{sum(1 for v in lex.values() if v >= a.min_freq):,}")

    if a.save:
        a.save.write_text(json.dumps(
            {w: n for w, n in lex.items() if n >= a.min_freq},
            ensure_ascii=False), encoding="utf-8")
        print(f"  -> {a.save}")

    if not a.dump:
        return

    rows = [json.loads(l) for l in a.dump.read_text(encoding="utf-8").splitlines() if l.strip()]
    n_fix = 0
    before = after = ceil_tot = 0
    gained, lost = [], []
    for r in rows:
        fixed, k = fix_text(r["ours"], lex, idx, a.min_freq)
        n_fix += k
        b = sum(1 for f in FIELDS_15 if f in r["ours"])
        af = sum(1 for f in FIELDS_15 if f in fixed)
        c = sum(1 for f in FIELDS_15 if f in r["ceiling"])
        before += b
        after += af
        ceil_tot += c
        for f in FIELDS_15:
            if f not in r["ours"] and f in fixed:
                gained.append(f)
            if f in r["ours"] and f not in fixed:
                lost.append(f)

    print()
    print(f"  {len(rows)} 张图, 改了 {n_fix:,} 个 token")
    print(f"    纠错前字段总数 {before:,}")
    print(f"    纠错后字段总数 {after:,}   ({after-before:+,})")
    print(f"    天花板         {ceil_tot:,}")
    print()
    print(f"  ★ 救回来的字段 {len(gained)}:  {dict(Counter(gained))}")
    print(f"  ★★ 弄丢的字段 {len(lost)}:  {dict(Counter(lost))}"
          f"   {'(一个都没弄丢)' if not lost else '<-- 帮倒忙了, 要收紧'}")


if __name__ == "__main__":
    main()
