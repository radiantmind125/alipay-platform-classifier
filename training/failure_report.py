"""把端到端跑出来的东西按**失败类型**归一归, 看清楚还差在哪。

输入是 e2e_bench.py --dump 出来的 jsonl, 每行一张图:

    ours      我们这条链读出来的全部文字
    base      现成 OCR 整页读出来的
    ceiling   这张图每行 label 裁图(拼音在框外)读出来的 —— 当参照
    segs      我们切出来的每一段, 带坐标和 has_pinyin

怎么判对错
----------
★★ 没有人工真值, 所以拿 **ceiling 当参照**: 它是现成 OCR 在**同一张图**、
   拼音裁在框外时读出来的。不是完美真值(它自己也有 5% 左右的错),
   但它和我们看的是同一张图、同样的行, 是手头最公平的参照。

★ 因此下面说的"丢了"要这么读: **ceiling 读到了而我们没读到**。
  反过来"多出来"里有一部分其实是我们读对了而 ceiling 读错了 ——
  这一类之前人工核过, 约占错误的 15%。

按什么分类
----------
    掩码姓名    带 * 的            张*明  **勇（个人）
    邮箱        带 @ 的
    纯数字      订单号 金额 时间
    汉字        其余

★ 分开看才知道该往哪使劲: 要是丢的全是邮箱, 那是一类很窄的问题,
  和"汉字读不准"完全不是一回事, 对经理那边的字段抽取影响也完全不同。
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

HAN = re.compile(r"[一-鿿]")
LATIN = re.compile(r"[a-zA-Z]{2,}")

FIELDS_15 = ["账单详情", "全部账单", "创建时间", "付款方式", "订单号", "转账备注",
             "收款方", "账户余额", "计入收支", "转账成功", "交易成功", "对方账户",
             "商家名称", "支付时间", "交易号"]


def kind(tok: str) -> str:
    if "@" in tok:
        return "邮箱"
    if "*" in tok:
        return "掩码"
    if HAN.search(tok):
        return "汉字"
    if re.fullmatch(r"[\d\-:.,/ ￥¥]+", tok):
        return "纯数字"
    return "其他"


def toks(text: str) -> list[str]:
    return [t for t in re.split(r"\s+", text or "") if t]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--top", type=int, default=14)
    a = ap.parse_args()

    rows = [json.loads(ln) for ln in a.dump.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not rows:
        print("dump 是空的")
        return

    print("=" * 60)
    print("  FAILURE REPORT  (ASCII only - safe to paste)")
    print("=" * 60)
    print(f"  images {len(rows):,}")
    print()

    # ---- 段级: 带拼音 vs 不带拼音, 空输出的比例 ----
    n_seg = n_pin = n_empty = n_empty_pin = 0
    widths_ok, widths_bad = [], []
    for r in rows:
        for s in r["segs"]:
            n_seg += 1
            p = s.get("pin")
            n_pin += bool(p)
            if not (s["t"] or "").strip():
                n_empty += 1
                n_empty_pin += bool(p)
                widths_bad.append(s["x1"] - s["x0"])
            else:
                widths_ok.append(s["x1"] - s["x0"])
    print(f"  段 {n_seg:,}   其中上面有拼音 {n_pin:,} ({n_pin/n_seg:.0%})")
    print(f"  读出来是空的 {n_empty:,} ({n_empty/n_seg:.1%})"
          f"   其中带拼音的 {n_empty_pin:,}")
    if widths_bad:
        import statistics as st
        print(f"  空的那些段宽度中位 {st.median(widths_bad):.0f}px,"
              f"  有字的 {st.median(widths_ok):.0f}px")
        print("  ★ 空的明显更窄的话, 说明是碎片没滤干净, 不是模型读不出来")
    print()

    # ---- 字段级: 谁读到了 ----
    print("  ---- 每个字段: ceiling 读到 / 我们读到 / 我们丢了 ----")
    print(f"    {'field':<10}{'ceiling':>9}{'ours':>7}{'我们丢的':>10}")
    lost_tot = 0
    for f in FIELDS_15:
        c = sum(1 for r in rows if f in r["ceiling"])
        o = sum(1 for r in rows if f in r["ours"])
        lost = sum(1 for r in rows if f in r["ceiling"] and f not in r["ours"])
        lost_tot += lost
        if c:
            print(f"    {f:<10}{c:>9}{o:>7}{lost:>10}")
    print(f"    {'合计丢的':<10}{'':>9}{'':>7}{lost_tot:>10}")
    print()

    # ---- 丢掉的东西按类型归类 ----
    lost_kind = Counter()
    extra_kind = Counter()
    lost_ex = []
    for r in rows:
        co, ou = set(toks(r["ceiling"])), set(toks(r["ours"]))
        for t in co - ou:
            lost_kind[kind(t)] += 1
            if len(lost_ex) < 400:
                lost_ex.append(t)
        for t in ou - co:
            extra_kind[kind(t)] += 1
    tot_l = sum(lost_kind.values()) or 1
    print("  ---- ceiling 有而我们没有的(按类型) ----")
    for k, v in lost_kind.most_common():
        print(f"    {k:<8}{v:>7}  ({v/tot_l:5.1%})")
    print()
    tot_e = sum(extra_kind.values()) or 1
    print("  ---- 我们有而 ceiling 没有的(按类型) ----")
    print("    ★ 这里面有一部分是**我们读对了而 ceiling 读错了**, 人工核过约 15%")
    for k, v in extra_kind.most_common():
        print(f"    {k:<8}{v:>7}  ({v/tot_e:5.1%})")
    print()
    print(f"  ---- 丢掉的里面最常见的几个 ----")
    for t, n in Counter(lost_ex).most_common(a.top):
        print(f"    {n:>4}x  {t[:40]}")


if __name__ == "__main__":
    main()
