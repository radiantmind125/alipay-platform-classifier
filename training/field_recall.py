"""把"字段命中率低"拆开: 是**页上没有**, 还是**我们读丢了**。

为什么要拆
----------
e2e_bench 打出来的是这样一张表:

    账户余额           18%     18%
    转账备注           34%     34%

★★★★★ 18% 这个数**同时混了两件事**:

    页上根本没有这个字段   ->  再准也是 18%, 没有改进空间
    页上有但我们读丢了     ->  有改进空间

   光看这张表分不开。天花板那一列是个旁证(天花板也是 18%, 说明多半是没有),
   但天花板**不是真值** —— 要是这个字段其实 60% 的页上都有, 而我们和天花板
   各自都只读到一半, 表上就会显示 30%/30%, 看着像"页上没有", 其实是两边都漏。

怎么拆
------
拿 `ours ∪ ceiling` 当"页上确实有"的下界(两边任一读到, 就说明它在页上),
然后在这个分母上算各自的召回:

    页上有        |ours ∪ ceiling|
    我们的召回     |ours| / |ours ∪ ceiling|
    天花板的召回   |ceiling| / |ours ∪ ceiling|

★ 这是**下界**不是真值 —— 两边都漏的那些页仍然数不进来。
  但它足够回答"我们还有多少可捡的", 这才是要决策的事。

漏掉的再分四类
--------------
对每一张"页上有但我们没读到"的图, 按这个顺序判:

    切开了        去掉空格之后能匹配上 -> 字段被切成了两段, 是**拼接**的问题
    读错一个字     某一段和字段名编辑距离<=1 -> 是**识别**的问题
    读错两个字     编辑距离<=2             -> 还是识别, 但错得多
    没看见        以上都不是              -> 是**版面/检测**的问题

★★★★★ 这四类对应**完全不同的修法**, 所以必须分开数:

    切开了    改拼接逻辑就行, **不用重训**
    读错字    要针对性补数据, 要重训
    没看见    要动行检测, 动的是共用几何, 风险最大

用法
----
    先让 e2e_bench 把每张图的文字存下来:
      python e2e_bench.py --src ... --onnx ... --pairs ... --n 400 --no-baseline ^
          --dump D:\\alipay-ai-data\\dump_white.jsonl
    再拆:
      python field_recall.py --dump D:\\alipay-ai-data\\dump_white.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

FIELDS_15 = ["账单详情", "全部账单", "创建时间", "付款方式", "订单号",
             "转账备注", "收款方", "账户余额", "计入收支", "转账成功",
             "交易成功", "对方账户", "商家名称", "支付时间", "交易号"]


def lev(a: str, b: str) -> int:
    """编辑距离。字段名都很短(3~4 个字), 直接 DP 就够快。"""
    if a == b:
        return 0
    if not a or not b:
        return len(a) or len(b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def classify(field: str, rec: dict) -> tuple[str, str]:
    """这一张图上我们为什么没读到这个字段。返回(类别, 证据)。"""
    ours = rec.get("ours") or ""
    # 1) 拼接问题: 整页文字是用空格连起来的, 字段被切成两段就匹配不上
    if field in ours.replace(" ", ""):
        return "切开了", ""
    # 2) ★★★★★ 裁切截断: 我们读出来的是字段名的**真子串**, 掉了头一个或末一个字。
    #    实测 '账单详情'->'账单详', '订单号'->'单号' —— 这**不是认错字**,
    #    是切段的边界把头尾吃掉了。修法是放宽边界, **不用补数据重训**,
    #    和 '付款方式'->'付默方式' 那种真认错字完全两回事。
    #    之前一律归进"读错N个字", 把便宜的和贵的混在一起了。
    for s in rec.get("segs") or []:
        t = (s.get("t") or "").strip()
        if t and t != field and t in field and len(t) >= len(field) - 2:
            return "裁切掉了字", t
    # 3) 识别问题: 有某一段长得很像
    best, bt = 99, ""
    for s in rec.get("segs") or []:
        t = (s.get("t") or "").strip()
        if not t:
            continue
        # 只和长度接近的比, 不然长句子里的子串会乱匹配
        if abs(len(t) - len(field)) <= 1:
            d = lev(t, field)
            if d < best:
                best, bt = d, t
        # 字段可能嵌在长段里, 拿同长度的窗口滑一遍
        elif len(t) > len(field):
            for i in range(len(t) - len(field) + 1):
                d = lev(t[i:i + len(field)], field)
                if d < best:
                    best, bt = d, t[i:i + len(field)]
    if best <= 1:
        return "读错1个字", bt
    if best <= 2:
        return "读错2个字", bt
    return "没看见", ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--show", type=int, default=3,
                    help="每个字段打几个漏掉的例子")
    a = ap.parse_args()

    recs = []
    with a.dump.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    if not recs:
        print("  dump 是空的")
        return

    print("=" * 74)
    print("  FIELD RECALL  (ASCII only - safe to paste)")
    print("=" * 74)
    print(f"  {len(recs)} 张图")
    print()
    print("  分母是 ours 和 ceiling 任一读到的页数 = 页上确实有这个字段(下界)")
    print()
    head = (f"  {'field':<10}{'页上有':>7}{'我们':>10}{'天花板':>10}"
            f"{'我们漏':>8}{'反超':>6}")
    print(head)
    print("  " + "-" * (len(head) - 2))

    misses = {}
    tot_present = tot_ours = tot_ceil = 0
    tot_onlyb = 0
    onlyb_by_field = {}
    for fd in FIELDS_15:
        pres, o_hit, c_hit, miss = 0, 0, 0, []
        won, only_b = 0, []
        for r in recs:
            ours = (r.get("ours") or "").replace(" ", "")
            ceil = (r.get("ceiling") or "").replace(" ", "")
            base = (r.get("base") or "").replace(" ", "")
            in_o, in_c, in_b = fd in ours, fd in ceil, fd in base
            # ★★★★★ 并集里**必须**带上 base。
            #   天花板是拿**我们自己切出来的行**去 OCR 的, 行检测漏掉的行
            #   天花板同样看不见 —— 两边一起漏, 那个字段就从分母里悄悄消失了,
            #   于是我们的召回看着永远是 100%。
            #   base 是现成 OCR 整页读的, **版面是它自己做的**, 和我们无关,
            #   所以它能照出我们行检测的漏。
            if not (in_o or in_c or in_b):
                continue
            pres += 1
            o_hit += in_o
            c_hit += in_c
            if in_o and not in_c:
                won += 1
            if in_b and not in_o and not in_c:
                only_b.append(r)      # ← 独立版面看到了, 我们这条线整个没看到
            if not in_o:
                miss.append(r)
        tot_onlyb += len(only_b)
        if only_b:
            onlyb_by_field[fd] = only_b
        if pres == 0:
            print(f"  {fd:<10}{0:>7}{'-':>10}{'-':>10}{'-':>8}{'-':>6}"
                  f"   <- 这批图上根本没有")
            continue
        tot_present += pres
        tot_ours += o_hit
        tot_ceil += c_hit
        misses[fd] = miss
        print(f"  {fd:<10}{pres:>7}{o_hit:>6}({o_hit/pres:>3.0%}){c_hit:>6}"
              f"({c_hit/pres:>3.0%}){len(miss):>8}{won:>6}")

    print("  " + "-" * (len(head) - 2))
    print(f"  {'合计':<10}{tot_present:>7}{tot_ours:>6}"
          f"({tot_ours/max(1,tot_present):>3.0%}){tot_ceil:>6}"
          f"({tot_ceil/max(1,tot_present):>3.0%})"
          f"{tot_present-tot_ours:>8}")
    print()
    print("  ★ '页上有'那一列才是真分母。e2e_bench 那张表的分母是全部 400 张,")
    print("    所以那里的 18% 和这里的召回**不是一回事**。")
    print()

    # ★★★★★ 这一段才是判"还有多少可捡"的关键
    has_base = any((r.get("base") or "").strip() for r in recs)
    print("=" * 74)
    if not has_base:
        print("  【这一跑没有独立版面的旁证 —— 上面的召回是**虚高**的】")
        print("=" * 74)
        print("  天花板是拿我们自己切出来的行去 OCR 的, 和我们**共用行检测**。")
        print("  行检测整个漏掉的行, 天花板也看不见, 那个字段就从分母里消失了。")
        print("  所以上面那个 100% 的准确说法是:")
        print("      **我们把版面找到的字段基本都读对了**")
        print("  而不是:")
        print("      我们把页面上的字段都找到了")
        print()
        print("  要量后者, 得让现成 OCR **整页自己做版面**读一遍当旁证:")
        print("    去掉 --no-baseline, 加 --dump, 再跑这个脚本")
        print("    (现成 OCR 整页只有 0.18 张/秒, 所以 --n 取 150 就够)")
    else:
        print("  【独立版面旁证: 现成 OCR 整页自己做版面, 它看到而我们没看到的】")
        print("=" * 74)
        print("  现成 OCR 的版面和我们无关, 所以它能照出**我们行检测的漏**。")
        print()
        if not tot_onlyb:
            print("  ★★★★★ 一条都没有 —— 独立版面也没找到我们漏掉的字段。")
            print("     说明行检测这一层**没有明显的漏**, 上面的召回可以当真。")
        else:
            print(f"  ★★ 共 {tot_onlyb} 条: 它读到了, 我们和天花板都没读到。")
            print()
            # ★★★★★ 这一类**不能一律算作"行检测漏了"**。
            #   实测白图上 账单详情 的 3 条在这里, 但按原因分是"读错2个字 2 +
            #   读错1个字 1" —— 我们**看见了那一行**, 只是读错, 天花板也读错,
            #   现成 OCR 读对了。那是识别问题, 不是版面问题。
            #   只有归到"没看见"的才是真的版面漏。
            blind, seen = 0, 0
            files_blind = set()
            for fd, lst in sorted(onlyb_by_field.items(),
                                  key=lambda kv: -len(kv[1])):
                cats = {}
                for r in lst:
                    c, _ = classify(fd, r)
                    cats[c] = cats.get(c, 0) + 1
                    if c == "没看见":
                        blind += 1
                        files_blind.add(r.get("file"))
                    else:
                        seen += 1
                line = "  ".join(f"{c} {n}" for c, n in
                                 sorted(cats.items(), key=lambda kv: -kv[1]))
                print(f"    {fd:<10}{len(lst):>4} 条   {line}")
                for r in lst[:a.show]:
                    print(f"               例 {r.get('file')}")
            print()
            print(f"  ★★★★★ 拆开看: 真正**版面没看见**的 {blind} 条"
                  f"(涉及 {len(files_blind)} 张图), "
                  f"看见了但读错的 {seen} 条")
            print("     前者才要动行检测。后者补数据就行, 便宜得多。")
    print()

    print("=" * 74)
    print("  我们漏掉的, 按原因分")
    print("=" * 74)
    grand = {}
    for fd, miss in misses.items():
        if not miss:
            continue
        cats = {}
        ex = {}
        for r in miss:
            c, evid = classify(fd, r)
            cats[c] = cats.get(c, 0) + 1
            grand[c] = grand.get(c, 0) + 1
            if evid and c not in ex:
                ex[c] = evid
        line = "  ".join(f"{c} {n}" for c, n in
                         sorted(cats.items(), key=lambda kv: -kv[1]))
        print(f"  {fd:<10}漏 {len(miss):>3}   {line}")
        for c, t in ex.items():
            print(f"             {c} 读成了 {t!r}")
        for r in miss[:a.show]:
            print(f"             例 {r.get('file')}")
    print()
    print("  " + "-" * 70)
    tot_miss = sum(grand.values())
    print(f"  全部漏掉的 {tot_miss} 条, 按原因:")
    for c, n in sorted(grand.items(), key=lambda kv: -kv[1]):
        fix = {"切开了": "改拼接, 不用重训",
               "裁切掉了字": "放宽切段边界, **不用重训**, 最便宜",
               "读错1个字": "补这个字段的数据, 要重训",
               "读错2个字": "补这个字段的数据, 要重训",
               "没看见": "要动行检测, 风险最大"}.get(c, "")
        print(f"    {c:<10}{n:>5}  ({n/max(1,tot_miss):>4.0%})   {fix}")
    print()
    print("  ★★★★★ 先修占比最大的那一类。'切开了'最便宜, '没看见'最贵。")


if __name__ == "__main__":
    main()
