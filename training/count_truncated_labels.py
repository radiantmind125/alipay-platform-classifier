"""估**训练标签里有多少是被底边裁掉字的** —— 不用重跑 OCR。

为什么要先估
------------
已经证实底边会切掉字: `订单号` 的段, 底下多留 8 像素就从 `单号` 变回 `订单号`,
8 个样本全中。而 `y_label` 和 `y_input` 共用同一个底边, 所以当初出训练数据时
**label 也是缺字的** —— 训练对是 (缺字的图 -> '单号')。

★★★★★ 但根治要**重出一遍训练数据**: 白图约 58 小时, 蓝图约 6 小时。
   花这个时间之前, 得先知道**到底有多少条是坏的**。
   1% 和 10% 是完全不同的决定。

怎么估 —— 拿宽度和字数对不上来判
--------------------------------
一段图有多宽, 它的文字就该有多少字, 两者大体成正比。要是某一段
**图挺宽但字很少**, 多半是标签掉了字:

```
段宽 / 字数  明显高于中位        ->  可疑
而且 段宽 / (字数+1) 更贴近中位  ->  **正好像是少了一个字**
```

★ 第二个条件很关键: 光"偏宽"还可能是字距大、是标点。
  "加一个字就正好对上"才是掉字的特征。

★★ 这是**下界**: 掉了字之后宽度恰好还落在正常区间的那些, 数不出来。

用法
----
    python count_truncated_labels.py --pairs D:\\alipay-ai-data\\pinyin-pairs ^
        --also D:\\alipay-ai-data\\pinyin-pairs-blue2
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

import numpy as np

HAN = re.compile(r"[一-鿿]")
# 回单上的固定字段名, 用来单独数"掉字掉在关键字段上"的那些
FIELDS = ["账单详情", "全部账单", "创建时间", "付款方式", "订单号", "转账备注",
          "收款方", "账户余额", "计入收支", "转账成功", "交易成功", "对方账户",
          "商家名称", "支付时间", "交易号", "商品说明", "收款方全称",
          "商家订单号", "账单管理", "账单分类", "标签", "备注"]


def units(t: str) -> float:
    """汉字算 1 个宽度单位, 半角算 0.5 —— 不然一串数字会被当成太宽。"""
    return sum(1.0 if HAN.match(c) else 0.5 for c in t)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--also", type=Path, nargs="*", default=[])
    ap.add_argument("--show", type=int, default=15)
    a = ap.parse_args()

    print("=" * 72)
    print("  COUNT TRUNCATED LABELS  (ASCII only - safe to paste)")
    print("=" * 72)
    print("  判据: 段宽/字数 明显偏高, 而且 段宽/(字数+1) 更贴近中位")
    print("        = 这一段的标签**正好像是少了一个字**")
    print()

    for d in [a.pairs, *a.also]:
        man = d / "_segments.csv"
        if not man.exists():
            print(f"  清单不在: {man}")
            continue
        rows = list(csv.DictReader(man.open(encoding="utf-8-sig")))
        keep = []
        for r in rows:
            t = (r.get("text") or "").strip()
            if not t or r.get("bad") == "1":
                continue
            u = units(t)
            w = int(r["x1"]) - int(r["x0"])
            if u <= 0 or w <= 0:
                continue
            # ★★★★★ 必须按**这一段自己的字号**归一, 不能拿整页中位当基准。
            #   第一版就是拿整页中位比的, 结果把**大字号的完整文字**全判成了"缺字":
            #       4898x 账单详情   3670x -100.00   2810x 添加   455x 全部账单
            #   这些一个字都不缺, 只是标题和金额的字号本来就大, 宽/字自然高。
            #   于是报出 14.70%, 而真实的量级是 0.29% —— **差了五十倍**。
            #   按段高分桶再比, 就把字号这个因素消掉了。
            h = 0
            if r.get("y0") is not None and r.get("y1") is not None:
                try:
                    h = int(r["y1"]) - int(r["y0"])
                except (TypeError, ValueError):
                    h = 0
            keep.append((r, t, u, w, h))
        if not keep:
            print(f"  {d.name}: 没有可用的段")
            continue

        has_h = sum(1 for _r, _t, _u, _w, h in keep if h > 0) > len(keep) * 0.9
        med_by = {}
        if has_h:
            # 按段高分桶(每 8 像素一桶), 每桶各自算中位
            bucket = {}
            for _r, _t, u, w, h in keep:
                bucket.setdefault(h // 8, []).append(w / u)
            med_by = {k: float(np.median(v)) for k, v in bucket.items()
                      if len(v) >= 30}
        med = float(np.median([w / u for _r, _t, u, w, _h in keep]))

        susp, field_hit = [], []
        for r, t, u, w, h in keep:
            ref = med_by.get(h // 8, med) if has_h else med
            cur = w / u
            nxt = w / (u + 1.0)
            # 偏宽, 且"再加一个字"之后更贴近**同字号**的中位
            if cur > ref * 1.35 and abs(nxt - ref) < abs(cur - ref):
                susp.append((r, t, cur))
                # 这一段的文字要是某个字段名的真子串, 那基本可以坐实
                if any(t != f and t in f for f in FIELDS):
                    field_hit.append((t, cur))

        n = len(keep)
        print("-" * 72)
        print(f"  {d.name}")
        print(f"    可用段            {n:>9,}")
        print(f"    宽/字 中位        {med:>9.1f} px"
              f"{'   (按段高分了 %d 个字号桶)' % len(med_by) if med_by else ''}")
        print(f"    疑似少一个字       {len(susp):>9,}  ({len(susp)/n:.2%})"
              f"   <- **宽判**, 大字号的完整文字也会混进来")
        print(f"    ★ 其中是字段名真子串 {len(field_hit):>7,}"
              f"  ({len(field_hit)/n:.2%})   <- **这个才是能当数用的**")
        print()
        if field_hit:
            print(f"    坐实的那些长什么样(前 {a.show} 种):")
            for t, c in Counter(t for t, _ in field_hit).most_common(a.show):
                print(f"      {c:>6}x  {t}")
            print()
        if susp:
            print(f"    疑似的整体样子(前 {a.show} 种):")
            for t, c in Counter(t for _r, t, _c in susp).most_common(a.show):
                print(f"      {c:>6}x  {t[:40]}")
        print()

    print("-" * 72)
    print("  ★★★★★ 只能拿**字段名真子串**那一行去做决定。")
    print("     上面那个'疑似'是宽判, 大字号的完整文字(账单详情、-100.00、添加)")
    print("     会大量混进来 —— 第一版就是这么报出 14.70% 的, 真实量级 0.29%。")
    print()
    print("     坐实的**低于 1%**  -> 不值得花 58 小时重跑, 推理端 pad 4 已经够了")
    print("     坐实的**高于 3%**  -> 值得重出, 而且白蓝都受益")
    print()
    print("  ★ 这是**下界**: 只数得出'截断之后仍然是字段名子串'的那些,")
    print("    商家名、金额那类截断了也认不出来。")


if __name__ == "__main__":
    main()
