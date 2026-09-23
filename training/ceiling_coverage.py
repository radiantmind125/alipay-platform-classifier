"""查天花板这个数是不是**被稀释**了。

为什么要查
----------
e2e_bench 里天花板是这么取的:

    tc = ceil_by_src.get(p.name, "")

★★★★★ 清单里**没有**这张图的时候, 不是跳过, 是当成**空字符串**算进去 ——
   这张图给天花板记 0 个字段 0 个字, 而 `ours` 那边照常读照常记分。
   于是清单缺一张, 天花板就被往下拽一点, **拽的方向正好对我们有利**。

   打出来的 `ceiling zero 78` 分不清是哪一种:
       真读不出      现成 OCR 在这页上确实一个字段都没读到
       清单里没有    这页压根没进过配对目录, 是被当成 0 混进去的

   之前 ceiling_bench 算出 nan 就是同一类毛病(清单存的是服务器路径,
   basename 对不上), 这次 basename 是对的, 但**覆盖率**还没人量过。

这个脚本干什么
--------------
照抄 e2e_bench 的抽样方式(同样 sorted + seed + sample), 然后回答两件:

    抽中的这 400 张, 有几张在清单里
    只按**在清单里**的那些算, 天花板是多少

★ 两个天花板差得多, 说明之前那个数是被缺图稀释出来的, 不能拿去比。
  差不多, 说明 78 个 0 是真读不出, 那个数能用。

用法
----
    python ceiling_coverage.py --src D:\\download2\\pinyin_hits_blue ^
        --pairs D:\\alipay-ai-data\\pinyin-pairs-blue2 --n 400
"""
from __future__ import annotations

import argparse
import csv
import random
import re
from collections import defaultdict
from pathlib import Path

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
HAN = re.compile(r"[一-鿿]")

# 和 e2e_bench 里那份保持一致
FIELDS_15 = ["账单详情", "全部账单", "创建时间", "付款方式", "订单号",
             "转账备注", "收款方", "账户余额", "计入收支", "转账成功",
             "交易成功", "对方账户", "商家名称", "支付时间", "交易号"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=17, help="要和 e2e_bench 一致")
    a = ap.parse_args()

    # ★ 抽样必须和 e2e_bench 一字不差, 否则量的不是同一批图
    files = sorted(p for p in a.src.iterdir() if p.suffix.lower() in EXTS)
    total_src = len(files)
    random.seed(a.seed)
    files = random.sample(files, min(a.n, len(files)))

    man = a.pairs / "_pairs_labeled.csv"
    if not man.exists():
        print(f"  清单不在: {man}")
        return

    agg = defaultdict(list)
    rows_total = 0
    for r in csv.DictReader(man.open(encoding="utf-8-sig")):
        rows_total += 1
        if (r.get("text") or "").strip():
            agg[Path(r["source"]).name].append(r["text"])
    ceil_by_src = {k: " ".join(v) for k, v in agg.items()}

    inside, outside = [], []
    for p in files:
        (inside if p.name in ceil_by_src else outside).append(p)

    print("=" * 58)
    print("  CEILING COVERAGE  (ASCII only - safe to paste)")
    print("=" * 58)
    print(f"  src        {a.src.name}   共 {total_src:,} 张")
    print(f"  pairs      {a.pairs.name}   清单 {rows_total:,} 行 / "
          f"{len(ceil_by_src):,} 张图有文字")
    print()
    print(f"  抽中 {len(files)} 张")
    print(f"    在清单里   {len(inside):>4}  ({len(inside)/len(files):.1%})")
    print(f"    不在清单里 {len(outside):>4}  ({len(outside)/len(files):.1%})"
          f"{'   <-- 这些被当成 0 拽低了天花板' if outside else '   <-- 天花板没被稀释'}")
    print()

    def stats(sel):
        if not sel:
            return 0.0, 0.0, 0
        chars = [len(HAN.findall(ceil_by_src.get(p.name, ""))) for p in sel]
        flds = [sum(1 for f in FIELDS_15 if f in ceil_by_src.get(p.name, ""))
                for p in sel]
        zero = sum(1 for v in flds if v == 0)
        return sum(chars) / len(sel), sum(flds) / len(sel), zero

    c_all, f_all, z_all = stats(files)
    c_in, f_in, z_in = stats(inside)

    print("  天花板两种算法:")
    print(f"    缺图当 0 算(e2e_bench 现在就是这样)   "
          f"{c_all:>6.1f} 字/张   {f_all:.2f} 字段/张   0 字段的 {z_all}")
    print(f"    只算在清单里的那些                  "
          f"{c_in:>6.1f} 字/张   {f_in:.2f} 字段/张   0 字段的 {z_in}")
    print()
    if outside:
        gap = c_in - c_all
        print(f"  ★★ 两者差 {gap:.1f} 字/张({gap/max(1e-9,c_in):.1%})。"
              f"e2e_bench 报的天花板**低了这么多**,")
        print(f"     ours/ceiling 那个比值得按 {c_in:.1f} 重算才作数。")
        print(f"     (e2e_bench 已经改成自动剔除缺清单的图, 重跑一遍就是对的)")
    else:
        print("  ★ 没有缺图, e2e_bench 报的天花板可以直接用。")
    print()
    print(f"  真正 0 字段的 {z_in} 张(在清单里但一个字段都没读到) —— "
          f"这些才是现成 OCR 真读不出的")


if __name__ == "__main__":
    main()
