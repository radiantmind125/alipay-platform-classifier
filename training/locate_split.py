r"""按**金额区有没有定位成功**把召回率拆开看 —— 不拆就会把两拨完全不同的图混在一个数里。

为什么要有这个
--------------
`predict_tiled --roi-amount` 有两条完全不同的通路:

- **定位成功**: 只切金额那一小块, 2x2 = 4 块。这才是"局部篡改检测"本来要测的东西。
- **定位失败**: 悄悄退回 `--roi-top`, 整个上半页切 3x6 = 18 块。
  这条路根本不是为局部篡改设计的, 信号弱得多。

**两拨图被平均进同一个召回率里, 而 CSV 里明明记了是哪一拨(`roi_amount_located`)。**
于是"蓝图召回 45.9%"这种数会同时包含两件事: 模型在蓝图上弱, 和蓝图金额定位不到。
这两件事的解法完全不同 —— 前者要补训练数据, 后者要修定位器 —— 混在一起就没法判断该修哪个。

★ 这和之前踩过的版式坑是同一类错误: **把两个分布平均成一个数, 再拿这个数去下结论。**

用法
----
  python training/locate_split.py D:\probe\wanblue_ld6\summary.csv --threshold 0.8670
  python training/locate_split.py D:\probe\*_ld6\summary.csv --threshold 0.8670 --agg tile_top3
  python training/locate_split.py D:\probe\genuine_20k_ld6\summary.csv --threshold 0.8670 --kind genuine

**只读**: 只读 CSV, 什么都不改。
"""

from __future__ import annotations

import argparse
import csv
import glob
import sys
from pathlib import Path


def load(path: Path, agg: str) -> tuple[list[float], list[float]]:
    """返回 (定位成功的分数, 定位失败的分数)。"""
    yes: list[float] = []
    no: list[float] = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if agg not in row:
                raise SystemExit(f"{path} 里没有列 {agg} —— 有的列: {', '.join(row)}")
            try:
                s = float(row[agg])
            except (TypeError, ValueError):
                continue
            (yes if row.get("roi_amount_located") == "1" else no).append(s)
    return yes, no


def label_of(p: Path) -> str:
    """数据集名字。**别只用父目录名** —— 多个清洗后的 CSV 放在同一个目录里时,
    每行都会印成同一个目录名, 表格完全读不出哪行是哪个集合(实测 8 行全印 `_clean_ld8`)。
    约定同 autoreject_threshold: 叫 summary.csv 就用父目录名, 否则用文件名。"""
    return p.parent.name if p.stem == "summary" else p.stem


def pct(hit: int, n: int) -> str:
    return f"{hit * 100.0 / n:5.1f}%" if n else "   -- "


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="按金额区定位成功与否拆分召回率(只读)")
    ap.add_argument("csvs", nargs="+", help="predict_tiled 出的 summary.csv, 可多个, 支持通配符")
    ap.add_argument("--threshold", type=float, required=True, help="判 AI 的阈值")
    ap.add_argument("--agg", default="tile_top3",
                    choices=["tile_top3", "tile_max", "tile_mean", "final_ai_score"],
                    help="用哪一列。注意 final_ai_score 取决于跑分时的 --agg, 跨批次不一定可比, "
                         "所以默认用明确的 tile_top3")
    ap.add_argument("--min-fail", type=int, default=15,
                    help="'定位失败'那一桶至少要有这么多张, 才允许下'缺口在定位器'的结论。"
                         "样本太少时那一列纯属噪声(实测千问蓝只有 1 张定位失败, 0.0% 照样触发了提示)")
    ap.add_argument("--kind", default="fake", choices=["fake", "genuine"],
                    help="fake=报召回率(越高越好); genuine=报误杀率(越低越好)")
    args = ap.parse_args()

    paths: list[Path] = []
    for pat in args.csvs:
        hit = [Path(p) for p in glob.glob(pat)]
        if not hit:
            print(f"!!! 没匹配到: {pat}")
        paths.extend(sorted(hit))
    if not paths:
        raise SystemExit("没有可读的 CSV")

    label = "召回" if args.kind == "fake" else "误杀"
    print("=" * 104)
    print(f"按定位成败拆分   列={args.agg}   阈值={args.threshold}   口径={label}率")
    print("=" * 104)
    print(f"  {'数据集':28s} {'张数':>6s} {'定位率':>7s} | "
          f"{'整体'+label:>8s} {'定位成功'+label:>10s} {'定位失败'+label:>10s}   判读")
    print("-" * 104)

    for p in paths:
        yes, no = load(p, args.agg)
        n = len(yes) + len(no)
        if n == 0:
            print(f"  {label_of(p):28s} 空文件")
            continue
        hy = sum(1 for s in yes if s >= args.threshold)
        hn = sum(1 for s in no if s >= args.threshold)

        # 判读: 定位率低本身就是覆盖缺口, 和模型强弱是两回事
        # ★ "定位失败"那一桶常常只有 1~4 张, 拿 1/4 或 0/1 去下"缺口在定位器"的结论就是噪声当发现。
        #   实测: 千问蓝 143 张里只有 1 张定位失败, 那个 0.0% 照样触发了提示。
        #   所以先要求这一桶有足够样本(--min-fail), 不够就只说样本太少。
        rate = len(yes) * 100.0 / n
        if rate < 60:
            note = f"**定位率只有 {rate:.0f}% —— 这个数主要在测退回通路, 不是测模型**"
        elif len(no) and len(no) < args.min_fail:
            note = f"(定位失败只有 {len(no)} 张, 那一列不作数)"
        elif args.kind == "fake" and len(yes) and len(no) >= args.min_fail and \
                (hy / len(yes) - hn / len(no)) > 0.25:
            note = "定位成败差距大 -> 缺口在定位器, 不在模型"
        else:
            note = ""
        print(f"  {label_of(p):28s} {n:6d} {pct(len(yes), n)} | "
              f"{pct(hy + hn, n):>8s} {pct(hy, len(yes)):>10s} {pct(hn, len(no)):>10s}   {note}")

    print()
    print("怎么读这张表:")
    print("  1) **定位成功那一列才是模型的真实水平。** 整体那列被退回通路稀释过。")
    print("  2) 定位成功高、定位失败低 -> 模型没问题, 缺口在**定位器**(补定位规则, 别急着补训练数据)。")
    print("  3) 定位成功也低 -> 模型确实没见过这种图, 缺口在**训练数据**。")
    print("  4) 真图那批同样要拆: 定位失败的真图走的是整页通路, 误杀特性和定位成功的完全不同。")


if __name__ == "__main__":
    main()
