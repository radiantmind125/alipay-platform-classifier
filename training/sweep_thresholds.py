r"""在"误杀预算"下扫阈值, 看分块打分到底有没有可用的工作点。

为什么要它: 分块打分把局部改动的漏检从 100% 降到 65%, 但真图误杀从 0.1% 涨到 6.2%(红线不可接受)。
问题变成: **有没有某个聚合方式 + 阈值, 能在误杀≤红线的前提下, 还抓到一些局部改动的假图?**
硬指标是误杀(经理: 误杀要赔钱), 所以做法是**先固定误杀预算, 再看能换到多少召回**。

对每种聚合(max/top3/mean):
  按真图分数取分位数当阈值(误杀预算 0.1% / 0.5% / 1% / 2%), 再算该阈值下假图的召回。
这样一眼看出"零误杀的代价"。

用法:
  python training/sweep_thresholds.py --fake D:\probe\localedit_tiled\summary.csv --genuine D:\probe\valnat_tiled\summary.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_COLS = ["tile_max", "tile_top3", "tile_mean"]


def _read(p: Path, col: str, require_located: bool = False) -> list[float]:
    out = []
    for r in csv.DictReader(open(p, encoding="utf-8-sig")):
        if require_located and (r.get("roi_amount_located") or "1").strip() != "1":
            continue          # 金额没定位到 = 打分打的不是金额区, 这种情况干脆不出信号
        v = (r.get(col) or "").strip()
        if v:
            try:
                out.append(float(v))
            except ValueError:
                pass
    return out


def _top_rows(p: Path, col: str, n: int):
    """分数最高的几张真图 —— 它们决定了严阈值下的召回上限, 看看是不是某种特定情况(如金额定位失败)。"""
    rows = []
    for r in csv.DictReader(open(p, encoding="utf-8-sig")):
        v = (r.get(col) or "").strip()
        if v:
            try:
                rows.append((float(v), r.get("roi_amount_located", "?"), r.get("image_name", "")))
            except ValueError:
                pass
    rows.sort(reverse=True)
    return rows[:n]


def _quantile(sorted_vals: list[float], q: float) -> float:
    """取分位数(q=0.999 -> 99.9% 的真图都低于它 -> 误杀约 0.1%)。"""
    if not sorted_vals:
        return float("inf")
    i = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[i]


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="固定误杀预算扫阈值, 看召回能有多少")
    ap.add_argument("--fake", type=Path, required=True, help="假图的 tiled summary.csv")
    ap.add_argument("--genuine", type=Path, required=True, help="真图的 tiled summary.csv")
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.001, 0.005, 0.01, 0.02],
                    help="误杀预算(比例)")
    ap.add_argument("--show-top", type=int, default=10,
                    help="列出分数最高的 N 张真图(它们决定严阈值下的召回上限; 看是不是金额定位失败那批)")
    ap.add_argument("--require-located", action="store_true",
                    help="只统计**金额定位成功**的图(定位失败时打的不是金额区, 上线就该不出这个信号)。"
                         "用来量化'定位失败不出信号'这条规则能把尾巴清多干净")
    args = ap.parse_args()
    if args.require_located:
        print("(只统计金额定位成功的图: 定位失败的不出信号)")

    print("聚合方式    误杀预算    阈值      假图召回(该阈值下)")
    print("-" * 58)
    best = None
    for col in _COLS:
        f = _read(args.fake, col, args.require_located)
        g = sorted(_read(args.genuine, col, args.require_located))
        if not f or not g:
            print(f"{col:10s} 读不到数据(检查 CSV 是否有这列)")
            continue
        for b in args.budgets:
            thr = _quantile(g, 1.0 - b)
            rec = sum(1 for x in f if x >= thr) / len(f)
            fp = sum(1 for x in g if x >= thr) / len(g)
            print(f"{col:10s}  {b*100:5.1f}%   {thr:7.4f}   {rec*100:5.1f}%   (实测误杀 {fp*100:4.1f}%)")
            if b <= 0.001 and (best is None or rec > best[1]):
                best = (col, rec, thr)
        print("-" * 58)

    print(f"\n真图 {len(_read(args.genuine, 'tile_max', args.require_located))} 张 | "
          f"假图 {len(_read(args.fake, 'tile_max', args.require_located))} 张"
          f"{'(仅金额定位成功)' if args.require_located else ''}")
    if best:
        col, rec, thr = best
        print(f"\n误杀 0.1% 预算下最好的是 {col}, 阈值 {thr:.4f}, 局部改动召回 {rec*100:.1f}%")
        if rec < 0.10:
            print("=> 召回太低, 分块打分这条路在零误杀前提下基本没用 —— 需要专门训练的局部篡改检测。")
        elif rec < 0.40:
            print("=> 有一点用但很有限, 可当辅助信号(不能单独承担这类欺诈)。")
        else:
            print("=> 可用! 按这个聚合和阈值配置上线。")
    print("\n注意: 真图样本量决定了能可靠估计的最小误杀率(500 张最多估到 0.2% 量级)。")
    print("若结论卡在边界, 需要用更多真图重跑(去掉 --limit)。")

    if args.show_top > 0:
        top = _top_rows(args.genuine, "tile_max", args.show_top)
        if top:
            print(f"\n分数最高的 {len(top)} 张真图(严阈值下就是被它们卡住的):")
            print("  分数     金额定位成功  文件名")
            for s, loc, nm in top:
                print(f"  {s:.4f}   {loc:^10s}   {nm}")
            locs = [l for _, l, _ in top]
            if locs.count("0") > locs.count("1"):
                print("  => 高分真图多是**金额定位失败**那批(退回了整片上部打分)。")
                print("     对策: 定位失败的图不出这个信号(或单独走更严阈值), 尾巴能干净很多。")
            elif locs.count("1") > locs.count("0"):
                print("  => 高分真图大多定位成功 —— 说明是金额区本身就容易被误判, 需要专训的局部检测器。")


if __name__ == "__main__":
    main()
