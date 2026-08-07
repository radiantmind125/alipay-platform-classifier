r"""标定"可以直接自动拒"的分数线 —— 经理要做到自动化, 所以误杀要压到能自动执行的水平。

和 sweep_thresholds 的区别:
- sweep_thresholds 用的是**百分比预算**(0.1%/0.5%…), 适合 review-only(命中只是进人工复核);
- 自动拒是**真的把用户拒掉**, 容忍度要按**万分之几**算, 而且要看**具体是哪几张真图卡住了线**
  (万分之一量级下, 阈值往往就由那么两三张图决定 —— 值得人工看一眼它们是不是本来就是假图)。

用法:
  python training/autoreject_threshold.py --genuine D:\probe\genuine_20k_v7\summary.csv \
      --fake D:\probe\wan_full_v7\summary.csv D:\probe\qwen_v7\summary.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# 自动拒能容忍的误杀上限(按"多少分之一"给, 比百分比直观)
_BUDGETS = [(1, 1000), (1, 5000), (1, 10000), (1, 20000), (0, 0)]


def _scores(p: Path, col: str = "final_ai_score") -> list[tuple[float, str]]:
    out = []
    for r in csv.DictReader(open(p, encoding="utf-8-sig")):
        v = (r.get(col) or "").strip()
        if not v:
            continue
        if "scores_json" in r and (r.get("scores_json") or "").strip() in ("", "{}"):
            continue                      # 评分失败的不计入
        try:
            out.append((float(v), r.get("image_name") or r.get("image") or ""))
        except ValueError:
            pass
    return out


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="标定自动拒绝的安全分数线")
    ap.add_argument("--genuine", type=Path, required=True, help="大批真图的 summary.csv")
    ap.add_argument("--fake", type=Path, nargs="+", required=True, help="若干假图集的 summary.csv")
    ap.add_argument("--show-top", type=int, default=15, help="列出分数最高的 N 张真图(卡住阈值的就是它们)")
    args = ap.parse_args()

    g = sorted(_scores(args.genuine), key=lambda t: -t[0])
    if not g:
        raise SystemExit("真图 summary 里没读到分数")
    n = len(g)
    fakes = {p.parent.name or p.stem: _scores(p) for p in args.fake}

    print(f"真图 {n} 张 | 假图集: " + ", ".join(f"{k}({len(v)})" for k, v in fakes.items()))
    print(f"真图最高分 {g[0][0]:.4f} | 能可靠分辨的最小误杀率 ≈ 1/{n}")
    print()
    print("误杀上限        分数线      " + "  ".join(f"{k[:14]:>14s}" for k in fakes))
    print("-" * (28 + 16 * len(fakes)))

    for num, den in _BUDGETS:
        if den == 0:                       # 零误杀: 阈值必须高过最高的那张真图
            thr = g[0][0] + 1e-6
            label = "0(零误杀)"
        else:
            if den > n:
                continue                   # 样本量不够, 这个档没法可靠估计
            k = max(0, int(n * num / den))     # 允许 k 张真图被误拒
            thr = g[k][0] + 1e-6 if k < n else 0.0
            label = f"{num}/{den}"
        cells = []
        for name, fs in fakes.items():
            rec = sum(1 for s, _ in fs if s >= thr) / max(1, len(fs))
            cells.append(f"{rec * 100:13.1f}%")
        print(f"{label:14s} {thr:8.4f}   " + "  ".join(cells))

    print()
    print(f"分数最高的 {min(args.show_top, n)} 张真图(自动拒的线就是被这几张卡住的):")
    for s, nm in g[:args.show_top]:
        print(f"  {s:.4f}  {nm}")
    print()
    print("**强烈建议人工看一眼上面这几张** —— 我们已知真图池里混着疑似假图。")
    print("如果它们本来就是假的, 真实误杀比这里算出来的还低, 分数线可以往下放。")


if __name__ == "__main__":
    main()
