r"""两条线**合起来**的误杀率 —— 之前一直是分开标的, 但上线是一起跑的。

为什么必须有这个
----------------
`autoreject_threshold.py` 是**一条线一条线**标的: 线路A 标到 1/5000, 线路B 也标到 1/5000。
**但上线时两条线同时跑, 任何一条判拒就真的拒掉了。**
所以真实的误杀率是**并集**, 不是任何一条单独的值:

    真实误杀 = P(线路A 命中 或 线路B 命中)

两条线**不完全相关**(实测线路A 和线路B 的榜首只有两张重合), 所以并集会**明显高于**单条,
最坏情况接近两条之和 —— **1/5000 + 1/5000 = 1/2500, 是我们以为的两倍。**
**跟经理报误杀数字、跟风控定 SLA, 用的都必须是这个并集值。**

还有一个容易漏的语义
--------------------
**线路B 只在"金额区定位成功"时才出信号**(实测约 83% 的图能定位到)。
定位不到的图**只由线路A 判**。所以并集要按下面这个真实逻辑算, 不能简单相加:

    拒绝 = (A分 >= thrA) 或 (B定位成功 且 B分 >= thrB)

用法
----
  # 给定两条线的阈值, 算合起来的误杀
  python training/combined_threshold.py --a D:\probe\genuine_20k_v7\summary.csv --a-thr 0.8881 \
      --b D:\probe\genuine_20k_ld3\summary.csv --b-thr 0.9183 --exclude D:\probe\exclude_evidence.txt

  # 反过来: 给一个并集预算, 扫出能达标的阈值组合
  python training/combined_threshold.py --a ... --b ... --exclude ... --target 5000
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def _load(p: Path, col: str, require_located: bool) -> dict[str, float]:
    out: dict[str, float] = {}
    for r in csv.DictReader(open(p, encoding="utf-8-sig")):
        if require_located and (r.get("roi_amount_located") or "1").strip() != "1":
            continue                       # 定位不到 = 这条线不出信号, 不是"判真"
        name = (r.get("image_name") or "").strip() or Path(r.get("image") or "").name
        v = (r.get(col) or "").strip()
        if not name or not v:
            continue
        try:
            out[name] = float(v)
        except ValueError:
            pass
    return out


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="两条线合起来的误杀率(上线看的就是这个)")
    ap.add_argument("--a", type=Path, required=True, help="线路A 的真图 summary.csv")
    ap.add_argument("--b", type=Path, required=True, help="线路B 的真图 summary.csv")
    ap.add_argument("--a-col", default="final_ai_score")
    ap.add_argument("--b-col", default="tile_top3")
    ap.add_argument("--a-thr", type=float, default=None)
    ap.add_argument("--b-thr", type=float, default=None)
    ap.add_argument("--exclude", type=Path, default=None, help="已确认是假图的文件名清单")
    ap.add_argument("--target", type=int, default=None,
                    help="给一个并集预算(比如 5000 表示 1/5000), 扫出能达标的阈值组合")
    args = ap.parse_args()

    drop = set()
    if args.exclude and args.exclude.exists():
        drop = {ln.strip().lstrip("\ufeff")
                for ln in args.exclude.read_text(encoding="utf-8-sig").splitlines() if ln.strip()}
        print(f"(已排除 {len(drop)} 个确认为假图的文件)")

    A = {k: v for k, v in _load(args.a, args.a_col, False).items() if Path(k).name not in drop}
    B = {k: v for k, v in _load(args.b, args.b_col, True).items() if Path(k).name not in drop}
    universe = set(A)                      # 全量以线路A 为准: 每张图都过线路A
    if not universe:
        raise SystemExit("线路A 的 summary 里没读到分数")
    n = len(universe)
    n_b = len(set(B) & universe)
    print(f"真图 {n:,} 张(线路A 全评); 其中线路B 能出信号的 {n_b:,} 张 "
          f"({n_b / n * 100:.1f}% —— 其余是金额定位不到, 只由线路A 判)")

    def union_rate(ta: float, tb: float):
        a_hit = {k for k in universe if A[k] >= ta}
        b_hit = {k for k in universe if k in B and B[k] >= tb}
        return a_hit, b_hit, a_hit | b_hit

    if args.a_thr is not None and args.b_thr is not None:
        a_hit, b_hit, both = union_rate(args.a_thr, args.b_thr)
        inter = a_hit & b_hit
        print()
        print(f"线路A >= {args.a_thr}: {len(a_hit):4d} 张  = 1/{n / max(1, len(a_hit)):,.0f}")
        print(f"线路B >= {args.b_thr}: {len(b_hit):4d} 张  = 1/{n / max(1, len(b_hit)):,.0f}")
        print(f"两条都命中          : {len(inter):4d} 张")
        print()
        print(f"**并集(上线真实误杀): {len(both)} 张 = 1/{n / max(1, len(both)):,.0f}**")
        if len(a_hit) and len(b_hit):
            worst = len(a_hit) + len(b_hit)
            print(f"  (若两条线完全独立, 最坏是 {worst} 张 = 1/{n / worst:,.0f}; "
                  f"实际重合 {len(inter)} 张, 所以比最坏好一些)")
        print()
        print("★ 报给经理/风控的误杀数字**必须用这个并集值**, 不是任何一条单独的值。")
        return

    if args.target:
        print()
        print(f"扫阈值组合, 目标: 并集误杀 <= 1/{args.target}")
        print(f"{'线路A阈值':>10} {'线路B阈值':>10} {'A命中':>7} {'B命中':>7} "
              f"{'并集':>7} {'并集误杀':>12}")
        sa = sorted(A.values(), reverse=True)
        sb = sorted((B[k] for k in set(B) & universe), reverse=True)
        cand = []
        for ka in (0, 1, 2, 3, 5, 8, 12, 20, 30, 50):
            for kb in (0, 1, 2, 3, 5, 8, 12, 20, 30, 50):
                if ka >= len(sa) or kb >= len(sb):
                    continue
                ta, tb = sa[ka] + 1e-9, sb[kb] + 1e-9
                a_hit, b_hit, both = union_rate(ta, tb)
                if len(both) and n / len(both) < args.target:
                    continue
                cand.append((len(both), ta, tb, len(a_hit), len(b_hit)))
        cand.sort(key=lambda t: (-t[1] - t[2]))       # 阈值越低越好(召回越高)
        seen = set()
        for nb, ta, tb, na, nbh in cand[:18]:
            key = (round(ta, 4), round(tb, 4))
            if key in seen:
                continue
            seen.add(key)
            print(f"{ta:10.4f} {tb:10.4f} {na:7d} {nbh:7d} {nb:7d} "
                  f"{'1/' + format(n / max(1, nb), ',.0f'):>12}")
        print()
        print("怎么挑: **两个阈值都尽量低**(召回高), 同时并集仍然达标。")
        print("拿到组合之后, 再用 autoreject_threshold.py 各自看该阈值下每个假图集的召回。")
        return

    raise SystemExit("要么同时给 --a-thr 和 --b-thr, 要么给 --target")


if __name__ == "__main__":
    main()
