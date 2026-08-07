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
    ap.add_argument("--a-fake", type=Path, nargs="*", default=None,
                    help="线路A 的假图 summary.csv(可多个) —— 扫阈值时一并显示召回")
    ap.add_argument("--b-fake", type=Path, nargs="*", default=None,
                    help="线路B 的假图 summary.csv(可多个)")
    ap.add_argument("--b-located", action="store_true", default=True,
                    help="线路B 的假图也只算金额定位成功的(默认开)")
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
        fa = {p.parent.name or p.stem: _load(p, args.a_col, False) for p in (args.a_fake or [])}
        fb = {p.parent.name or p.stem: _load(p, args.b_col, args.b_located)
              for p in (args.b_fake or [])}
        print()
        print(f"扫阈值组合, 目标: 并集误杀 <= 1/{args.target}")
        cols = ("".join(f"{k[:9]:>10s}" for k in fa) + ("|" if fa and fb else "")
                + "".join(f"{k[:9]:>10s}" for k in fb))
        print(f"{'线路A阈值':>10} {'线路B阈值':>10} {'并集':>6} {'并集误杀':>11}  {cols}")
        sa = sorted(A.values(), reverse=True)
        sb = sorted((B[k] for k in set(B) & universe), reverse=True)
        cand, seen = [], set()
        for ka in (0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 30, 50, 80):
            for kb in (0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 30, 50, 80):
                if ka >= len(sa) or kb >= len(sb):
                    continue
                ta, tb = sa[ka] + 1e-9, sb[kb] + 1e-9
                key = (round(ta, 4), round(tb, 4))
                if key in seen:
                    continue
                seen.add(key)
                a_hit, b_hit, both = union_rate(ta, tb)
                rate = n / len(both) if both else float("inf")
                if rate < args.target:
                    continue
                cand.append((ta, tb, len(both), rate))
        # 阈值越低召回越高 -> 按两个阈值之和**升序**排, 最好的排最前面
        cand.sort(key=lambda t: t[0] + t[1])
        for ta, tb, nb, rate in cand[:20]:
            cells = "".join(f"{sum(1 for s in v.values() if s >= ta) / max(1, len(v)) * 100:9.1f}%"
                            for v in fa.values())
            if fa and fb:
                cells += "|"
            cells += "".join(f"{sum(1 for s in v.values() if s >= tb) / max(1, len(v)) * 100:9.1f}%"
                             for v in fb.values())
            print(f"{ta:10.4f} {tb:10.4f} {nb:6d} {'1/' + format(rate, ',.0f'):>11}  {cells}")
        print()
        print("★ 表按'两个阈值之和'**升序**排, 所以**第一行就是达标组合里召回最高的那个**。")
        print("  最后几列是该阈值下各假图集的召回(左边线路A, 右边线路B)。")
        print(f"  注意: 刚好差一点点的组合被过滤掉了(比如并集 1/4,994 达不到 1/{args.target}),")
        print("  想看这些就把 --target 调松一点再跑。")
        return

    raise SystemExit("要么同时给 --a-thr 和 --b-thr, 要么给 --target")


if __name__ == "__main__":
    main()
