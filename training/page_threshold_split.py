r"""蓝图和白图分开定阈值, 看能不能白捡一段召回(只读)。

为什么会想到这个
----------------
两种版式**共用一条分数线**。修白图定位器之后, 真图池里多了三千张白图进标定,
把线拉高了 0.9270 -> 0.9462; 结果**蓝图那几格明明没动定位器, 召回却跟着掉了**
(豆包蓝 96.3% -> 93.3%, 合成蓝 94.3% -> 90.3%)。

也就是说: **白图真图的高分尾巴, 在替蓝图定工作点。**

如果蓝图真图的分数分布本来就更紧, 那给它单独定一条线, 就能在**不多花一分误杀预算**的
前提下把蓝图召回拿回来。预算是可加的:

    蓝图容忍 n_blue/N 张 + 白图容忍 n_white/N 张 = 总共 n/N 张

和共用一条线的总预算**完全一样**, 只是分配得更合理。

用法
----
  python training/page_threshold_split.py --genuine D:\probe\genuine_20k_ld9fix\summary.csv \
      --fake D:\probe\_clean_ld9fix\*_clean.csv --exclude D:\probe\exclude_evidence.txt --budget 5000

**只读**: 只读 CSV 和图片, 什么都不写。
"""

from __future__ import annotations

import argparse
import csv
import glob
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from locate_blue import is_blue_page  # noqa: E402


def page_of(path: str) -> str | None:
    """按像素判版式, 口径同 predict_tiled。用 draft 半解, 两万张也能跑。"""
    try:
        im = Image.open(path)
        im.draft("RGB", (im.size[0] // 4, im.size[1] // 4))
        return "blue" if is_blue_page(np.asarray(im.convert("RGB"))) else "white"
    except Exception:
        return None


def load(p: Path, col: str, drop: set[str], located_only: bool):
    out = []
    for r in csv.DictReader(open(p, encoding="utf-8-sig")):
        if located_only and (r.get("roi_amount_located") or "1").strip() != "1":
            continue
        nm = (r.get("image_name") or "").strip()
        if nm in drop:
            continue
        v = (r.get(col) or "").strip()
        if not v:
            continue
        try:
            out.append((float(v), r.get("image") or ""))
        except ValueError:
            pass
    return out


def thr_at(scores: list[float], budget: int) -> tuple[float, int]:
    """和 autoreject_threshold 同一套算法: 容忍 k 张, 线取降序第 k 个 +1e-6。"""
    g = sorted(scores, reverse=True)
    k = int(len(g) / budget)
    return (g[k] if k < len(g) else 0.0) + 1e-6, k


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="蓝白分开定阈值的收益(只读)")
    ap.add_argument("--genuine", type=Path, required=True)
    ap.add_argument("--fake", nargs="+", required=True, help="假图 summary/clean csv, 支持通配符")
    ap.add_argument("--exclude", type=Path, default=None)
    ap.add_argument("--col", default="tile_top3")
    ap.add_argument("--budget", type=int, default=5000)
    ap.add_argument("--limit-genuine", type=int, default=0,
                    help="真图只判前 N 张版式(0=全判)。**要定线就别限**, 抽样只能看分布不能定线")
    args = ap.parse_args()

    drop: set[str] = set()
    if args.exclude and args.exclude.exists():
        drop = {ln.strip().lstrip("\ufeff")
                for ln in args.exclude.read_text(encoding="utf-8-sig").splitlines() if ln.strip()}
        print(f"(已排除 {len(drop)} 个确认为假图的文件)")

    gen = load(args.genuine, args.col, drop, True)
    if args.limit_genuine:
        gen = gen[: args.limit_genuine]
    print(f"真图(定位成功) {len(gen)} 张 —— 正在按像素判版式, 稍等...", flush=True)
    by: dict[str, list[float]] = {"blue": [], "white": []}
    bad = 0
    for s, path in gen:
        pt = page_of(path)
        if pt is None:
            bad += 1
            continue
        by[pt].append(s)
    if bad:
        print(f"  {bad} 张读不了, 已跳过")

    shared_thr, shared_k = thr_at([s for s, _ in gen], args.budget)
    print()
    print("=" * 92)
    print(f"真图分数分布 (列={args.col}, 预算 1/{args.budget})")
    print("=" * 92)
    print(f"  {'版式':6s} {'张数':>7s} {'中位数':>9s} {'p99':>9s} {'最高分':>9s} {'容忍':>5s} {'单独定线':>10s}")
    thr: dict[str, float] = {}
    for pt in ("blue", "white"):
        v = sorted(by[pt], reverse=True)
        if not v:
            print(f"  {pt:6s} 没有样本")
            continue
        t, k = thr_at(v, args.budget)
        thr[pt] = t
        print(f"  {pt:6s} {len(v):7d} {v[len(v)//2]:9.4f} {v[len(v)//100]:9.4f} {v[0]:9.4f} "
              f"{k:5d} {t:10.4f}")
    print(f"\n  共用一条线(现状): {shared_thr:.4f}  (容忍 {shared_k} 张)")
    print(f"  分开定线之后总容忍 = " +
          " + ".join(f"{int(len(by[p])/args.budget)}" for p in ('blue', 'white') if by[p]) +
          f" = {sum(int(len(by[p])/args.budget) for p in ('blue','white'))} 张  "
          f"—— **和共用一条线的预算一样**")

    paths: list[Path] = []
    for pat in args.fake:
        paths.extend(sorted(Path(p) for p in glob.glob(pat)))
    if not paths:
        print("\n没匹配到假图 CSV, 只出真图分布")
        return

    print()
    print("=" * 92)
    print("假图召回: 共用一条线 vs 蓝白分开")
    print("=" * 92)
    print(f"  {'数据集':28s} {'版式':6s} {'张数':>6s} {'共用':>8s} {'分开':>8s} {'差':>7s}")
    print("-" * 92)
    tot_a = tot_b = tot_n = 0
    for p in paths:
        rows = load(p, args.col, set(), True)
        if not rows:
            continue
        pts = [page_of(path) for _s, path in rows]
        main_pt = "blue" if pts.count("blue") >= pts.count("white") else "white"
        if main_pt not in thr:
            continue
        a = sum(1 for s, _ in rows if s >= shared_thr)
        b = sum(1 for (s, _), q in zip(rows, pts) if q in thr and s >= thr[q])
        tot_a += a; tot_b += b; tot_n += len(rows)
        nm = p.parent.name if p.stem == "summary" else p.stem
        print(f"  {nm:28s} {main_pt:6s} {len(rows):6d} {a*100.0/len(rows):7.1f}% "
              f"{b*100.0/len(rows):7.1f}% {(b-a)*100.0/len(rows):+6.1f}")
    if tot_n:
        print("-" * 92)
        print(f"  {'加权合计':28s} {'':6s} {tot_n:6d} {tot_a*100.0/tot_n:7.1f}% "
              f"{tot_b*100.0/tot_n:7.1f}% {(tot_b-tot_a)*100.0/tot_n:+6.1f}")
    print()
    print("判读: 分开那一列明显更高 -> **同样的误杀预算下白捡一段召回**, 值得把阈值按版式拆开。")
    print("      差不多 -> 两种版式的真图分数分布本来就接近, 拆了没意义, 维持一条线更简单。")


if __name__ == "__main__":
    main()
