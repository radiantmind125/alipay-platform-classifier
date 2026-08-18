r"""两次打分**逐张对比**: 定种子到底把分数动了多少, 阈值还站不站得住(只读 CSV)。

为什么不能直接比总数
--------------------
重搭的池子比当初小(有些原图按名字找不回来了), 而且**少掉的那批不是随机的**。
拿新的"自动拒 N 张"去和老的"26 张"比, 等于把**两件事混在一起**:
定种子带来的变化, 和池子换了一批图带来的变化。

所以这里只比**两次都在**的那些图 —— 同一批图, 只有打分方式变了。

具体验什么
----------
1. **分数往哪边动了。** 定种子之后多个模型看到**同一批小块**(以前各随机各的),
   而 `final_ai_score` 是多模型取 **max** —— 以前那个 max 里含有"哪个模型运气好抽到怪块"
   的成分, 现在没有了, 所以**预期分数整体略微偏低**。这里验证这个预期对不对。
2. **判定翻了多少张, 往哪个方向翻。**
3. **两档阈值上的比例(每万)还站不站得住** —— 总量口径是比例, 不是绝对张数。

用法
----
  python training/compare_runs.py --old D:\probe\gen100k_v7.csv ^
      --new D:\probe\accept_seeded\_lineA\summary.csv ^
      --exclude D:\probe\exclude_v4_full.txt
"""

from __future__ import annotations

import argparse
import csv
import statistics as st
import sys
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))


def _read(p: Path, col: str) -> dict[str, float]:
    """读一份 summary.csv。**打分失败的行要剔掉** —— 那种行 final_ai_score 是 0.0 但
    scores_json 是 {}, 混进来会被当成'0 分真图', 把对比结果带偏。"""
    out: dict[str, float] = {}
    skipped = 0
    with open(p, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            nm = (r.get("image_name") or "").strip() or Path(r.get("image") or "").name
            v = (r.get(col) or "").strip()
            if not nm or not v:
                continue
            if "scores_json" in r and (r.get("scores_json") or "").strip() in ("", "{}"):
                skipped += 1
                continue
            try:
                out[nm] = float(v)
            except ValueError:
                continue
    if skipped:
        print(f"  ({p.name}: 跳过 {skipped} 行打分失败的)")
    return out


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="两次打分逐张对比(只读 CSV)")
    ap.add_argument("--old", type=Path, required=True, help="老的线路A summary.csv(未定种子)")
    ap.add_argument("--new", type=Path, required=True, help="新的线路A summary.csv(已定种子)")
    ap.add_argument("--col", default=None, help="不给就用配置里的 score_col")
    ap.add_argument("--exclude", type=Path, default=None, help="验收用的排除名单")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None, help="把逐张差异写成 CSV")
    args = ap.parse_args()

    from ssp_decide import load_config
    cfg = load_config(args.config or (_HERE / "ssp_config.json"))
    la = cfg["line_a"]
    col = args.col or la.get("score_col", "final_ai_score")
    thr_s, thr_r = la["strict"], la["review"]

    old, new = _read(args.old, col), _read(args.new, col)
    drop: set[str] = set()
    if args.exclude and args.exclude.exists():
        drop = {ln.strip().lstrip("\ufeff")
                for ln in args.exclude.read_text(encoding="utf-8-sig").splitlines() if ln.strip()}
    both = sorted((set(old) & set(new)) - drop)
    if not both:
        raise SystemExit("两份 CSV 没有共同文件名, 没法比 —— 先确认 --col 和文件对不对")

    print(f"\n老的 {len(old):,} 张 | 新的 {len(new):,} 张 | "
          f"**两次都在且不在排除名单里的 {len(both):,} 张**(下面只看这些)")
    if drop:
        print(f"  (排除名单 {len(drop):,} 条)")
    only_old, only_new = len(set(old) - set(new) - drop), len(set(new) - set(old) - drop)
    if only_old or only_new:
        print(f"  (只在老的里 {only_old:,} 张, 只在新的里 {only_new:,} 张 —— 这些不参与对比)")

    d = [new[n] - old[n] for n in both]
    down = sum(1 for x in d if x < 0)
    print(f"\n分数变化(新 - 老):")
    print(f"  中位数 {st.median(d):+.6f} | 平均 {st.fmean(d):+.6f}")
    print(f"  变低的 {down:,} 张 ({down / len(d) * 100:.1f}%) | 变高的 {len(d) - down:,} 张")
    ad = sorted(abs(x) for x in d)
    print(f"  变动幅度: 中位 {ad[len(ad) // 2]:.6f} | p95 {ad[int(len(ad) * .95)]:.6f} | 最大 {ad[-1]:.6f}")
    print(f"  -> 预期是'整体略微偏低'(多模型不再各抽各的运气块)。"
          f"{'**符合预期**' if st.median(d) < 0 else '**和预期相反, 要查**'}")

    def tier(v: float) -> str:
        return "自动拒" if v >= thr_s else "人工复核" if v >= thr_r else "放行"

    n = len(both)
    co, cn = Counter(tier(old[x]) for x in both), Counter(tier(new[x]) for x in both)
    print(f"\n只看线路A 的话, 同一批 {n:,} 张图上的判定:")
    print(f"  {'判定':10s}{'老':>10s}{'每万':>9s}{'新':>10s}{'每万':>9s}{'变化':>9s}")
    print("  " + "-" * 56)
    for t in ("自动拒", "人工复核", "放行"):
        a, b = co.get(t, 0), cn.get(t, 0)
        print(f"  {t:10s}{a:>10,d}{a / n * 1e4:>9.1f}{b:>10,d}{b / n * 1e4:>9.1f}{b - a:>+9,d}")

    moved = [(x, old[x], new[x]) for x in both if tier(old[x]) != tier(new[x])]
    print(f"\n判定变了的: **{len(moved):,} 张 = {len(moved) / n * 100:.2f}%**")
    if moved:
        mv = Counter(f"{tier(o)} -> {tier(w)}" for _, o, w in moved)
        for k, c in mv.most_common():
            print(f"    {k:22s} {c:,} 张")
        print(f"  ★ 变严(放行->复核/拒) 和 变松(拒/复核->放行) 大致相当才正常; "
              f"一边倒说明分数整体平移了。")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f); w.writerow(["image_name", "老分", "新分", "差", "老判定", "新判定"])
            for x in sorted(both, key=lambda k: -abs(new[k] - old[k])):
                w.writerow([x, f"{old[x]:.6f}", f"{new[x]:.6f}", f"{new[x] - old[x]:+.6f}",
                            tier(old[x]), tier(new[x])])
        print(f"\n-> 逐张差异 {args.out}")

    print(f"\n判读: 每万的两个数(老 vs 新)差得不多 -> 阈值可以照用, 只需把 DEPLOY_SPEC 的数字更新一下;")
    print(f"      差得明显 -> 要在**重搭的这个池子上重新标定**(池子已经找回来了, 标得动)。")


if __name__ == "__main__":
    main()
