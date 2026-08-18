r"""**同图重打 K 次**, 量线路A 的分数抖动到底有多大(只读源图)。

为什么要单独量这个
------------------
回归测试发现: 同一批 597 张图, **什么都没改**连跑两次, **570 张分数不一样**,
最大偏差 0.161256, **1 张判定翻了**。原因在 SSP 官方取块法 `utils/patch.py`:

    每次随机取 64 个 32x32 小块 -> 按纹理复杂度排序 -> **只留最简单的那一块**
    repeat 16 次求平均 -> 再在多个模型之间取 max

一张 1080x2400 的截图有约 **248 万**个可能的取块位置, 16 次 repeat 一共只看了 1,024 个,
**占 0.041%**。而且收据截图有大片纯色背景, "最简单的块"基本就是随机挑一块空白区域 ——
挑到哪一块, 分数就不一样。

★ **不能拿全是低分的样本来量。** 597 张真图几乎都在 0.00 附近, 那里抖不动也没意义。
  真正决定命运的是**阈值附近**那一小撮。所以本脚本默认从**已算好的 CSV 里挑高分图**来测。

判读
----
- **判定翻转率**: K 次里判定不完全一致的图占多少 —— 这就是"**靠运气**"的那部分进件
- **危险带**: 分数 ±2 倍标准差跨过阈值的图占多少 —— 比翻转率更稳健的估计
- **标准差随分数分档**: 抖动在中高分区最大, 而阈值恰好在那里

用法
----
  # 从 10 万池的线路A 结果里挑出 0.3 分以上的图, 每张重打 5 次
  python training/ssp_stability.py --from-csv D:\probe\gen100k_v7.csv --min-score 0.3 ^
      --k 5 --out D:\probe\stab

  # 或者直接给一个目录
  python training/ssp_stability.py --input D:\probe\speedtest --k 5 --out D:\probe\stab

**只读源图**: 只往 --out 写(用硬链接搭临时目录, 不复制不改动原图)。
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import statistics as st
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))


def _stage(paths: list[Path], dst: Path) -> int:
    """搭一个临时目录指向选中的图。优先硬链接, 不行才复制。"""
    dst.mkdir(parents=True, exist_ok=True)
    seen: dict[str, Path] = {}
    n = 0
    for p in paths:
        if p.name in seen:                       # 重名在这个仓库里已经坑过三次, 直接停
            raise SystemExit(f"**重名**: {p.name}\n  {seen[p.name]}\n  {p}\n"
                             "  两个不同目录下同名文件会互相覆盖, 结果会静悄悄地错。")
        seen[p.name] = p
        tgt = dst / p.name
        if not tgt.exists():
            try:
                os.link(p, tgt)
            except OSError:
                shutil.copy2(p, tgt)
        n += 1
    return n


def _read_scores(csv_path: Path, col: str) -> dict[str, tuple[float, str]]:
    """读一份 summary.csv -> {文件名: (分数, 原图全路径)}"""
    out: dict[str, tuple[float, str]] = {}
    with open(csv_path, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            nm = (r.get("image_name") or "").strip() or Path(r.get("image") or "").name
            try:
                out[nm] = (float(r[col]), (r.get("image") or "").strip())
            except (KeyError, ValueError, TypeError):
                continue
    return out


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="同图重打 K 次, 量线路A 的分数抖动(只读源图)")
    ap.add_argument("--from-csv", type=Path, default=None, help="已算好的线路A summary.csv, 从里面挑图")
    ap.add_argument("--min-score", type=float, default=0.3,
                    help="只测这个分以上的图 —— **低分区抖不抖都不影响判定**, 测了浪费时间")
    ap.add_argument("--max-n", type=int, default=400, help="最多测多少张")
    ap.add_argument("--input", type=Path, default=None, help="或者直接给图片目录")
    ap.add_argument("--k", type=int, default=5, help="同一批图重打几次")
    ap.add_argument("--ssp-repo", type=Path, default=Path(r"D:\SSP"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from ssp_decide import load_config, line_a_root
    cfg = load_config(args.config or (_HERE / "ssp_config.json"))
    la = cfg["line_a"]
    col = la.get("score_col", "final_ai_score")
    thr_s, thr_r = la["strict"], la["review"]
    dev = args.device or cfg.get("device", "cpu")
    args.out.mkdir(parents=True, exist_ok=True)

    # ---- 选图 ----
    if args.from_csv:
        sc = _read_scores(args.from_csv, col)
        picked = sorted((v for v in sc.values() if v[0] >= args.min_score),
                        key=lambda t: -t[0])[:args.max_n]
        missing = [p for _, p in picked if not p or not Path(p).is_file()]
        if missing:
            print(f"  !!!! {len(missing)} 张原图找不到了(CSV 里的路径已失效), 这些跳过")
            print(f"       例如: {missing[0]}")
        paths = [Path(p) for _, p in picked if p and Path(p).is_file()]
        if not paths:
            raise SystemExit(f"{args.from_csv} 里没有 >= {args.min_score} 分且原图还在的图。"
                             f"\n  调低 --min-score, 或改用 --input 直接给目录。")
        print(f"从 {args.from_csv.name} 里挑出 {len(paths)} 张 (>= {args.min_score} 分, "
              f"分数范围 {picked[-1][0]:.4f} ~ {picked[0][0]:.4f})")
        stage = args.out / "_imgs"
        _stage(paths, stage)
    elif args.input:
        stage = args.input
        print(f"直接用目录 {stage}")
    else:
        raise SystemExit("要么给 --from-csv, 要么给 --input")

    # ---- 重打 K 次 ----
    a_root = line_a_root(cfg)
    runs: list[dict[str, float]] = []
    print(f"\n重打 {args.k} 次 | 设备 {dev} | model_root={a_root}")
    print("  (什么都不改, 纯粹同样的命令跑 K 遍 —— 分数应该一模一样, 看看是不是)")
    for i in range(1, args.k + 1):
        od = args.out / f"_run{i}"
        t = time.time()
        r = subprocess.run([sys.executable, "predict_all_models.py", "--model_root", a_root,
                            "--model_pattern", la.get("model_pattern", "Net_epoch_best.pth"),
                            "--input", str(stage), "--output_dir", str(od), "--device", dev],
                           cwd=str(args.ssp_repo),
                           stdout=subprocess.DEVNULL, stderr=None)
        if r.returncode != 0:
            raise SystemExit(f"第 {i} 次打分失败(退出码 {r.returncode})")
        runs.append({k: v[0] for k, v in _read_scores(od / "summary.csv", col).items()})
        print(f"  第 {i}/{args.k} 次完成 ({time.time() - t:.0f}s, {len(runs[-1])} 张)", flush=True)

    # ---- 统计 ----
    names = sorted(set.intersection(*(set(r) for r in runs)))
    if not names:
        raise SystemExit("K 次结果对不上名字, 没法比")
    rows, flip, band = [], [], []
    sds, sd_by_bin = [], defaultdict(list)
    for nm in names:
        vs = [r[nm] for r in runs]
        mu, sd = st.fmean(vs), (st.stdev(vs) if len(vs) > 1 else 0.0)
        decs = {("自动拒" if v >= thr_s else "人工复核" if v >= thr_r else "放行") for v in vs}
        sds.append(sd)
        sd_by_bin[min(int(mu * 5), 4)].append(sd)
        if len(decs) > 1:
            flip.append((nm, min(vs), max(vs), sorted(decs)))
        # 危险带: 均值 ±2sd 跨过任一条线 —— 比"这 K 次恰好翻了"更稳健
        if any(mu - 2 * sd < t <= mu + 2 * sd for t in (thr_s, thr_r)):
            band.append(nm)
        rows.append({"image_name": nm, "mean": f"{mu:.6f}", "sd": f"{sd:.6f}",
                     "min": f"{min(vs):.6f}", "max": f"{max(vs):.6f}",
                     "range": f"{max(vs) - min(vs):.6f}", "判定是否一致": int(len(decs) == 1),
                     **{f"run{i+1}": f"{v:.6f}" for i, v in enumerate(vs)}})

    sp = args.out / "stability.csv"
    with open(sp, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    n = len(names)
    print(f"\n{'=' * 62}\n{n} 张图, 每张打了 {args.k} 次\n{'=' * 62}")
    print(f"\n每张图 K 次分数的标准差:")
    print(f"  中位数 {st.median(sds):.6f} | 平均 {st.fmean(sds):.6f} | "
          f"最大 {max(sds):.6f}")
    print(f"  单张最大极差(max-min): {max(float(r['range']) for r in rows):.6f}")

    print(f"\n标准差按分数分档(**阈值在 0.65/0.80, 正好落在抖得最厉害的区间**):")
    for b in range(5):
        v = sd_by_bin.get(b, [])
        if v:
            print(f"  {b * 0.2:.1f}~{(b + 1) * 0.2:.1f}  {len(v):>5d} 张   标准差中位数 {st.median(v):.6f}")

    print(f"\n★ 判定不一致的图: **{len(flip)} 张 / {n} 张 = {len(flip) / n * 100:.1f}%**")
    for nm, lo, hi, ds in flip[:8]:
        print(f"    {nm[:44]:46s} {lo:.4f}~{hi:.4f}  {'/'.join(ds)}")
    if len(flip) > 8:
        print(f"    ...另外 {len(flip) - 8} 张")
    print(f"\n★ 危险带(均值 ±2 倍标准差跨过阈值): **{len(band)} 张 = {len(band) / n * 100:.1f}%**")
    print("  这些图的判定**基本靠运气**, 同一张图再交一次可能就是另一个结果。")

    print(f"\n-> {sp}")
    print("\n注: 这里测的是**已经筛过的高分图**, 不是全体进件。要换算成线上比例, "
          "得乘以高分图在进件里的占比。")


if __name__ == "__main__":
    main()
