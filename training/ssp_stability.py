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


_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _index_by_name(root: Path) -> dict[str, Path]:
    """把 --search-root 底下的图按**文件名**建索引 —— CSV 里的路径失效时靠它找回原图。"""
    idx: dict[str, Path] = {}
    dup = 0
    print(f"扫描 {root} 建文件名索引...", flush=True)
    for p in root.rglob("*"):
        if p.suffix.lower() in _EXTS and p.is_file():
            if p.name in idx:
                dup += 1
                continue                       # 重名只认第一个, 但下面会报出来
            idx[p.name] = p
    print(f"  索引到 {len(idx):,} 个文件名" + (f"  (**{dup:,} 个重名已跳过**)" if dup else ""))
    if dup:
        print("  重名意味着按名字找回的图**可能不是当初打分的那一张**, 这批结果要打个问号。")
    return idx


def _survey(paths: list[str]) -> None:
    """CSV 里的路径按父目录归类, 逐个报还在不在 —— 一眼看出是哪个目录没了。"""
    by_par: Counter = Counter()
    for p in paths:
        if p:
            by_par[str(Path(p).parent)] += 1
    print("\nCSV 里这些图当初放在哪儿, 现在还在不在:")
    for par, c in by_par.most_common(12):
        print(f"  {'在' if Path(par).is_dir() else '**没了**'}  {c:>7,d} 张  {par}")


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
    ap.add_argument("--search-root", type=Path, default=None,
                    help="CSV 里的路径失效时, 到这个目录底下**按文件名**把原图找回来")
    ap.add_argument("--verify-b-csv", type=Path, default=None,
                    help="按名字找回来的图**未必是当初打分的那一张**。给一份历史的线路B summary.csv, "
                         "用线路B(确定性)重打一遍比对 —— 对得上才说明找回来的是同一批图")
    ap.add_argument("--input", type=Path, default=None, help="或者直接给图片目录")
    ap.add_argument("--k", type=int, default=5, help="同一批图重打几次")
    ap.add_argument("--ssp-repo", type=Path,
                    default=Path(__import__("ssp_decide")._drive_fallback(r"D:\SSP")))
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
        if not picked:
            raise SystemExit(f"{args.from_csv} 里没有 >= {args.min_score} 分的图, 调低 --min-score。")
        idx = _index_by_name(args.search_root) if args.search_root else {}
        paths, missing = [], []
        for _, p in picked:
            q = Path(p) if p else None
            if q and q.is_file():
                paths.append(q)
            elif q and idx.get(q.name):
                paths.append(idx[q.name])      # 路径变了但图还在, 按名字找回来
            else:
                missing.append(p)
        if missing:
            print(f"\n  !!!! {len(missing)}/{len(picked)} 张原图按 CSV 里的路径找不到了")
            _survey([p for _, p in sc.values()])   # 报**整个池子**, 不只是挑中的这几百张
        if not paths:
            raise SystemExit(
                f"\n一张都没找到 —— **当初打分用的图目录已经不在了**。\n"
                f"  这不只是这个脚本跑不了: 池子没了就**再也无法重新标定阈值**,\n"
                f"  DEPLOY_SPEC 里每个数字都成了没法复现的历史值。\n\n"
                f"  先确认图还在不在别处, 在的话加 --search-root 按文件名找回:\n"
                f"    python training/ssp_stability.py --from-csv {args.from_csv} "
                f"--search-root D:\\download2 --min-score {args.min_score} --k {args.k} --out {args.out}\n"
                f"  真没了就改用 --input 指一个现有目录(但低分图量不出阈值附近的抖动)。")
        print(f"\n从 {args.from_csv.name} 里挑出 {len(paths)} 张 (>= {args.min_score} 分, "
              f"分数范围 {picked[len(paths) - 1][0]:.4f} ~ {picked[0][0]:.4f})")
        stage = args.out / "_imgs"
        _stage(paths, stage)
    elif args.input:
        stage = args.input
        print(f"直接用目录 {stage}")
    else:
        raise SystemExit("要么给 --from-csv, 要么给 --input")

    # ---- 先验明正身: 找回来的到底是不是当初那批图 ----
    # 线路B 是确定性的(同图同分), 所以它能当**指纹**用: 重打一遍对得上历史 CSV,
    # 就说明像素一模一样; 对不上说明按名字找回的是另一张图, 后面的抖动数字全都不能信。
    if args.verify_b_csv:
        lb = cfg["line_b"]
        bcol = lb.get("score_col", "tile_top3")
        vd = args.out / "_verifyB"
        print(f"\n验明正身: 用线路B 重打一遍, 和 {args.verify_b_csv.name} 比对...", flush=True)
        r = subprocess.run([sys.executable, str(_HERE / "predict_tiled.py"),
                            "--ssp-repo", str(args.ssp_repo), "--model", lb["model"],
                            "--input", str(stage), "--output_dir", str(vd),
                            *lb.get("flags", []), "--device", dev],
                           stdout=subprocess.DEVNULL, stderr=None)
        if r.returncode != 0:
            raise SystemExit(f"线路B 验证失败(退出码 {r.returncode})")
        hist = _read_scores(args.verify_b_csv, bcol)
        now = _read_scores(vd / "summary.csv", bcol)
        both = sorted(set(hist) & set(now))
        if not both:
            print("  !!!! 历史 CSV 和这次的结果没有共同文件名, 没法验证")
        else:
            # ★ 判据必须看**差多大**, 不能只看"是不是完全相等"。
            #   历史 CSV 多半是 cuda 上算的, 这里默认 cpu —— 同一张图跨设备本来就差 1e-4 量级。
            #   拿 1e-6 当门槛会把"同一张图"误判成"换了图"(第一版就是这么错的)。
            #   真换了图, 分数是满盘散落的, 差值该是 0.1~0.9 量级。
            diffs = sorted(abs(hist[n][0] - now[n][0]) for n in both)
            big = [n for n in both if abs(hist[n][0] - now[n][0]) > 0.01]
            exact = sum(1 for d in diffs if d <= 1e-9)
            print(f"  可比 {len(both)} 张 | 差值 中位 {diffs[len(diffs) // 2]:.6f} | "
                  f"p95 {diffs[int(len(diffs) * 0.95)]:.6f} | 最大 {diffs[-1]:.6f} | 完全相等 {exact} 张")
            if not big:
                print(f"  -> **是同一批图**。全部 {len(both)} 张差值都在 0.01 以内, "
                      f"这是跨设备浮点差异(标定多半在 cuda, 这次在 {dev}), 不是换了图。")
            else:
                print(f"  !!!! **{len(big)} 张差得离谱(>0.01)** —— 这些多半真不是当初那一张:")
                for n in sorted(big, key=lambda n: -abs(hist[n][0] - now[n][0]))[:5]:
                    print(f"       {n[:44]:46s} 历史 {hist[n][0]:.6f}  这次 {now[n][0]:.6f}")
                print(f"  另外 {len(both) - len(big)} 张是对得上的。只有上面这些要单独查。")

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
