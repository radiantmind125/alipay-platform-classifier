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


def _scores(p: Path, col: str = "final_ai_score", require_located: bool = False) -> list[tuple[float, str]]:
    out = []
    for r in csv.DictReader(open(p, encoding="utf-8-sig")):
        if require_located and (r.get("roi_amount_located") or "1").strip() != "1":
            continue                      # 金额没定位到 = 这条线不出信号
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
    ap.add_argument("--col", default="final_ai_score",
                    help="用哪一列打分。线路B(predict_tiled)的 CSV 里有 tile_max/tile_top3/tile_mean, "
                         "上线用哪个聚合就填哪个(现在线路B 用 tile_top3), 这样不用重新打分")
    ap.add_argument("--require-located", action="store_true",
                    help="线路B 专用: 只统计金额定位成功的图(定位失败本来就不出信号)")
    ap.add_argument("--exclude", type=Path, default=None,
                    help="一个文本文件, 每行一个要排除的图片文件名(用来剔掉已确认是假图的'真图')")
    ap.add_argument("--exclude-ext", nargs="*", default=None, metavar="EXT",
                    help="按扩展名把某一类图移出**自动拒的适用人群**(不是当假图剔掉!)。"
                         "典型用法 `--exclude-ext .jpeg`: 实测 .jpeg 只占 0.6% 却霸占高分区"
                         "(线路A >=0.94 占比 5.43% vs png/jpg 的 0.00-0.03%), 因为它们是"
                         "**过了修图App(醒图/美图秀秀)的图**。"
                         "★这只有在**线上真的实现了'有修图痕迹就不自动拒, 一律走人工复核'**时才成立 —— "
                         "否则就是在人为压低误杀率, 会标出过于激进的阈值。")
    args = ap.parse_args()

    drop = set()
    if args.exclude and args.exclude.exists():
        # utf-8-sig: PowerShell 的 `Out-File -Encoding utf8` 会写 BOM, 不吃掉的话第一行永远匹配不上
        drop = {ln.strip().lstrip("﻿")
                for ln in args.exclude.read_text(encoding="utf-8-sig").splitlines() if ln.strip()}
        print(f"(已排除 {len(drop)} 个确认为假图的文件)")

    g = sorted(_scores(args.genuine, args.col, args.require_located), key=lambda t: -t[0])
    if drop:
        g = [(s, nm) for s, nm in g if Path(nm).name not in drop]
    if args.exclude_ext:
        exts = {e.lower() if e.startswith(".") else "." + e.lower() for e in args.exclude_ext}
        before = len(g)
        g = [(s, nm) for s, nm in g if Path(nm).suffix.lower() not in exts]
        print(f"(已把 {before - len(g)} 张 {sorted(exts)} 移出自动拒的适用人群 —— "
              f"**它们不是被当成假图, 而是改走人工复核**; 上线必须真的这么实现)")
    if not g:
        raise SystemExit("真图 summary 里没读到分数")
    n = len(g)
    # ★ 名字要按**文件**取, 不能只按父目录名。
    #   踩过的坑: 两个清洗后的 CSV 放在同一个目录里(_clean_ld7\wanwhite_...csv 和
    #   _clean_ld7\qwenwhite_...csv), 用父目录名当 key 时**字典把前一个静默覆盖了**,
    #   表里只剩一列, 而且列名还是目录名, 完全看不出少了一个假图集。
    #   约定: 文件名叫 summary.csv 就用父目录名(老用法), 否则用文件名本身。
    def _key(p: Path) -> str:
        return p.parent.name if p.stem == "summary" else p.stem

    keys = [_key(p) for p in args.fake]
    dup = sorted({k for k in keys if keys.count(k) > 1})
    if dup:
        raise SystemExit(f"假图集重名, 会互相覆盖: {', '.join(dup)} —— 把文件改名或分开放")
    fakes = {k: _scores(p, args.col, args.require_located) for k, p in zip(keys, args.fake)}
    if len(fakes) != len(args.fake):
        raise SystemExit(f"给了 {len(args.fake)} 个假图集, 只认出 {len(fakes)} 个")

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

    # 按文件格式分开看: 训练集被 reencode 成了全 JPEG, 真实提交里的 PNG 是**分布外**输入。
    # PNG 是无损的, 高频细节全保留, 而我们的模型正是看高频 -> PNG 真图可能被系统性打高分。
    by_ext: dict[str, list[float]] = {}
    for s, nm in g:
        ext = Path(nm).suffix.lower() or "(无)"
        by_ext.setdefault(ext, []).append(s)
    if len(by_ext) > 1:
        print()
        print("按文件格式分开看真图分数(看 PNG 是不是被系统性打高):")
        print("格式      张数    占比    中位数   90分位   99分位    最高分   >=0.94占比")
        for ext, ss in sorted(by_ext.items(), key=lambda kv: -len(kv[1])):
            ss = sorted(ss)
            m = len(ss)
            q = lambda f: ss[min(m - 1, int(m * f))]  # noqa: E731
            hi = sum(1 for x in ss if x >= 0.94) / m
            print(f"{ext:8s} {m:6d} {m / n * 100:6.1f}% {q(0.5):8.4f} {q(0.90):8.4f} "
                  f"{q(0.99):8.4f} {ss[-1]:9.4f} {hi * 100:9.2f}%")
        print("=> 若 PNG 的高分位/最高分明显高于 JPG, 说明**训练集被统一成JPEG导致PNG是分布外**,")
        print("   这不是模型不行, 是数据格式缺口 —— 把无损PNG真图加进训练即可, 自动拒的线能往下放。")

    print()
    print(f"分数最高的 {min(args.show_top, n)} 张真图(自动拒的线就是被这几张卡住的):")
    for s, nm in g[:args.show_top]:
        print(f"  {s:.4f}  {nm}")
    print()
    print("**强烈建议人工看一眼上面这几张** —— 我们已知真图池里混着疑似假图。")
    print("如果它们本来就是假的, 真实误杀比这里算出来的还低, 分数线可以往下放。")


if __name__ == "__main__":
    main()
