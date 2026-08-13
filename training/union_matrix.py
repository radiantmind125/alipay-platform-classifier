r"""八个格子 x 两档的**并集召回**一次跑完, 出一张表(只读)。

为什么要有这个
--------------
`combined_threshold --fa/--fb` 一次只算**一个**假图集, 而且算完就 return, 不再打误杀。
八格两档 = 16 次调用。手敲有两个真实的坑:

1. **配错对**: 线路A 的 CSV 和线路B 的 CSV 必须是**同一批图**。配错了按裸文件名对齐后交集为 0,
   召回直接变成 0 —— 这个坑踩过一次(线路A 用了旧目录的分数, 三格并集召回全成 0)。
   而且 `seedblue_v7` 旁边就躺着 `seedblue_v7_stale`, 名字只差一点。
2. **分母没剔干净**: `--fa/--fb` 模式下全集取的是 `set(FA)`, 也就是**线路A 那份**。
   我们的 `_clean.csv` 已经把同源图剔掉了, 线路A 那份**没有** —— 不给 `--fake-exclude`,
   那几张会留在分母里, 把每一格的召回都压低一点。

**这个脚本不重新实现任何算法**, 只是按正确的配对反复调用 `combined_threshold.py`,
再把它的输出解析成一张表。数字和手敲 16 次**逐字一致**, 因为跑的就是同一份代码。

配对规则
--------
  <clean>\<格>_<标记>_clean.csv    -> 线路B(已剔同源)
  <probe>\<格>_v7\summary.csv      -> 线路A
  <clean>\<格>_<标记>_leaked.txt   -> 同源名单, 有就自动传给 --fake-exclude

用法
----
  python training/union_matrix.py --probe D:\probe --clean D:\probe\_clean_ld9fix ^
      --a D:\probe\gen100k_v7.csv --b D:\probe\gen100k_b.csv ^
      --exclude D:\probe\exclude_big3.txt --budgets 5000 1000

  # 先看它打算怎么配对, 不真跑:
  python training/union_matrix.py ... --dry-run

**只读**: 只跑 combined_threshold 并解析输出, 不写任何文件。
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

# 格名 -> 中文显示名; 真实生成器(万相/豆包/千问)和我们自己合成的 VAE 要分开加权,
# 因为对外报的"覆盖率"指的是真实生成器那几格。
_CN = {
    "wanwhite": "万相白", "wanblue": "万相蓝",
    "seedwhite": "豆包白", "seedblue": "豆包蓝",
    "qwenwhite": "千问白", "qwenblue": "千问蓝",
    "white": "合成白(VAE)", "blue": "合成蓝(VAE)",
}
_REAL = ("wan", "seed", "qwen")          # 真实生成器前缀

_RE_N = re.compile(r"并集召回\(同一批假图 (\d+) 张; 线路B 能出信号的 (\d+) 张\)")
_RE_A = re.compile(r"只靠线路A 抓到\s*:\s*(\d+)\s*=\s*([\d.]+)%")
_RE_B = re.compile(r"只靠线路B 抓到\s*:\s*(\d+)\s*=\s*([\d.]+)%")
_RE_U = re.compile(r"并集\(实际能抓到的\): (\d+) = ([\d.]+)%")
_RE_TA = re.compile(r"线路A 预算 1/\d+:.*?阈值 ([\d.]+)")
_RE_TB = re.compile(r"线路B 预算 1/\d+:.*?阈值 ([\d.]+)")
_RE_EX = re.compile(r"假图侧已剔除 (\d+) 张")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="八格 x 两档并集召回一次跑完(只读)")
    ap.add_argument("--probe", type=Path, required=True, help="放各 <格>_v7\\summary.csv 的根目录")
    ap.add_argument("--clean", type=Path, required=True, help="放 <格>_*_clean.csv 的目录")
    ap.add_argument("--a", type=Path, required=True, help="线路A 真图 CSV")
    ap.add_argument("--b", type=Path, required=True, help="线路B 真图 CSV")
    ap.add_argument("--exclude", type=Path, required=True, help="真图侧排除名单")
    ap.add_argument("--budgets", type=int, nargs="+", default=[5000, 1000])
    ap.add_argument("--a-suffix", default="_v7", help="线路A 的目录后缀")
    ap.add_argument("--dry-run", action="store_true", help="只打印配对结果, 不真跑")
    args = ap.parse_args()

    cleans = sorted(args.clean.glob("*_clean.csv"))
    if not cleans:
        raise SystemExit(f"{args.clean} 里一个 *_clean.csv 都没有 —— 先确认 --clean 路径")

    # 配对
    cells: list[tuple[str, Path, Path, Path | None]] = []
    for cp in cleans:
        stem = cp.stem[: -len("_clean")]                 # 如 blue_ld9fix
        cell = stem.split("_")[0]                        # 如 blue
        fa = args.probe / f"{cell}{args.a_suffix}" / "summary.csv"
        leaked = cp.with_name(f"{stem}_leaked.txt")
        if not fa.exists():
            print(f"  !!!! {_CN.get(cell, cell)}: 线路A 找不到 {fa} —— **这一格跳过**")
            continue
        cells.append((cell, fa, cp, leaked if leaked.exists() else None))

    print(f"配对结果({len(cells)}/{len(cleans)} 格):")
    for cell, fa, fb, lk in cells:
        print(f"  {_CN.get(cell, cell):12s} A={fa}")
        print(f"  {'':12s} B={fb}   同源名单={'有' if lk else '无(不剔)'}")
    if len(cells) < len(cleans):
        print(f"\n  !!!! 有 {len(cleans) - len(cells)} 格配不上对, **表是不完整的**, 别当全集读。")
    if args.dry_run:
        print("\n(--dry-run: 没有真跑)")
        return

    here = Path(__file__).resolve().parent
    res: dict[int, dict[str, dict]] = {b: {} for b in args.budgets}
    thr: dict[int, tuple[str, str]] = {}

    for b in args.budgets:
        for cell, fa, fb, lk in cells:
            cmd = [sys.executable, str(here / "combined_threshold.py"),
                   "--a", str(args.a), "--b", str(args.b), "--b-col", "tile_top3",
                   "--exclude", str(args.exclude), "--budget", str(b),
                   "--fa", str(fa), "--fb", str(fb)]
            if lk:
                cmd += ["--fake-exclude", str(lk)]
            p = subprocess.run(cmd, capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
            out = p.stdout or ""
            u = _RE_U.search(out)
            if not u:
                print(f"  !!!! 1/{b} {_CN.get(cell, cell)}: 没解析到并集召回, "
                      f"退出码 {p.returncode}")
                print("       " + (out.strip().splitlines() or ["(无输出)"])[-1])
                if p.stderr.strip():
                    print("       stderr: " + p.stderr.strip().splitlines()[-1])
                continue
            n = _RE_N.search(out)
            a, bb = _RE_A.search(out), _RE_B.search(out)
            ta, tb = _RE_TA.search(out), _RE_TB.search(out)
            ex = _RE_EX.search(out)
            if ta and tb:
                thr[b] = (ta.group(1), tb.group(1))
            res[b][cell] = {
                "n": int(n.group(1)) if n else 0,
                "nb": int(n.group(2)) if n else 0,
                "a": float(a.group(2)) if a else 0.0,
                "b": float(bb.group(2)) if bb else 0.0,
                "u": float(u.group(2)),
                "hit": int(u.group(1)),
                "leak": int(ex.group(1)) if ex else 0,
            }
            print(f"  1/{b} {_CN.get(cell, cell):12s} n={res[b][cell]['n']:4d} "
                  f"并集 {res[b][cell]['u']:5.1f}%", flush=True)

    # 先按习惯顺序排已知的几格, **认不出的格名一律追加在后面** ——
    # 早先这里只遍历写死的名单, 名字对不上的格子会**从表里静静消失**,
    # 而表看上去仍然是完整的。宁可多一行怪名字, 不能少一行。
    known = ("wanwhite", "wanblue", "seedwhite", "seedblue",
             "qwenwhite", "qwenblue", "white", "blue")
    got = {c for b in args.budgets for c in res[b]}
    order = [c for c in known if c in got] + sorted(got - set(known))

    print()
    print("=" * 92)
    print("并集召回(线路A 或 线路B 任一命中即算抓到 —— 上线就是这么跑的)")
    print("=" * 92)
    for b in args.budgets:
        t = thr.get(b)
        print(f"  1/{b}: 线路A 阈值 {t[0] if t else '?'} | 线路B 阈值 {t[1] if t else '?'}")
    print()
    head = f"{'格':14s}{'n':>6s}{'同源剔':>7s}"
    for b in args.budgets:
        head += f"{'1/' + str(b):>28s}"
    print(head)
    sub = " " * 27
    for _ in args.budgets:
        sub += f"{'仅A':>9s}{'仅B':>9s}{'并集':>10s}"
    print(sub)
    print("-" * 92)
    for cell in order:
        row = f"{_CN.get(cell, cell):14s}"
        first = next((res[b][cell] for b in args.budgets if cell in res[b]), None)
        row += f"{first['n'] if first else 0:>6d}{first['leak'] if first else 0:>7d}"
        for b in args.budgets:
            d = res[b].get(cell)
            row += (f"{d['a']:>8.1f}%{d['b']:>8.1f}%{d['u']:>9.1f}%") if d else f"{'-':>27s}"
        print(row)

    # 加权: 真实生成器那几格 和 全部格
    print("-" * 92)
    for label, keys in (("真实生成器加权", [c for c in order if c.startswith(_REAL)]),
                        ("全部格加权", order)):
        row = f"{label:14s}"
        tot = sum(res[args.budgets[0]][c]["n"] for c in keys if c in res[args.budgets[0]])
        row += f"{tot:>6d}{'':>7s}"
        for b in args.budgets:
            ks = [c for c in keys if c in res[b]]
            n = sum(res[b][c]["n"] for c in ks)
            if not n:
                row += f"{'-':>27s}"
                continue
            wa = sum(res[b][c]["a"] * res[b][c]["n"] for c in ks) / n
            wb = sum(res[b][c]["b"] * res[b][c]["n"] for c in ks) / n
            wu = sum(res[b][c]["u"] * res[b][c]["n"] for c in ks) / n
            row += f"{wa:>8.1f}%{wb:>8.1f}%{wu:>9.1f}%"
        print(row)
    print()
    print("★ 对外报覆盖率用**真实生成器加权**那一行 —— 合成(VAE)两格是我们自己造的,")
    print("  模型训练时见过同一个生成器, 算进去会把数字抬高。")
    print("★ 误杀要另外跑一次不带 --fa/--fb 的 combined_threshold ——")
    print("  给了 --fa/--fb 它就只算召回然后 return, 不再打误杀。")


if __name__ == "__main__":
    main()
