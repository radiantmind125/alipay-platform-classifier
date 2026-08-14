r"""某一类图单独路由时, **自动拒多少、人工看多少** —— 按两种实现分别算(只读)。

背景
----
`.jpeg` 这个扩展名在两条线的高分尾巴里都严重超配(27 倍 / 41 倍)。
把它从**定线的池子**里拿掉, 严格档的线从 0.9440/0.9883 降到 0.9064/0.9811,
改金额覆盖 71.9% -> **76.9%**, 而误杀几乎不动(2.9 -> 2.7/万)。

**但"拿出定线池子"和"运行时怎么处理它"是两件事。**

运行时**必须做的只有一件**: `.jpeg` **不许自动拒**。
否则重标之后的低阈值会把这一撮成片拒掉 —— 它们本来就是分数高的那一类。

至于要不要顺带**全部**送人工, 是另一回事, 两种实现差别很大:

- **方案A 全送**: `.jpeg` 一律进人工。简单, 但人工量里有一大半是**纯按文件名**送的, 不是模型报的信号。
- **方案B 只免拒**: `.jpeg` 不自动拒, 但复核线照常作用于它。
  改名进这个桶的**高分**假图仍然过复核线 -> 仍然到人工手上, **一样不构成绕过**;
  而分数低的 `.jpeg` 真图直接放行, 不占人工。

**两种方案的覆盖率是一样的**(覆盖由阈值决定, 而阈值来自定线池子), 差的只是人工量。
这个脚本把两者的量算出来, 好让"要不要全送"变成一个有数字的选择。

用法
----
  python training/route_volume.py --a D:\probe\gen100k_v7.csv --b D:\probe\gen100k_b.csv ^
      --exclude D:\probe\exclude_big3.txt --group D:\probe\jpeg_names.txt ^
      --strict 0.9064 0.9811 --review 0.6596 0.7549

**只读**: 只读 CSV 和名单, 什么都不写。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def _load(p: Path, col: str, require_located: bool) -> dict[str, float]:
    out: dict[str, float] = {}
    with open(p, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if require_located and (r.get("roi_amount_located") or "1").strip() != "1":
                continue
            nm = (r.get("image_name") or "").strip() or Path(r.get("image") or "").name
            v = (r.get(col) or "").strip()
            if nm and v:
                try:
                    out[nm] = float(v)
                except ValueError:
                    pass
    return out


def _names(p: Path | None) -> set[str]:
    if not p or not p.exists():
        return set()
    return {ln.strip().lstrip("\ufeff")
            for ln in p.read_text(encoding="utf-8-sig").splitlines() if ln.strip()}


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="单独路由某一类图时的自动拒/人工量(只读)")
    ap.add_argument("--a", type=Path, required=True, help="线路A 真图 CSV")
    ap.add_argument("--b", type=Path, required=True, help="线路B 真图 CSV")
    ap.add_argument("--a-col", default="final_ai_score")
    ap.add_argument("--b-col", default="tile_top3")
    ap.add_argument("--exclude", type=Path, default=None, help="已确认假图的名单")
    ap.add_argument("--group", type=Path, required=True, help="要单独路由的那一类的文件名清单")
    ap.add_argument("--strict", type=float, nargs=2, required=True, metavar=("A", "B"))
    ap.add_argument("--review", type=float, nargs=2, required=True, metavar=("A", "B"))
    args = ap.parse_args()

    drop = _names(args.exclude)
    grp = _names(args.group)
    A = {k: v for k, v in _load(args.a, args.a_col, False).items() if k not in drop}
    B = {k: v for k, v in _load(args.b, args.b_col, True).items() if k not in drop}
    if not A:
        raise SystemExit("线路A 没读到分数")
    uni = set(A)
    g = uni & grp
    print(f"(排除名单 {len(drop)} | 单独路由的那一类 {len(grp)} 个, 其中在池子里的 {len(g)} 张)")
    print(f"真图 {len(uni):,} 张; 线路B 能出信号的 {len(set(B) & uni):,} 张\n")

    def hit(nm: str, ta: float, tb: float) -> bool:
        return A[nm] >= ta or (nm in B and B[nm] >= tb)

    sa, sb = args.strict
    ra, rb = args.review
    n = len(uni)
    non = uni - g

    # 基线: 不做任何单独路由(仅供对照; 注意阈值本来就是按"拿掉这一类"重标的)
    base_rej = {k for k in uni if hit(k, sa, sb)}
    base_rev = {k for k in uni if hit(k, ra, rb)} - base_rej

    # 方案A: 这一类**全部**进人工, 永不自动拒
    a_rej = {k for k in non if hit(k, sa, sb)}
    a_rev = ({k for k in non if hit(k, ra, rb)} - a_rej) | g

    # 方案B: 这一类**只免自动拒**, 复核线照常作用
    b_rej = a_rej
    b_rev = {k for k in uni if hit(k, ra, rb)} - b_rej

    print(f"{'方案':30s}{'自动拒':>8s}{'/万':>8s}{'人工':>8s}{'/万':>8s}")
    print("-" * 62)
    for label, rej, rev in (("对照 这一类不单独处理", base_rej, base_rev),
                            ("方案A 这一类全部进人工", a_rej, a_rev),
                            ("方案B 这一类只免自动拒", b_rej, b_rev)):
        print(f"{label:30s}{len(rej):>8d}{len(rej)/n*1e4:>8.1f}{len(rev):>8d}{len(rev)/n*1e4:>8.1f}")

    only = len(b_rev & g)
    print(f"\n方案B 里, 这一类**因为分数高而**进人工的有 {only} 张 "
          f"({only/max(1,len(g))*100:.1f}% 的该类图)")
    print(f"方案A 比方案B 多送 {len(a_rev) - len(b_rev)} 张 = {(len(a_rev)-len(b_rev))/n*1e4:.1f}/万, "
          f"**换来的覆盖率是一样的**")
    print()
    print("★ 对照那一行只是给个参照 —— 阈值本来就是按'把这一类拿出定线池子'重标出来的,")
    print("  真拿它去上线, 这一类会被成片自动拒, 那正是必须免拒的原因。")
    print("★ 这里的量都是**每万张真图**。真假图落在复核带的还要另加, 取决于欺诈率(我们不知道)。")


if __name__ == "__main__":
    main()
