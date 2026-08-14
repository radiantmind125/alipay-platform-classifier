r"""把**定位框的几何**和**线路B 的分数**连起来 —— 回答"抓不到的那些是不是框就找错了"(只读)。

要回答的问题
------------
按版式拆开召回, 发现一件反直觉的事:

| | n | 严格档 | 复核档 |
|---|---|---|---|
| 白图 | 441 | 71.9% | **92.0%** |
| 蓝图 | 471 | 72.0% | **98.7%** |

**严格档两者几乎一样, 差距只出现在复核档。**
所以不是"白图信号普遍弱"(那样严格档就该先拉开), 而是
**有一小撮白图假样本, 阈值怎么放松都够不到。**

而 `locate_audit` 量出来的假图白图几何里正好有这么一撮: **框高是正常值的 3 倍**。
一个很自然的解释是: **造样本的时候定位器就找错了框**, VAE 改的地方压根不是金额区 ——
裁块里没有篡改, 分数自然贴着 0, 放松阈值也够不到。

**若属实, 那些是坏测试样本而不是模型盲区**, 白图复核档的召回被低估了。
论证和"切错的裁块"是同一套: 判据只看**框的几何**, 不看模型分数, 所以不构成循环论证。

判据从**真图**的几何分位数来, 不是拍脑袋定的。

用法
----
  python training/box_score_join.py --geom-genuine D:\probe\locate_geom.csv ^
      --geom-fake D:\probe\locate_geom_fake.csv ^
      --scores D:\probe\_clean_ld9fix\*_clean.csv ^
      --thresholds 0.9883 0.7838

**只读**: 只读这几个 CSV, 什么都不写。
"""

from __future__ import annotations

import argparse
import csv
import glob
import sys
from pathlib import Path

import numpy as np

_KEYS = ("h_rel", "aspect", "w_rel")
_LABEL = {"h_rel": "框高/页高", "aspect": "宽高比", "w_rel": "框宽/页宽"}


def _read(p: Path) -> list[dict]:
    with open(p, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="定位框几何 x 线路B 分数(只读)")
    ap.add_argument("--geom-genuine", type=Path, required=True, help="真图的 locate_audit 输出")
    ap.add_argument("--geom-fake", type=Path, required=True, help="假图的 locate_audit 输出")
    ap.add_argument("--scores", nargs="+", required=True, help="假图分数 CSV, 支持通配符")
    ap.add_argument("--col", default="tile_top3")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.9883, 0.7838],
                    help="按从严到松给, 默认就是当前两档的线路B 阈值")
    ap.add_argument("--lo-pct", type=float, default=1.0, help="真图几何的下界分位")
    ap.add_argument("--hi-pct", type=float, default=99.0, help="真图几何的上界分位")
    args = ap.parse_args()

    # 1) 从真图几何定出每种版式的"合理区间"
    gen = [r for r in _read(args.geom_genuine) if r.get("located") == "1"]
    bounds: dict[str, dict[str, tuple[float, float]]] = {}
    for page in ("blue", "white"):
        sub = [r for r in gen if r["page"] == page]
        if not sub:
            continue
        bounds[page] = {}
        for k in _KEYS:
            v = np.asarray([float(r[k]) for r in sub if r[k] != ""], dtype=float)
            bounds[page][k] = (float(np.percentile(v, args.lo_pct)),
                               float(np.percentile(v, args.hi_pct)))
        b = bounds[page]
        print(f"{page:6s} 真图 {len(sub):6d} 张 -> 合理区间(p{args.lo_pct:g}~p{args.hi_pct:g}): "
              + "  ".join(f"{_LABEL[k]} {b[k][0]:.3f}~{b[k][1]:.3f}" for k in _KEYS))

    # 2) 读假图分数
    files: list[str] = []
    for pat in args.scores:
        hit = glob.glob(pat)
        if not hit:
            print(f"  !!!! {pat} 没匹配到文件")
        files.extend(hit)
    score: dict[str, float] = {}
    cell_of: dict[str, str] = {}
    for f in sorted(files):
        stem = Path(f).stem
        for r in _read(Path(f)):
            nm = (r.get("image_name") or "").strip()
            v = (r.get(args.col) or "").strip()
            if nm and v:
                try:
                    score[nm] = float(v)
                    cell_of[nm] = stem
                except ValueError:
                    pass
    if not score:
        raise SystemExit("假图分数一个都没读到 —— 确认 --scores 和 --col")
    print(f"\n假图分数 {len(score)} 张, 来自 {len(files)} 个文件")

    # 3) 连接
    fake = _read(args.geom_fake)
    miss = sum(1 for r in fake if r["image_name"] not in score)
    if miss:
        print(f"  !!!! 有 {miss} 张在几何表里但**分数表里没有**, 这些不参与统计 —— "
              f"两边不是同一批图的话结论就不成立, 值得回头看一眼")

    for page in ("white", "blue"):
        rows = [r for r in fake if r["page"] == page and r["image_name"] in score]
        if not rows:
            continue
        b = bounds.get(page)
        groups: dict[str, list[float]] = {"框合理": [], "框不合理": [], "定位失败": []}
        why: dict[str, int] = {}
        for r in rows:
            s = score[r["image_name"]]
            if r["located"] != "1":
                groups["定位失败"].append(s)
                continue
            bad = []
            for k in _KEYS:
                if r[k] == "" or not b:
                    continue
                v = float(r[k])
                if v < b[k][0] or v > b[k][1]:
                    bad.append(_LABEL[k])
            if bad:
                groups["框不合理"].append(s)
                why[" ".join(bad)] = why.get(" ".join(bad), 0) + 1
            else:
                groups["框合理"].append(s)

        n_all = sum(len(v) for v in groups.values())
        print(f"\n=== {page} === 假图 {n_all} 张")
        head = f"{'分组':10s}{'张数':>6s}{'占比':>8s}{'中位分':>9s}"
        for t in args.thresholds:
            head += f"{'>=' + format(t, '.4f'):>12s}"
        print(head)
        print("-" * (33 + 12 * len(args.thresholds)))
        for g, v in groups.items():
            if not v:
                continue
            a = np.asarray(v)
            line = f"{g:10s}{len(v):>6d}{len(v)/n_all*100:>7.1f}%{np.median(a):>9.4f}"
            for t in args.thresholds:
                line += f"{(a >= t).mean()*100:>11.1f}%"
            print(line)
        if why:
            print("  框不合理的原因:")
            for k, c in sorted(why.items(), key=lambda kv: -kv[1]):
                print(f"    {k:28s} {c:4d} 张")

    print()
    print("怎么读:")
    print("  **框不合理 / 定位失败**那两行的召回若**明显低于**框合理那一行,")
    print("  说明抓不到的主要是**造样本时就切错了地方**的那些 —— 属于坏测试样本, 不是模型盲区,")
    print("  白图复核档的召回是被它们拖低的。")
    print("  若三行召回**差不多**, 那这个解释不成立, 白图那 8% 是真的漏检, 要如实写进短板。")
    print()
    print("★ 就算证实了, **也不能直接把它们从对外数字里剔掉**再报一个更好看的数 ——")
    print("  要报就得写清楚: 剔的是'金额区没被改到'的样本, 剔的依据是框的几何而不是模型分数。")


if __name__ == "__main__":
    main()
