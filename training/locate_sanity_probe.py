r"""定位框"像不像一行金额" —— 先量, 别急着改。

要解决的是 `DEPLOY_SPEC` 短板 3 那件还没做的事:

> **★ 金额定位器会在非收据页上误触发。** top30 里第 20 名(豆浆机商品页)和第 22 名(近乎全灰的空图)
> **都通过了"定位成功"**, 然后线路B 给那个垃圾裁块打了 0.99。
> **便宜的非循环修法: 在信任线路B 之前先对定位框做合理性校验**(位置/尺寸/里面是不是数字)。
> **这条还没做。**

为什么先量不先改
----------------
"合理性校验"要卡阈值。**拍脑袋定的阈值会把正常收据也卡掉**, 而定位失败的代价很重 ——
短板 6 已经量过: 定位不到 = 线路B 直接失明, 而且**两条替代修法(roi-top 兜底 / 单开一条线)都已被否掉**,
`DEPLOY_SPEC` 的结论是"真正的修法只有一条: 把白图定位器做得更稳"。

所以这一份**只测量不改动**: 把高分图(疑似误触发)和普通收据的特征分布摆在一起,
**看哪个特征真的分得开**, 再拿数据定阈值。

量哪些特征
----------
`locate_amount` 返回的第 5 项就是选中那一行的**连通域列表**, 每个是 [x, y, w, h, area],
所以行内部的结构是现成的:

  n_glyph      这一行几个块        金额有好几位数字; 太少像噪点, 太多像整段文字
  aspect_med   块的宽高比中位      **数字是竖长的**, 大约 0.3~0.8; 方块/横条不是
  h_cv         块高的变异系数      同一行数字高度接近 -> 很小; 杂物拼出来的一行 -> 大
  fill         块宽之和 / 框宽     金额是连续的几位; 稀疏散落的块 fill 很低
  rel_h/rel_w  框高/框宽 占页比例
  rel_y        框顶 占页高比例

只读, 不写任何东西, 不改定位器。

用法
----
  python training/locate_sanity_probe.py --csv E:\SSP_Work\probe\accept_seeded\_lineA\summary.csv ^
      --roots E:\SSP_Work\probe D:\download2 --top 60 --sample 400
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import statistics as st
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _norm(s: str) -> str:
    return os.path.basename(s.strip().replace("\\", "/"))


def feats(rgb, loc) -> dict | None:
    """从定位框和它的连通域算特征。loc = (x0,y0,x1,y1,glyphs)。"""
    if not loc or len(loc) < 5:
        return None
    x0, y0, x1, y1, g = loc[0], loc[1], loc[2], loc[3], loc[4]
    if not g:
        return None
    h, w = rgb.shape[:2]
    hs = [c[3] for c in g if c[3] > 0]
    asp = [c[2] / c[3] for c in g if c[3] > 0]
    if not hs or not asp:
        return None
    bw = max(1, x1 - x0)
    return {
        "n_glyph": len(g),
        "aspect_med": st.median(asp),
        "h_cv": (st.pstdev(hs) / st.mean(hs)) if st.mean(hs) else 0.0,
        "fill": sum(c[2] for c in g) / bw,
        "rel_h": (y1 - y0) / h,
        "rel_w": bw / w,
        "rel_y": y0 / h,
    }


def _q(v: list[float], p: float) -> float:
    if not v:
        return float("nan")
    s = sorted(v)
    i = max(0, min(len(s) - 1, int(round(p * (len(s) - 1)))))
    return s[i]


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="定位框合理性特征分布(只测不改)")
    # ★★ 分组要用**线路B 的分数**, 不是线路A。
    #   短板 3 说的是"定位到垃圾之后**线路B** 给它打 0.99", 而线路A 压根不用定位器
    #   (它从整图随机取 16 块)。第一版拿 final_ai_score 分组, 等于没测到要测的东西。
    #   所以默认列改成 tile_top3, 并且请传**线路B 的 summary.csv**。
    ap.add_argument("--csv", type=Path, required=True,
                    help="**线路B** summary.csv(_lineB 那份) —— 按 tile_top3 分组")
    ap.add_argument("--roots", type=Path, nargs="+", required=True)
    ap.add_argument("--a-col", default="tile_top3",
                    help="分组用的分数列。线路B 用 tile_top3; 要看线路A 才用 final_ai_score")
    ap.add_argument("--top", type=int, default=60, help="看分数最高的前 N 张(疑似误触发)")
    ap.add_argument("--sample", type=int, default=400, help="随机抽多少张当对照")
    ap.add_argument("--seed", type=int, default=20260824)
    args = ap.parse_args()

    sys.path.insert(0, str(_HERE))
    import numpy as np                                     # noqa: E402
    from PIL import Image                                  # noqa: E402
    from locate_blue import locate_amount_auto             # noqa: E402

    rows = []
    with args.csv.open(encoding="utf-8-sig", newline="") as f:
        rd = csv.DictReader(f)
        have = rd.fieldnames or []
        key = "image_name" if "image_name" in have else "image"
        if key not in have or args.a_col not in have:
            raise SystemExit(f"!! 缺列; 表头 = {have}")
        for r in rd:
            try:
                rows.append((_norm(r[key]), float(r[args.a_col])))
            except (TypeError, ValueError):
                pass
    print(f"读到 {len(rows)} 行", flush=True)

    idx: dict[str, str] = {}
    for r in args.roots:
        if not r.is_dir():
            continue
        for dp, _, fns in os.walk(r):
            for fn in fns:
                idx.setdefault(fn, os.path.join(dp, fn))
    print(f"图片索引 {len(idx)} 个", flush=True)

    rows.sort(key=lambda t: -t[1])
    top = rows[: args.top]
    rest = rows[args.top:]
    random.Random(args.seed).shuffle(rest)
    samp = rest[: args.sample]
    print(f"高分组 {len(top)} 张(分数 {top[-1][1]:.4f} ~ {top[0][1]:.4f})"
          f"   对照组 {len(samp)} 张\n", flush=True)

    def collect(items, label):
        out, nloc, miss = [], 0, 0
        for nm, sc in items:
            p = idx.get(nm)
            if not p:
                miss += 1
                continue
            try:
                rgb = np.asarray(Image.open(p).convert("RGB"))
            except Exception:
                miss += 1
                continue
            loc, page = locate_amount_auto(rgb)
            if not loc:
                nloc += 1
                continue
            f = feats(rgb, loc)
            if f:
                f["_score"] = sc
                f["_name"] = nm
                f["_page"] = page
                out.append(f)
        print(f"  {label}: 定位成功 {len(out)}   定位失败 {nloc}   读不到 {miss}", flush=True)
        return out

    print("跑定位器(要解码像素, 慢一些)...", flush=True)
    A = collect(top, "高分组")
    B = collect(samp, "对照组")
    if not A or not B:
        raise SystemExit("!! 有一组是空的, 没法比")

    KEYS = ["n_glyph", "aspect_med", "h_cv", "fill", "rel_h", "rel_w", "rel_y"]
    print("\n" + "=" * 92)
    print(f"{'特征':<12}{'高分组 p10':>11}{'中位':>9}{'p90':>9}   |{'对照 p10':>10}{'中位':>9}{'p90':>9}")
    print("-" * 92)
    for k in KEYS:
        a = [x[k] for x in A]
        b = [x[k] for x in B]
        print(f"{k:<12}{_q(a,0.1):>11.3f}{_q(a,0.5):>9.3f}{_q(a,0.9):>9.3f}   |"
              f"{_q(b,0.1):>10.3f}{_q(b,0.5):>9.3f}{_q(b,0.9):>9.3f}")
    print("=" * 92)

    # ---- (a) 分位数带 ----
    print("\n[a] 拿**对照组 p2~p98** 当区间, 各特征能筛掉多少高分图:")
    print("    (对照组按定义就会被误伤约 4% —— 所以这一档天花板很低)")
    for k in KEYS:
        b = [x[k] for x in B]
        lo, hi = _q(b, 0.02), _q(b, 0.98)
        killA = sum(1 for x in A if not (lo <= x[k] <= hi))
        killB = sum(1 for x in B if not (lo <= x[k] <= hi))
        print(f"  {k:<12} 区间 [{lo:>7.3f}, {hi:>7.3f}]   筛掉高分 {killA:>3}/{len(A)}"
              f" = {100.0*killA/len(A):>5.1f}%   误伤对照 {100.0*killB/len(B):>4.1f}%")

    # ---- (b) 物理硬约束: 不是"罕见", 是"不可能" ----
    # 分位数带的问题在于, 它是从分布里推出来的, 而离群值本身几乎不改变分布 ——
    # 所以最离谱的那几张(fill=1.98 / n=86 / asp=2.53)反而卡不掉。
    # 硬约束不看分布, 只看这一行**能不能是一行金额**:
    HARD = [
        ("fill>1.05",      lambda x: x["fill"] > 1.05,
         "块宽之和超过框宽 —— 同一行不重叠的块**不可能**做到"),
        ("aspect>1.4",     lambda x: x["aspect_med"] > 1.4,
         "块比高还宽 —— 数字是竖长的"),
        ("aspect<0.15",    lambda x: x["aspect_med"] < 0.15,
         "块细成一条线 —— 是边框/分隔线, 不是字"),
        ("n_glyph>20",     lambda x: x["n_glyph"] > 20,
         "一行 20 多个块 —— 那是整段文字或噪点, 不是金额"),
        ("rel_h>0.10",     lambda x: x["rel_h"] > 0.10,
         "框高超过页高 10% —— 一行字不会这么高"),
        ("h_cv>0.45",      lambda x: x["h_cv"] > 0.45,
         "同一行里块高差得离谱 —— 不是一行字"),
    ]
    print("\n[b] **物理硬约束**(不是'罕见'而是'不可能', 所以误伤应当接近 0):")
    for nm, fn, why in HARD:
        ka = sum(1 for x in A if fn(x))
        kb = sum(1 for x in B if fn(x))
        print(f"  {nm:<14} 筛掉高分 {ka:>3}/{len(A)} = {100.0*ka/len(A):>5.1f}%"
              f"   误伤对照 {kb:>3}/{len(B)} = {100.0*kb/len(B):>4.1f}%   {why}")

    anyA = sum(1 for x in A if any(fn(x) for _, fn, _ in HARD))
    anyB = sum(1 for x in B if any(fn(x) for _, fn, _ in HARD))
    print(f"\n  ** 任一条命中 **  筛掉高分 {anyA}/{len(A)} = {100.0*anyA/len(A):.1f}%"
          f"   误伤对照 {anyB}/{len(B)} = {100.0*anyB/len(B):.1f}%")
    print("  -> 误伤低而筛出率可观, 这组硬约束就值得做成校验;")
    print("     误伤也高的话, 说明真实收据里本来就有这些形态, 阈值要放宽。")

    print("\n高分组里最可疑的 12 张(按与对照中位的偏离排):")
    med = {k: _q([x[k] for x in B], 0.5) for k in KEYS}
    spread = {k: max(1e-9, _q([x[k] for x in B], 0.9) - _q([x[k] for x in B], 0.1)) for k in KEYS}
    A.sort(key=lambda x: -sum(abs(x[k] - med[k]) / spread[k] for k in KEYS))
    for x in A[:12]:
        print(f"  {x['_name'][:44]:<46} 分 {x['_score']:.4f} {x['_page']:<5} "
              f"n={x['n_glyph']:<3} asp={x['aspect_med']:.2f} h_cv={x['h_cv']:.2f} "
              f"fill={x['fill']:.2f} rel_h={x['rel_h']:.3f}")

    print("\n★ 判读: 某个特征若能筛掉相当比例的高分图而误伤对照很低, 它就值得做成校验。")
    print("  都分不开的话, 说明高分不是'定位到垃圾'造成的, 短板 3 的修法要另找 —— 那也是结论。")


if __name__ == "__main__":
    main()
