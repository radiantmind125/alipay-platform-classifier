r"""**报出来的这些图, 彼此像不像?** —— 定"篡改 vs 渲染变体"的最后一招。

为什么这条能定性, 而前面五条都不能
----------------------------------
2026-09-02 试过五个判据, **一个都分不开**"篡改"和"新版本换了渲染":

| 试过的 | 为什么不行 |
|---|---|
| 图内 金额负号 vs 日期连字符 | 连字符只有 3px, 量不准 |
| 数字偏离同群幅度(中位 9.2%) | 是真差异, 但两个假说都预测得出 |
| 金额行内数字是否等宽 | 真图本身变异就有 17.8% |
| 取值聚集(1.0238 出现 35 次跨 11 种分辨率) | 字体的字形比例固定, 缩放模板也保留比例 |
| 同群离群(93/108) | 低渗透率的新版本同样解释得通 |

**失败是结构性的**: 用一个工具篡改、用一套字体渲染, **都会产生内部一致的字形**。

★★ **但有一件事只有篡改做得到**:
**把一份模板改成很多张不同的收据, 于是这些收据彼此长得几乎一样。**
**渲染变体永远做不到这个** —— 不同的人、不同的订单号、不同的时间,
换个字体渲染出来仍然是**互不相同**的收据。

所以: **报出来的这些图之间是不是近似重复?**
- **是** -> 同一份模板改出来的 -> **篡改, 定死了**
- **否** -> 它们是各不相同的真实交易 -> 更像是渲染变体

怎么比
------
用**低分辨率灰度指纹**(缩到 32x64 再二值化), 这样**局部小改动(比如只动金额)不影响整体**,
而版式/内容不同的图会明显不同。两两比汉明距离。

★ 只比**报出来的那些**(一百来张), 不需要扫全池, 秒级出结果。

★ 边界: 近似重复也可能是**同一个人反复提交同一张真图**(那是重复提交, 不是篡改)。
所以下面会**顺带报出这些近似组里金额是不是不一样** ——
**金额不同而其余几乎一样, 才是"一份模板改出多张"的铁证。**

用法
----
  python training/flagged_kinship.py --flagged D:\probe\minus_aug.csv ^
      --pool D:\download2\OtherImages --out D:\probe\kinship.csv
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from locate_blue import locate_amount_auto  # noqa: E402

_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _sig(path: str, w: int = 32, h: int = 64):
    """整图的低分辨率指纹。缩到 32x64 灰度, 再按中位数二值化。

    ★ 缩得这么小, **只改金额那一小块基本不影响指纹** ——
      正是我们要的: 认出"同一份底图"而不被局部改动干扰。
    """
    try:
        with Image.open(path) as im:
            g = im.convert("L").resize((w, h), Image.BILINEAR)
    except Exception:
        return None
    a = np.asarray(g, dtype=np.float32)
    return (a > np.median(a)).flatten()


def _amount_sig(path: str):
    """金额区的小图, 用来看"这两张的金额是不是同一个数"。"""
    try:
        rgb = np.asarray(Image.open(path).convert("RGB"))
    except Exception:
        return None
    loc, _pg = locate_amount_auto(rgb)
    if not loc:
        return None
    x0, y0, x1, y1, _ = loc
    sub = rgb[y0:y1, x0:x1]
    if sub.size == 0:
        return None
    g = Image.fromarray(sub).convert("L").resize((48, 16), Image.BILINEAR)
    a = np.asarray(g, dtype=np.float32)
    return (a > np.median(a)).flatten()


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="报出的图彼此是不是近似重复")
    ap.add_argument("--flagged", type=Path, required=True)
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--thr", type=float, default=0.06,
                    help="整图指纹的汉明距离阈值(占比), 小于它就算近似重复。默认 0.06")
    ap.add_argument("--amt-thr", type=float, default=0.15,
                    help="金额区指纹距离大于它就算'金额不一样'。默认 0.15")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rows = list(csv.DictReader(args.flagged.read_text(encoding="utf-8-sig").splitlines()))
    names = [r["image_name"] for r in rows]
    print(f"报出名单 {len(names)} 张")

    print(f"清点 {args.pool} ...", flush=True)
    idx = {}
    for dp, _, fns in os.walk(args.pool):
        for fn in fns:
            if os.path.splitext(fn)[1].lower() in _EXTS:
                idx[fn] = os.path.join(dp, fn)

    sigs, asigs, keep = {}, {}, []
    for n in names:
        p = idx.get(n)
        if not p:
            continue
        s = _sig(p)
        if s is None:
            continue
        sigs[n] = s
        asigs[n] = _amount_sig(p)
        keep.append(n)
    print(f"算出指纹 {len(keep)} 张", flush=True)
    if len(keep) < 2:
        raise SystemExit("!! 不够两张, 没法比")

    # 两两比
    pairs = []
    for a, b in itertools.combinations(keep, 2):
        d = float(np.mean(sigs[a] != sigs[b]))
        if d <= args.thr:
            ad = (float(np.mean(asigs[a] != asigs[b]))
                  if (asigs.get(a) is not None and asigs.get(b) is not None) else None)
            pairs.append((a, b, d, ad))
    print(f"\n整图指纹距离 <= {args.thr} 的配对: **{len(pairs)}** 对")

    # 连通分量 = 近似重复组
    parent = {n: n for n in keep}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, _d, _ad in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    groups = {}
    for n in keep:
        groups.setdefault(find(n), []).append(n)
    multi = {k: v for k, v in groups.items() if len(v) > 1}
    print(f"近似重复组: **{len(multi)}** 组, 覆盖 {sum(len(v) for v in multi.values())} 张")

    # ★ 关键: 组内金额是不是不一样
    n_tmpl = 0
    print()
    for gi, (_k, mem) in enumerate(sorted(multi.items(), key=lambda kv: -len(kv[1])), 1):
        diffs = []
        for a, b in itertools.combinations(mem, 2):
            if asigs.get(a) is not None and asigs.get(b) is not None:
                diffs.append(float(np.mean(asigs[a] != asigs[b])))
        amt_diff = max(diffs) if diffs else 0.0
        tag = ("★★ **整体几乎一样但金额不同 -> 同一份模板改出多张 -> 篡改**"
               if amt_diff > args.amt_thr else "整体和金额都一样 -> 多半是同一张重复提交")
        if amt_diff > args.amt_thr:
            n_tmpl += 1
        print(f"组 {gi}: {len(mem)} 张, 金额区最大差异 {amt_diff:.3f}  {tag}")
        for n in mem[:4]:
            print(f"    {n[:56]}")

    print(f"\n{'='*66}")
    if n_tmpl:
        print(f"★★ **{n_tmpl} 组是'整体几乎一样、金额却不同'** ——")
        print("   **渲染变体做不到这个**(不同交易换个字体渲染仍然互不相同),")
        print("   **只有'拿一份模板改出多张'才会这样。到此可以定性为篡改。**")
    else:
        print("没有'整体一样但金额不同'的组。")
        print("  -> **这条定不了性**: 报出的图彼此并不相像, 说明它们是各不相同的真实收据,")
        print("     那么'新版本换了渲染'仍然是可能的解释。**还是得靠后台核对真实交易。**")

    if args.out and pairs:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["a", "b", "整图距离", "金额区距离"])
            for a, b, d, ad in sorted(pairs, key=lambda x: x[2]):
                w.writerow([a, b, round(d, 4), round(ad, 4) if ad is not None else ""])
        print(f"\n配对明细 -> {args.out}")


if __name__ == "__main__":
    main()
