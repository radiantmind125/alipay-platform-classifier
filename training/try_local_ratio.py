"""试新的注音判法(和配对的大块比, 而不是和整页比), 但**不改默认行为**。

要解决的
--------
大标题上面的拼音检测不到:

```
标题字 60 像素, 它的拼音 25 像素, 整页 big_h 才 30
老判法 小 = h <= 0.55*big_h = 16.5   ->  25 不算小  ->  当成正文, 不裁
```

实测漏掉的全是大号标题和按钮: `zhangdanxiangqing 账单详情`、
`zhuanzhangchenggong 回首页`、`zaizhuanyibi 再转一笔`。
白图漏 3.69%, 蓝图漏 17.51%。蓝图版面被大标题主导, 所以特别严重。

新判法: `h <= 0.55 * 配对的那个大块的高`, 25 对 60 就成立了。

★★★★★ 但这是**白蓝共用的几何**, 白图现在 96.31% 已经很好, 改坏了赔不起。
   所以 `annotation_labels(local_ratio=)` 默认还是老行为,
   **这个脚本只是拿两种判法跑一遍对比, 一个像素都不改。**

两件都要看
----------
```
赚      有拼音的行多认出来多少 —— 蓝图那 17% 能回来多少
赔      有没有把**正文**也判成注音 —— 判错了就是啃掉汉字
```

★★★★★ 第二件比第一件要紧。之前记过: `方 = 亠 + 万`、`注 = 氵 + 主`,
   汉字的分离笔画和拼音长得一模一样, 判宽了就会
   `付款方式 -> 付款万式`、`转账备注 -> 转账备汪`。
   挡住这个的是"同一高度凑得成一排"(MIN_RUN)那一关, 不是大小那一关,
   所以放宽大小**理论上**还有那一关兜着 —— 但理论归理论, **要看图**。

用法
----
    python try_local_ratio.py --src D:\\download2\\pinyin_hits_blue --n 60
    python try_local_ratio.py --src D:\\download2\\pinyin_hits_blue --n 12 ^
        --save D:\\alipay-ai-data\\local_ratio_look
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from erase_pinyin import (EXTS, annotation_labels, local_background,  # noqa: E402
                          text_mask)
from make_pinyin_pairs import _pinyin_bands, _rows  # noqa: E402


def analyse(img, local_ratio: bool):
    """返回(注音块数, 行数, 有拼音的行数, 标签图, kept)。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    mask = text_mask(gray, local_background(gray))
    _n, _lab, st, _c = cv2.connectedComponentsWithStats(mask, connectivity=8)
    labels, kept, _cnt, big = annotation_labels(mask, local_ratio=local_ratio)
    if not kept:
        return 0, 0, 0, labels, []
    bands = _pinyin_bands(kept, st, big)
    rows = _rows(big, bands)
    withp = 0
    for (y0, y1) in rows:
        above = [(x, y) for x, y in bands
                 if y <= y0 + 2 and (y0 - y) < (y1 - y0)]
        withp += bool(above)
    return len(kept), len(rows), withp, labels, kept


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--save", type=Path, default=None,
                    help="存对比图: 判成注音的块涂红, 肉眼看有没有啃到汉字")
    # ★ 默认只存前 12 张, 而**要看的偏偏是暴涨的那几张**, 多半不在前 12 里。
    ap.add_argument("--only-blown", action="store_true",
                    help="只存块数暴涨的那几张 —— 要看的就是它们")
    ap.add_argument("--blow", type=float, default=1.8,
                    help="涨到多少倍算暴涨。1.8 是随手定的, 看分布再调")
    a = ap.parse_args()

    files = sorted(p for p in a.src.iterdir() if p.suffix.lower() in EXTS)
    random.seed(a.seed)
    files = random.sample(files, min(a.n, len(files)))

    print("=" * 70)
    print("  TRY LOCAL RATIO  (ASCII only - safe to paste)")
    print("=" * 70)
    print(f"  {a.src.name}   抽 {len(files)} 张")
    print("  老判法 = 和整页 big_h 比 (现在的默认, 一个像素没改)")
    print("  新判法 = 和配对的那个大块比")
    print()

    tot = {"old_g": 0, "new_g": 0, "old_r": 0, "new_r": 0,
           "old_wp": 0, "new_wp": 0}
    gain_rows, lose_rows, blew_up = 0, 0, []
    ratios: list[tuple[float, str, int, int]] = []
    ok = 0
    for i, p in enumerate(files, 1):
        img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        ok += 1
        og, orow, owp, olab, okept = analyse(img, False)
        ng, nrow, nwp, nlab, nkept = analyse(img, True)
        tot["old_g"] += og; tot["new_g"] += ng
        tot["old_r"] += orow; tot["new_r"] += nrow
        tot["old_wp"] += owp; tot["new_wp"] += nwp
        if nwp > owp:
            gain_rows += 1
        elif nwp < owp:
            lose_rows += 1
        # ★ 注音块数暴涨 = 很可能把正文也判进去了, 单独记下来看图
        ratio = ng / og if og else 0.0
        if og > 0:
            ratios.append((ratio, p.name, og, ng))
        blown = og > 0 and ratio > a.blow
        if blown:
            blew_up.append((p.name, og, ng))
        if a.save and (blown if a.only_blown else i <= 12):
            # ★ 红色 = 判成注音(也就是**会被擦掉**的那些块)。
            #   老新各存一张, 对着看新判法多涂红了哪些 —— 多涂在拼音上才对,
            #   多涂到汉字上就是要啃字了。
            a.save.mkdir(parents=True, exist_ok=True)
            for tag, lab, k in (("old", olab, okept), ("new", nlab, nkept)):
                vis = img.copy()
                if k:
                    vis[np.isin(lab, k)] = (0, 0, 255)
                out = a.save / f"{tag}_{p.stem[:40]}.png"
                cv2.imencode(".png", vis)[1].tofile(str(out))
        if i % 20 == 0:
            print(f"    {i}/{len(files)}", flush=True)

    print()
    print(f"  跑了 {ok} 张")
    print()
    print(f"  {'':<16}{'老判法':>10}{'新判法':>10}{'变化':>10}")
    print("  " + "-" * 46)
    for k, name in (("g", "判成注音的块"), ("r", "切出来的行"),
                    ("wp", "认出有拼音的行")):
        o, n = tot[f"old_{k}"], tot[f"new_{k}"]
        d = f"{n-o:+,}" + (f"  ({(n-o)/o:+.1%})" if o else "")
        print(f"  {name:<16}{o:>10,}{n:>10,}{d:>16}")
    print()
    print(f"  ★ 有拼音的行变多的图 {gain_rows} 张, 变少的 {lose_rows} 张")
    print()
    # ★★★★★ 1.8 倍这条线是**随手定的**。先看看分布:
    #   要是绝大多数图都贴近 1.0, 只有一两张冲到 2 倍以上, 那是真的异常;
    #   要是从 1.0 到 2.0 连成一片, 那这条线本身就没意义, 得换判法。
    if ratios:
        ratios.sort(reverse=True)
        qs = [ratios[int(len(ratios) * f)][0]
              for f in (0.0, 0.05, 0.25, 0.5, 0.75)]
        print("  注音块数 新/老 的分布(从大到小):")
        print(f"    最大 {qs[0]:.2f}   95分位 {qs[1]:.2f}   75分位 {qs[2]:.2f}"
              f"   中位 {qs[3]:.2f}   25分位 {qs[4]:.2f}")
        print("    最高的几张:")
        for r, nm, o, n in ratios[:5]:
            print(f"      {r:>5.2f}x  {o:>5} -> {n:>5}   {nm[:44]}")
        print()
    if blew_up:
        print(f"  ★★★★★ 注音块数暴涨(超过 1.8 倍)的有 {len(blew_up)} 张 ——")
        print("     **很可能把正文也判成注音了**, 这些必须看图:")
        for nm, o, n in blew_up[:10]:
            print(f"       {o:>5} -> {n:>5}   {nm[:46]}")
    else:
        print("  ★ 没有注音块数暴涨的图 —— 初步看没把正文大片judge进去")
    print()
    print("  ★★★★★ 数只能说明'多认了多少', **说明不了有没有啃掉汉字**。")
    print("     加 --save 存图, 红色是判成注音的块, 必须肉眼确认红色只盖在拼音上。")


if __name__ == "__main__":
    main()
