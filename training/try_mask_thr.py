r"""试不同的掩膜阈值, 看蓝图顶部那片粘连能不能拆开。

起因
----
蓝图拼音检测只有 82.49%(白图 96.31%)。存掩膜图看出来的:

```
蓝底渐变区   转账成功 和它上面的 zhuanzhangchenggong **糊成一坨**
            收款方 / 付款方式 / 回首页 也全糊住
同一张图
白卡片区     立即通知收款人 和它的拼音 **清清楚楚分开**, 全都认出来了
```

★★★★★ 粘上之后那一坨又高又大, 既不算"小块", 底下也没有可配对的大块,
   **结构上就不可能被判成注音**。why_no_pinyin 打出来顶部两带 annotation 数正好是 0。

为什么会粘
----------
`text_mask` 的阈值是**绝对值** 28。对比度越高, 笔画边缘的抗锯齿像素越容易
也过线, 笔画就越**胖**。而拼音离它底下的汉字只有 1~3 像素, 胖一点就粘上了。

★ 所以**提高阈值**应该能把它们拆开。但提太高会把笔画细的字整个丢掉,
  所以要扫一遍看哪个值最好, **不能拍脑袋定**。

要同时看两件
------------
```
赚   蓝图顶部那两带能认出多少注音(现在是 0), 以及整页认出有拼音的行数
赔   白图会不会掉 —— 白图现在 96.31%, 掉了就得不偿失
```

用法
----
    python try_mask_thr.py --src D:\download2\pinyin_hits_blue --n 40
    python try_mask_thr.py --src D:\download2\pinyin_hits      --n 40
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from erase_pinyin import (EXTS, annotation_labels, local_background,
                          text_mask)
from make_pinyin_pairs import _pinyin_bands, _rows

THRS = [28, 36, 44, 52, 60, 70]


def run(img, thr: int):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    mask = text_mask(gray, local_background(gray), diff_thr=thr)
    n, _lab, st, _c = cv2.connectedComponentsWithStats(mask, connectivity=8)
    labels, kept, _cnt, big = annotation_labels(mask)
    H = img.shape[0]
    top = 0
    for i in kept:
        if st[i][1] < H * 0.25:
            top += 1
    rows_with = 0
    n_rows = 0
    if kept:
        bands = _pinyin_bands(kept, st, big)
        rows = _rows(big, bands)
        n_rows = len(rows)
        for (y0, y1) in rows:
            above = [(x, y) for x, y in bands
                     if y <= y0 + 2 and (y0 - y) < (y1 - y0)]
            rows_with += bool(above)
    return n - 1, len(kept), top, n_rows, rows_with


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=17)
    a = ap.parse_args()

    files = sorted(p for p in a.src.iterdir() if p.suffix.lower() in EXTS)
    random.seed(a.seed)
    files = random.sample(files, min(a.n, len(files)))

    print("=" * 74)
    print("  TRY MASK THRESHOLD  (ASCII only - safe to paste)")
    print("=" * 74)
    print(f"  {a.src.name}   抽 {len(files)} 张")
    print("  28 是现在的默认值, 其余是试的")
    print()

    agg = {t: [0, 0, 0, 0, 0] for t in THRS}
    ok = 0
    for i, p in enumerate(files, 1):
        img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        ok += 1
        for t in THRS:
            r = run(img, t)
            for k in range(5):
                agg[t][k] += r[k]
        if i % 10 == 0:
            print(f"    {i}/{len(files)}", flush=True)

    print()
    print(f"  跑了 {ok} 张")
    print()
    hd = (f"  {'阈值':<7}{'连通块':>9}{'判成注音':>10}{'顶部1/4的':>11}"
          f"{'切出的行':>10}{'有拼音的行':>12}")
    print(hd)
    print("  " + "-" * 58)
    base = agg[THRS[0]]
    for t in THRS:
        v = agg[t]
        mark = "   <- 现在的默认" if t == 28 else ""
        d = ""
        if t != 28 and base[4]:
            d = f"  ({(v[4]-base[4])/base[4]:+.1%})"
        print(f"  {t:<7}{v[0]:>9,}{v[1]:>10,}{v[2]:>11,}{v[3]:>10,}"
              f"{v[4]:>12,}{mark}{d}")
    print()
    print("  ★ 连通块数变多 = 原来粘住的被拆开了, 这正是我们要的")
    print("  ★★ 但要盯住'有拼音的行' —— 阈值提太高会把细笔画的字整个丢掉,")
    print("     那时候块数还是多, 认出来的行却会掉下来。**挑那一列最高的。**")
    print()
    print("  ★★★★★ 蓝图白图**都要跑**。蓝图看能赚多少, 白图看会不会赔。")


if __name__ == "__main__":
    main()
