r"""**图内比对**: 金额的负号 vs 同一张图里日期行的连字符。

为什么需要这一条
----------------
2026-09-01 在八月的图上跑负号判据, 报出率从七月的 **14.2/万** 涨到 **33.5/万**(2.4 倍)。
但逐张看拼图之后, **不能把这 108 张报成检出**, 因为有两条证据指向"渲染变体"而不是欺诈:

1. **八月整个分布都上移了**: 5 位金额的中位从 **0.7021 -> 0.7143**, p99.5 从 0.7381 -> 0.7500。
   主体在往上走 —— 这是 **app 版本变化**的样子。
2. **报出的图在日期上是散开的**(0803~0810)。欺诈批次会在时间上聚集
   (那个 149 张的重复组是 72 分钟内跑完的), **版本灰度则是散开的**。
3. 而且那 23 张的金额**压倒性是整数**(100/200/300/1000/2000), 那正是真实转账的样子。

**跨人群比较到这里已经失效了** —— 拿七月标的阈值去判八月的图, 分不清
"负号被拉长了"和"新版本把负号画长了"。

★★★ 2026-09-01 实测结论: **这条路走不通, 别再试了**
--------------------------------------------------
在 1,061 张真图上量完:

| | 中位 | p5 | p95 | 变异系数 |
|---|---|---|---|---|
| 金额负号宽比 | 0.7000 | 0.6667 | 0.7381 | 24.7% |
| 日期横杠宽比 | 0.8235 | 0.3333 | 1.1220 | 32.8% |
| **两者之比** | 0.8358 | **0.6149** | **2.0924** | **80.2%** |

**"两者之比"在真图上根本不是常数** —— p5~p95 跨 0.61 到 2.09, 差 3.4 倍。
`03.jpg` 是 1.0909, **稳稳落在真图范围里面, 分不出来**。

**原因**: **日期的连字符只有 3 px 高**, 那个尺度上二值化差一个像素, 量出的宽度就差约 10%。
这一点在最早逐张量 03/04 的时候就发现过("细横杠的宽度对二值化阈值极敏感"),
**建这条测法时却忘了** —— 拿一个已知不可靠的量当参照系。

**替代方案**: 不做图内比对, 改做**同分辨率 + 同编码器指纹的同群比对** ——
同一批(同机型同版本)里绝大多数在 0.71 而少数在 1.10, 那少数就是异常;
整批都在 1.10 则是渲染变体。那条控制住了 app 版本和机型, 且不需要第二个字形。

以下保留原始设计说明, 仅供记录。

这个脚本怎么绕开
----------------
**只在同一张图内部比**:

  `金额负号宽 / 金额数字宽`   对上   `日期连字符宽 / 日期数字宽`

- **两个一起长** -> 同一次渲染里就是这样 -> **渲染变体, 不是篡改**
- **只有金额的负号长, 日期连字符正常** -> **同一张图内部不一致 -> 篡改**

**这条不依赖任何外部基线** —— 不看七月的数、不看 app 版本、不看机型分布。
两个字形来自同一次渲染、同一个字体、经历同样的压缩和缩放, 唯一的差别只有"有没有被改过"。

★ 已知的一处不对称: 金额里的 `−` 和日期里的 `-` **很可能不是同一个码位**
(减号 vs 连字符), 字体设计上宽度本来就不同。
**所以这个比值不该等于 1, 但它应当是个常数** —— 要看的是**它稳不稳**, 不是它等于几。

用法
----
  python training/minus_vs_date.py --input D:\download2\OtherImages --n 2000 ^
      --since 20260801 --out D:\probe\mvd_aug.csv
  # 只测某一批(比如上一步报出来的那些)
  python training/minus_vs_date.py --input D:\download2\OtherImages ^
      --names D:\probe\minus_aug_names.txt --out D:\probe\mvd_flagged.csv

**只读**: 只读图片, 只往 --out 写。
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from locate_blue import locate_amount_auto      # noqa: E402
from glyph_baseline import _reextract           # noqa: E402

_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_TS = re.compile(r"_(\d{8})\d{6}")


def _bars_and_digits(glyphs, min_digits: int = 3):
    """把一行连通块分成 数字 和 横杠。返回 (数字中位宽, 数字中位高, 最左横杠) 或 None。

    ★ `min_digits` 对日期行要调大。实测教训: 金额下面还有"交易成功"这类行,
      它们也能凑出 3 个块 + 1 根横杠, 于是扫描**在错的行上就停了**
      —— `03.jpg` 因此把 w=23 的块当成日期连字符(真正的日期行在更下面, 连字符 w=11)。
      日期行形如 `2026-07-23 05:06:24`, **有 12 个以上的数字**, 用这个把它认出来。
    """
    if len(glyphs) < 4:
        return None
    hs = sorted(g[3] for g in glyphs)
    med_h = hs[len(hs) // 2]
    if med_h < 6:
        return None
    digits = [g for g in glyphs if g[3] > 0.75 * med_h and g[2] < 1.5 * g[3]]
    bars = [g for g in glyphs if g[2] >= 1.5 * g[3] and g[3] <= 0.45 * med_h]
    if len(digits) < min_digits or not bars:
        return None
    dw = float(np.median([g[2] for g in digits]))
    dh = float(np.median([g[3] for g in digits]))
    if dw <= 0 or dh <= 0:
        return None
    return dw, dh, sorted(bars, key=lambda g: g[0])[0]


def measure(path: Path) -> dict | None:
    """量同一张图里 金额负号 与 日期连字符 各自相对本行数字的宽度比。"""
    try:
        rgb = np.asarray(Image.open(path).convert("RGB"))
    except Exception:
        return None
    loc, page = locate_amount_auto(rgb)
    if not loc or page != "white":
        return None
    x0, y0, x1, y1, _ = loc

    # ---- 金额行 ----
    ag = _reextract(rgb, (x0, y0, x1, y1), page)
    a = _bars_and_digits(ag)
    if not a:
        return None
    a_dw, a_dh, a_bar = a

    # ---- 日期行: 金额下方的第一条含横杠的文字行 ----
    # 日期形如 2026-08-03 12:34:56, 一行里有两个连字符。
    # 往下找几倍行高, 按行扫过去, 找到第一条"有数字也有横杠"的行就用它。
    H, W = rgb.shape[:2]
    row_h = max(8, y1 - y0)
    best = None
    y = y1 + int(row_h * 0.2)
    while y < min(H - 8, y1 + row_h * 6) and best is None:
        band_h = int(row_h * 0.9)
        sub_box = (0, y, W, min(H, y + band_h))
        dg = _reextract(rgb, sub_box, page)
        # 日期行的字比金额小得多; 太大的说明还在金额行里, 跳过
        if dg:
            hs = sorted(g[3] for g in dg)
            mh = hs[len(hs) // 2]
            if mh < a_dh * 0.75:
                # ★ 日期行至少 8 个数字 —— 否则会停在"交易成功"那种短行上, 量到错的横杠
                d = _bars_and_digits(dg, min_digits=8)
                if d:
                    best = d
        y += max(6, int(row_h * 0.35))
    if not best:
        return None
    d_dw, d_dh, d_bar = best

    a_ratio = a_bar[2] / a_dw
    d_ratio = d_bar[2] / d_dw
    ts = _TS.search(path.name)
    return {
        "image_name": path.name,
        "date": ts.group(1) if ts else "",
        "金额数字高": round(a_dh, 1), "金额数字宽": round(a_dw, 1),
        "金额负号宽比": round(a_ratio, 4),
        "日期数字高": round(d_dh, 1), "日期数字宽": round(d_dw, 1),
        "日期横杠宽比": round(d_ratio, 4),
        "两者之比": round(a_ratio / d_ratio, 4) if d_ratio else "",
    }


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="图内比对: 金额负号 vs 日期连字符")
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--since", type=str, default=None, help="YYYYMMDD")
    ap.add_argument("--until", type=str, default=None, help="YYYYMMDD")
    ap.add_argument("--names", type=Path, default=None,
                    help="只测这份名单里的图(每行一个文件名; 也吃 minus_outlier 的 csv)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=20260901)
    args = ap.parse_args()

    want = None
    if args.names:
        txt = args.names.read_text(encoding="utf-8-sig")
        if args.names.suffix.lower() == ".csv":
            want = {r["image_name"] for r in csv.DictReader(txt.splitlines())}
        else:
            want = {os.path.basename(x.split("\t")[0].strip())
                    for x in txt.splitlines() if x.strip()}
        print(f"只测名单里的 {len(want):,} 张")

    files = []
    for dp, _, fns in os.walk(args.input):
        for fn in fns:
            if os.path.splitext(fn)[1].lower() not in _EXTS:
                continue
            if want is not None and fn not in want:
                continue
            if args.since or args.until:
                m = _TS.search(fn)
                if not m:
                    continue
                d = m.group(1)
                if (args.since and d < args.since) or (args.until and d > args.until):
                    continue
            files.append(Path(dp) / fn)
    print(f"候选 {len(files):,} 张", flush=True)
    if not files:
        raise SystemExit("!! 没有候选")
    if want is None and args.n and args.n < len(files):
        random.Random(args.seed).shuffle(files)
        files = files[: args.n]

    rows = []
    for i, p in enumerate(files, 1):
        r = measure(p)
        if r:
            rows.append(r)
        if i % 500 == 0:
            print(f"  {i:,}/{len(files):,}  两行都量到 {len(rows):,}", flush=True)

    n = len(rows)
    print(f"\n看了 {len(files):,} 张, **金额行和日期行都量到的 {n:,} 张** "
          f"({100.0*n/len(files):.1f}%)")
    if n < 20:
        raise SystemExit("!! 样本太少, 结论不可靠")

    ar = np.array([r["金额负号宽比"] for r in rows])
    dr = np.array([r["日期横杠宽比"] for r in rows])
    rr = np.array([r["两者之比"] for r in rows if r["两者之比"] != ""], dtype=float)
    print(f"\n{'':<16}{'中位':>9}{'p5':>9}{'p95':>9}{'变异系数':>10}")
    for nm, v in (("金额负号宽比", ar), ("日期横杠宽比", dr), ("★ 两者之比", rr)):
        cv = np.std(v) / abs(np.mean(v)) * 100 if np.mean(v) else float("nan")
        print(f"{nm:<16}{np.median(v):>9.4f}{np.percentile(v,5):>9.4f}"
              f"{np.percentile(v,95):>9.4f}{cv:>9.1f}%")

    print("\n怎么读:")
    print("  **两者之比**在真图上应当是个**常数**(不一定等于 1 —— 金额的减号和日期的连字符")
    print("  很可能不是同一个码位, 字体设计上宽度本来就不同)。**要看的是它稳不稳。**")
    print("  · 某张图**金额负号宽比很高、但两者之比正常** -> 两个字形一起长")
    print("    -> **是渲染变体, 不是篡改**")
    print("  · 某张图**两者之比明显偏高** -> 只有金额的负号长, 日期的没长")
    print("    -> **同一张图内部不一致 -> 篡改**")

    if args.out and rows:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n明细 -> {args.out}")


if __name__ == "__main__":
    main()
