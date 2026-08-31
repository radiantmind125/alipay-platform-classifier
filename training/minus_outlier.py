r"""挑出**负号被拉长/加粗**的账单详情图 —— 不用模型, 不用假图样本。

背景
----
经理 2026-07-24 给过两张图并说明:

  "上面这张图是假的 **肉眼能分辨的就只有金额前面的负号**"
  "**长一些**···"
  "OCR 按照精度0.9也会识别成 同样的 **这里要做特殊处理**··"
  "但是现在**样本特别少**"

这个脚本就是那个"特殊处理", 并且**正好绕开"样本特别少"**:
它**不学假图**, 只从真图算出"正常负号长什么样", 再把离群的挑出来。
真实假图占比极低(实测元数据口径 6.5/万), 所以**分布本身几乎全是真图**, 可以自标定。

判据(2026-08-31 在 6,000 张白单上实测得出)
-----------------------------------------
量两个**与字号无关**的比值(实测对 `digit_h` 的相关系数只有 -0.08 / +0.03):

  `bar_width` = 负号宽 / 数字中位宽
  `bar_thick` = 负号厚 / 数字中位高

★ **必须按金额位数分组**: `bar_width` 对位数的相关系数是 **+0.522** ——
  中位数字宽取决于出现了哪些数字("1" 比 "0" 窄), 位数不同基线就不同。
  实测中位: 4 位 **0.674** / 5 位 **0.702** / 6 位 **0.702**。全局一刀切会在 4 位金额上误伤。

经理那两张图的实测(与他肉眼结论一致):
  `03.jpg` 宽 **1.0000** 厚 **0.1489** -> 4,204 张同位数真图里只有 2 张更宽  <- **他说的假图**
  `04.jpg` 宽 **0.7097** 厚 **0.1321** -> 第 **44.6** 百分位, 完全正常

覆盖边界(必须写清, 别让人以为管得更宽)
-------------------------------------
- **只管白底账单详情页。蓝底转账页金额不带符号, 这条线对蓝图覆盖率是 0。**
- 只管"负号被拉长/加粗"这一种手法。**改数字、复制粘贴数字, 这条一律看不见。**
- 判的是"这个负号和同一张图里的数字比例不对", **不是**"这张图是 AI 生成的"。

用法
----
  python training/minus_outlier.py --input D:\download\white --n 20000 ^
      --out D:\probe\minus_flag.csv --sheet D:\probe\minus_flag.png
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from locate_blue import locate_amount_auto      # noqa: E402
from glyph_baseline import _reextract           # noqa: E402  复用: 行内重取块, 不设高度下限

_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def measure(path: Path) -> dict | None:
    """量一张图的金额行。返回 None = 定位不到 / 不像金额行 / 没有负号。"""
    try:
        rgb = np.asarray(Image.open(path).convert("RGB"))
    except Exception:
        return None
    loc, page = locate_amount_auto(rgb)
    if not loc:
        return None
    x0, y0, x1, y1, _ = loc
    glyphs = _reextract(rgb, (x0, y0, x1, y1), page)
    if len(glyphs) < 4:
        return None

    hs = sorted(int(g[3]) for g in glyphs)
    med_h = hs[len(hs) // 2]
    if med_h < 20:
        return None
    digits = [g for g in glyphs if g[3] > 0.75 * med_h and g[2] < 1.5 * g[3]]
    bars = [g for g in glyphs if g[2] >= 1.5 * g[3] and g[3] <= 0.45 * med_h]
    if len(digits) < 4 or not bars:
        return None

    dh = np.array([g[3] for g in digits], dtype=float)
    dw = np.array([g[2] for g in digits], dtype=float)

    # ★★ 有效性闸 —— 没有这道闸, **一整个二维码**会混进来并且排在离群榜第一
    #    (实测: bw=1.3964 那张就是二维码, 它照样满足"有 4 个近方块 + 一根横条")。
    #    真金额行的数字**高度几乎完全一致**, 二维码/图标行不会。
    if dh.std() / dh.mean() > 0.08:       # 数字高度变异 > 8% -> 不是一行数字
        return None
    if dw.std() / dw.mean() > 0.30:       # 宽度可以差一点("1" 比 "0" 窄), 但差太多不行
        return None
    ar = float(np.median(dw) / np.median(dh))
    if not (0.45 <= ar <= 0.75):          # 数字宽高比, 真图实测中位 0.59
        return None

    b = sorted(bars, key=lambda g: g[0])[0]        # 最靠左那条 = 负号
    mw, mh = float(np.median(dw)), float(np.median(dh))
    return {
        "image_name": path.name,
        "n_digit": len(digits),
        "digit_h": round(mh, 1),
        "digit_ar": round(ar, 4),
        "bar_width": round(b[2] / mw, 4),
        "bar_thick": round(b[3] / mh, 4),
    }


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="挑出负号被拉长/加粗的账单详情图(自标定, 不需要假图)")
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--n", type=int, default=0, help="随机抽多少张; 0 = 全扫")
    ap.add_argument("--thr", type=float, default=0.90,
                    help="宽比绝对阈值, 默认 0.90 —— **落在实测的分布空隙里**, 见下")
    ap.add_argument("--pct", type=float, default=99.5,
                    help="仅用于打印各位数的分位参考, **不再用来判定**(见下)")
    ap.add_argument("--min-group", type=int, default=200,
                    help="某个位数不够这么多张就不给它单独定阈值(样本太少定不准)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--sheet", type=Path, default=None, help="把报出来的金额区拼成一张图, 好人工看")
    ap.add_argument("--seed", type=int, default=20260831)
    args = ap.parse_args()

    files = []
    for dp, _, fns in os.walk(args.input):
        for fn in fns:
            if os.path.splitext(fn)[1].lower() in _EXTS:
                files.append(Path(dp) / fn)
    print(f"候选 {len(files):,} 张", flush=True)
    if args.n and args.n < len(files):
        random.Random(args.seed).shuffle(files)
        files = files[: args.n]

    rows = []
    for i, p in enumerate(files, 1):
        r = measure(p)
        if r:
            rows.append(r)
        if i % 2000 == 0:
            print(f"  {i:,}/{len(files):,}  可量 {len(rows):,}", flush=True)

    if not rows:
        raise SystemExit("!! 一张都没量到")
    print(f"\n看了 {len(files):,} 张, **可量的 {len(rows):,} 张** "
          f"({100.0*len(rows)/len(files):.1f}%; 其余是定位不到/不是白单/金额没有负号)")

    # ---- 自标定: 按位数分组定阈值 ----
    # 假图占比极低, 所以分布本身几乎全是真图, 可以拿它当本底。
    by_nd: dict[int, list[dict]] = {}
    for r in rows:
        by_nd.setdefault(r["n_digit"], []).append(r)

    thr: dict[int, tuple[float, float]] = {}
    print(f"\n按金额位数自标定(分位 {args.pct}):")
    for nd in sorted(by_nd):
        sub = by_nd[nd]
        if len(sub) < args.min_group:
            print(f"  {nd} 位: n={len(sub):<6} 样本太少, 跳过不报")
            continue
        w = np.array([r["bar_width"] for r in sub])
        t = np.array([r["bar_thick"] for r in sub])
        thr[nd] = (float(np.percentile(w, args.pct)), float(np.percentile(t, args.pct)))
        print(f"  {nd} 位: n={len(sub):<6} 宽比 中位 {np.median(w):.4f} 阈值 {thr[nd][0]:.4f}"
              f" | 厚比 中位 {np.median(t):.4f} 阈值 {thr[nd][1]:.4f}")

    # ---- 判定: **只看宽比, 用绝对阈值** ----
    # ★★ 2026-08-31 实测(20,000 张白单, 可量 17,515)得出的两条修正:
    #
    # 1. **`bar_thick` 不能用, 它量的是字重不是篡改。**
    #    用"负号厚 / 数字笔画宽"(距离变换测笔画)重新归一化之后:
    #    普通图中位 **1.178**(p5~p95 = 1.076~1.464), 而 `03.jpg` 是 **1.278** —— **完全落在正常范围内**。
    #    也就是说 03 的负号"厚", 只是因为它那版字重偏粗(笔画/字高 0.1165 对 0.1032)。
    #    **厚度这一项已从判定里去掉。**
    #
    # 2. **不能用分位数当阈值。** 这些比值是小整数相除, **高度离散**,
    #    p99.5 恰好落在一个**很多张图共享的取值**上(5 位是 0.7381), `>=` 一刀切进去
    #    直接报出 **1,146 张(6.54%)** —— 真实篡改绝不可能这么多。
    #
    # ★ 真正的结构: **分布在 0.80~0.90 之间是空的**。
    #   实测 >=0.80 / >=0.85 / >=0.90 都是**同样的 14 张**, 到 >=0.95 才降到 10 张。
    #   主体全在 <=0.76, 中间一段真空, 然后才是这 14 张离群。
    #   -> **0.90 这个阈值落在天然空隙里**, 报出率 **8.0/万**, 复核量可以接受。
    #   (`03.jpg` = 1.0000 报得出; `04.jpg` = 0.7097 报不出 —— 与经理 7/24 的肉眼结论一致。)
    flagged = []
    for r in rows:
        if r["bar_width"] >= args.thr:
            r = dict(r)
            ref = thr.get(r["n_digit"])
            r["本组中位"] = round(float(np.median([x["bar_width"] for x in by_nd[r["n_digit"]]])), 4)
            r["超本组分位"] = "1" if (ref and r["bar_width"] >= ref[0]) else "0"
            flagged.append(r)
    flagged.sort(key=lambda r: -r["bar_width"])

    print(f"\n宽比 >= {args.thr} 报出 **{len(flagged)}** 张 "
          f"({100.0*len(flagged)/len(rows):.3f}%, **{10000.0*len(flagged)/len(rows):.1f}/万**)")
    print("  (厚度那一项已弃用 —— 实测它量的是字重, 不是篡改; 详见代码注释)")
    for r in flagged[:20]:
        print(f"    宽 {r['bar_width']:.4f}  ({r['n_digit']} 位, 本组中位 {r['本组中位']:.4f})  "
              f"{r['image_name'][:48]}")

    if args.out and flagged:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8-sig", newline="") as f:
            w_ = csv.DictWriter(f, fieldnames=list(flagged[0].keys()))
            w_.writeheader()
            w_.writerows(flagged)
        print(f"\n明细 -> {args.out}")

    if args.sheet and flagged:
        tiles = []
        for r in flagged[:24]:
            p = None
            for dp, _, fns in os.walk(args.input):
                if r["image_name"] in fns:
                    p = Path(dp) / r["image_name"]
                    break
            if not p:
                continue
            try:
                rgb = np.asarray(Image.open(p).convert("RGB"))
                loc, _pg = locate_amount_auto(rgb)
                if not loc:
                    continue
                x0, y0, x1, y1, _ = loc
                px = int((x1 - x0) * 0.40); py = int((y1 - y0) * 0.30)
                c = Image.fromarray(rgb[max(0, y0 - py):y1 + py, max(0, x0 - px):x1 + px])
                c = c.resize((int(c.width * 90 / c.height), 90), Image.LANCZOS)
                tiles.append(c)
            except Exception:
                continue
        if tiles:
            W = max(t.width for t in tiles); H = sum(t.height + 8 for t in tiles)
            sh = Image.new("RGB", (W, H), (255, 255, 255)); y = 0
            for im in tiles:
                sh.paste(im, (0, y)); y += im.height + 8
            args.sheet.parent.mkdir(parents=True, exist_ok=True)
            sh.save(args.sheet)
            print(f"拼图 -> {args.sheet}  (**一定要人工看一遍再下结论**)")


if __name__ == "__main__":
    main()
