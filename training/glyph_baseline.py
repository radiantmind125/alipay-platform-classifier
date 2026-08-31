r"""金额区字形几何的**真图本底** —— 字体这条线的第一道闸门。

为什么先量本底
--------------
经理 2026-08-31 派的活: "按字体分类的识别 / **一张图只能出现一种字体** / 研究下 /
主要是和手机系统相关的 估计有点麻烦"。冲的是"减号骗单", 他说**只有 6 个像素的差别**。

在写任何检测器之前, 必须先回答一个问题:
**真图自己在这些几何量上散不散?** 真图本身就散 -> 没有余量, 这条路当场死掉, 省下后面全部工夫。

★ 已经先在两张真图(`03.jpg` / `04.jpg`, 多半是经理最早给的那两张加负号的例子)上手量过, 结论:

  - **横杠厚度 / 数字高度**在**同一张图内部很稳**(两张分别差 6% 和 1.5%),
    而且是在**金额字号约为正文 2.3 倍**的前提下 -> **抗尺度, 最有希望的特征**
  - **宽度比和垂直偏移不能直接比**: 金额里的 `−` 和日期里的 `–` **很可能不是同一个码位**
    (减号 vs 连字符/en dash), 字体设计上宽度和垂直位置本来就不同;
    而且**细横杠的宽度对二值化阈值极敏感**(日期横杠只有 3 px 厚, 差一个像素就差约 10%)
  - **金额数字比正文数字系统性地更窄**(03: 0.574 vs 0.632; 04: 0.585 vs 0.652),
    **两台不同设备都复现** -> 这是**支付宝渲染本身的属性, 不是篡改痕迹**
    -> **"一张图只能出现一种字体"按字面说不成立**, 规则得改成**同字段内比**或**先标定掉这个固有差**

这个脚本把上面第三条推到几百上千张图上, 看那个"固有差"到底有多稳。

量什么
------
只量**金额那一行**(用线上同一个定位器 `locate_amount_auto`, 白图蓝图都支持, 实测 98.0%):

  - `digit_ar`   数字宽高比中位数  —— 字体形状, 与字号无关
  - `bar_thick`  横杠厚 / 数字高   —— 笔画权重, 与字号无关
  - `bar_width`  横杠宽 / 数字宽   —— **已知不稳**, 一并记下来好证明它不稳
  - `bar_dy`     横杠中心相对数字中心的偏移 / 数字高

**只读**: 不改任何源图, 只往 --out 写一份 CSV。

用法
----
  python training/glyph_baseline.py --input c:\projects\China\TempFakeImages --n 800 ^
      --out D:\probe\glyph_baseline.csv
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

from locate_blue import locate_amount_auto  # noqa: E402

_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _reextract(rgb: np.ndarray, box, page: str):
    r"""在**已定位的金额行内**重新取连通块 —— 关键是**不带高度下限**。

    ★ 为什么必须重取: 线上的 `locate_amount` 在 `engine_b_tamper.py:97` 有一道
      `hh < 0.02 * h` 的高度闸。对 `03.jpg`(高 1280)那就是 **25.6 px 的下限**,
      而负号只有 **7 px 高** —— **负号被这道闸整个滤掉了**, 小数点也一样。
      那道闸对它本来的用途(给线路B 找金额数字去切块)是对的,
      但**做负号取证恰恰相反**: 我们要的就是那个被它丢掉的东西。
      (实测: 588 张蓝图 + 667 张白图, 用原 glyphs 一根横杠都取不到。)
    """
    import cv2
    x0, y0, x1, y1 = box
    pad = max(2, int((y1 - y0) * 0.15))
    # ★ 左边要**大幅**外扩: `locate_amount` 的 x0 是**只用数字**算出来的
    #   (负号在 :97 那道高度闸就被丢了), 所以负号**落在 x0 左边**, 小的 pad 根本够不着。
    #   实测: 只留 15% 行高的 pad, 03/04 两张都取不到负号。
    padx = max(pad, int((x1 - x0) * 0.30))
    H, W = rgb.shape[:2]
    sub = rgb[max(0, y0 - pad):min(H, y1 + pad), max(0, x0 - padx):min(W, x1 + pad)]
    if sub.size == 0:
        return []
    g = cv2.cvtColor(sub, cv2.COLOR_RGB2GRAY)
    # 白图是浅底深字, 蓝图是蓝底白字 —— 极性相反, 用 Otsu 之后按前景占比自动纠正
    th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    if th.mean() > 127:                      # 前景过半 = 极性反了
        th = 255 - th
    n, _, st, cen = cv2.connectedComponentsWithStats(th, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < 6:                            # 只挡真正的噪点, **不设高度下限**
            continue
        out.append([int(x), int(y), int(w), int(h), int(a), float(cen[i][1])])
    return sorted(out, key=lambda c: c[0])


def measure(path: Path) -> dict | None:
    """返回金额行的几何量; 定位不到或块数不够就返回 None。"""
    try:
        rgb = np.asarray(Image.open(path).convert("RGB"))
    except Exception:
        return None
    loc, page = locate_amount_auto(rgb)
    if not loc:
        return None
    x0, y0, x1, y1, _orig = loc
    glyphs = _reextract(rgb, (x0, y0, x1, y1), page)
    if len(glyphs) < 3:
        return None

    hs = sorted(int(g[3]) for g in glyphs)
    med_h = hs[len(hs) // 2]
    if med_h < 8:                     # 太小, 量出来全是噪声
        return None

    # 数字 = 高度接近中位高 且 不是横条; 横杠 = 明显扁 且 矮
    digits = [g for g in glyphs if g[3] > 0.75 * med_h and g[2] < 1.5 * g[3]]
    bars = [g for g in glyphs if g[2] >= 1.5 * g[3] and g[3] <= 0.45 * med_h]
    if len(digits) < 2:
        return None

    dh = float(np.median([g[3] for g in digits]))
    dw = float(np.median([g[2] for g in digits]))
    dcy = float(np.median([g[1] + g[3] / 2.0 for g in digits]))

    row = {
        "image_name": path.name,
        "page_type": page,
        "n_glyph": len(glyphs),
        "n_digit": len(digits),
        "n_bar": len(bars),
        "digit_h": round(dh, 2),
        "digit_w": round(dw, 2),
        "digit_ar": round(dw / dh, 4) if dh else "",
        "bar_thick": "", "bar_width": "", "bar_dy": "",
    }
    if bars:
        # 有多条时取最靠左的 —— 金额前面那个负号
        b = sorted(bars, key=lambda g: g[0])[0]
        bcy = b[1] + b[3] / 2.0
        row["bar_thick"] = round(b[3] / dh, 4)
        row["bar_width"] = round(b[2] / dw, 4)
        row["bar_dy"] = round((bcy - dcy) / dh, 4)
    return row


def _pct(a, q):
    return round(float(np.percentile(a, q)), 4) if len(a) else float("nan")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="量真图金额区字形几何的本底分布")
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--n", type=int, default=800, help="随机抽多少张")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=20260831)
    args = ap.parse_args()

    files = []
    for dp, _, fns in os.walk(args.input):
        for fn in fns:
            if os.path.splitext(fn)[1].lower() in _EXTS:
                files.append(Path(dp) / fn)
    print(f"候选 {len(files):,} 张", flush=True)
    if not files:
        raise SystemExit("!! 一张图都没有")
    random.Random(args.seed).shuffle(files)
    files = files[: args.n]

    rows, no_loc = [], 0
    for i, p in enumerate(files, 1):
        r = measure(p)
        if r is None:
            no_loc += 1
        else:
            rows.append(r)
        if i % 200 == 0:
            print(f"  {i}/{len(files)}  定位成功 {len(rows)}", flush=True)

    n = len(rows)
    print(f"\n看了 {len(files)} 张, 定位成功 **{n}** 张 ({100.0*n/len(files):.1f}%), "
          f"定位不到 {no_loc} 张")
    if not n:
        raise SystemExit("!! 一张都没定位到")

    for tag in ("white", "blue"):
        sub = [r for r in rows if r["page_type"] == tag]
        if not sub:
            continue
        print(f"\n=== {tag} 图 n={len(sub)} ===")
        ar = np.array([r["digit_ar"] for r in sub if r["digit_ar"] != ""], dtype=float)
        print(f"  数字宽高比  中位 {np.median(ar):.4f}   "
              f"p5 {_pct(ar,5)}  p95 {_pct(ar,95)}   "
              f"变异系数 {np.std(ar)/np.mean(ar)*100:.1f}%")
        for key, name in (("bar_thick", "横杠厚/数字高"), ("bar_width", "横杠宽/数字宽"),
                          ("bar_dy", "横杠中心偏移")):
            v = np.array([r[key] for r in sub if r[key] != ""], dtype=float)
            if len(v) < 3:
                print(f"  {name}: 样本太少 (n={len(v)})")
                continue
            cv = np.std(v) / abs(np.mean(v)) * 100 if np.mean(v) else float("nan")
            print(f"  {name:<14} n={len(v):<4} 中位 {np.median(v):+.4f}  "
                  f"p5 {_pct(v,5):+.4f}  p95 {_pct(v,95):+.4f}  变异系数 {cv:.1f}%")

    print("\n怎么读: **变异系数越小 = 真图越集中 = 留给篡改的余量越大**。")
    print("      某个量在真图上就散得厉害, 那它**不能**用来判假 —— 先淘汰掉它, 别拿去训模型。")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n明细 -> {args.out}")


if __name__ == "__main__":
    main()
