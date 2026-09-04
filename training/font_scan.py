r"""金额字号异常扫描 —— 不用模型, 只量金额数字有多高。

背景
----
经理 2026-09-03: "可以在这个思路上做扩展 **比如金额字体过大字体异常**什么的"。
这个脚本做"字体过大"那一半。

金额数字高在**同一分辨率下几乎是个定值**。本机 12,000 张实测, 每个分辨率的高度
集中在两个相邻整数上(1179x2556 是 72 或 71, 合计占 98%), 组内变异系数 0.8%~4.2%。
所以"金额被重打成更大的字号"是能量出来的。

★★ 自标定, 不写死常态表
----------------------
和 `chevron_scan.py` 同一个思路: **不写死每个分辨率的常态高**,
而是**从你给的这批图里自己统计**。原因是本机那批图**全部是 2026-07 的**,
拿七月的常态去判九月的图, 量到的是 app 改版, 不是造假。
在服务器上跑, 常态就来自服务器上的真实数据, 这才是对的。

★ 判据是**两个条件同时满足**, 缺一不可
-------------------------------------
1. `金额高 / 该分辨率常态高` 偏大
2. `金额高 / 同一张图里的正文高` 也偏大

为什么必须要第二条: **开了系统大字体的用户, 整页字号都放大**, 金额高会到常态的 1.35 倍,
但这是完全正常的用户。实测四张这样的真图, 金额/正文分别是 2.425 / 2.459 / 2.432 / 2.282,
正落在真图中位 2.414 上 —— **整页一起放大, 金额本身没被动。**
只用第一条的话, 这个规则会**系统性地误伤需要大字体的人**。这不是误报率的问题。

★ 必须有负号才判
---------------
常态表是在**带负号的付款页**上标定的。**收款页是另一套排版, 金额字号大 1.37 倍**
(1170x2532 上付款页 71 像素, 收款页 97~109 像素)。拿付款页的常态去套收款页,
本机实测误报会从 20.7/万 涨到 **200.5/万**。所以没有负号一律跳过。

覆盖边界(必须一起报, 别让人以为管得更宽)
--------------------------------------
- **只管白底付款页。** 收款页、蓝底转账页都跳过, 覆盖率是 0。
- **只判偏大, 不判偏小。** 常态往下 1~2 像素就是另一个正常取值, 低侧没有余量。
  本机实测: 经理给的三张假图金额高 71, 常态 72, 是 **0.986 倍**, 方向相反, 这条抓不到它们。
- **抓的是"金额被重打成更大字号"这一种手法。** 我们手上**一个这样的样本都没有** ——
  经理说他见过, 但没给过图。所以这条的检出率是**推算的, 不是实测的**。
- 本机实测放大 10% 能查出 99% 以上, 放大 5% 要把阈值收到 1.03 才有 88%,
  **放大 3% 及以下查不出来**(常态本身就有 1 个像素的抖动)。
- 判的是"这张图的金额和同分辨率的其它图不一样", **不是**"这张图一定是假的"。

用法
----
  # 先小样本试跑, 看看定位率和常态表对不对
  python training/font_scan.py D:\download2\OtherImages --limit 5000 --out D:\probe\font.csv

  # 全量
  python training/font_scan.py D:\download2\OtherImages --out D:\probe\font.csv

  # 只看某个时期(常态和被判的图必须同期!)
  python training/font_scan.py D:\download2\OtherImages --since 20260801 --out D:\probe\font9.csv

  # 把标定出来的常态表打成 C# 字典, 贴进 FontCheck.cs
  python training/font_scan.py D:\download2\OtherImages --emit-table

★ 输出里有一栏**按月的常态高**。各月都一样 = 字号没变过, 可以混着判;
  某个月不一样 = app 改过版, **那就要用 --since/--until 分月各判各的**,
  而且 FontCheck.cs 里那张表必须重新生成。

★ 输出的是**一张阈值对照表**, 不是一个判定。选哪个操作点是经理定的, 不是脚本定的。
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_TS = re.compile(r"_(\d{8})\d{6}")   # 文件名里的时间戳, 形如 _20260902211530


def _label(bw: np.ndarray, min_area: int):
    """8 连通块。口径与 FontCheck.cs 的 Label 一致: 面积过滤 + 跳过背景标签 0。"""
    n, _, st, _ = cv2.connectedComponentsWithStats(bw, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a >= min_area:
            out.append((int(x), int(y), int(w), int(h), int(a)))
    return out


def _locate_amount(gray: np.ndarray, W: int, H: int):
    """金额行定位。与 MinusCheck.cs / FontCheck.cs 的 LocateAmount 同口径。"""
    y0b, y1b = int(H * 0.08), int(H * 0.55)
    if y1b - y0b < 8:
        return None
    _, dark = cv2.threshold(gray[y0b:y1b], 139, 255, cv2.THRESH_BINARY_INV)
    comps = []
    for (x, y, w, h, a) in _label(dark, 20):
        if h < 0.02 * H or h > 0.22 * H or w > 0.5 * W:
            continue
        comps.append((x, y + y0b, w, h, a))
    if not comps or len(comps) > 20000:
        return None
    rows, bounds = [], []
    for c in sorted(comps, key=lambda k: (k[1], k[0])):
        x, y, w, h, _a = c
        placed = False
        for i in range(len(rows)):
            ry0, ry1 = bounds[i]
            if min(y + h, ry1) - max(y, ry0) >= 0.5 * min(h, ry1 - ry0):
                rows[i].append(c)
                bounds[i] = (min(ry0, y), max(ry1, y + h))
                placed = True
                break
        if not placed:
            rows.append([c])
            bounds.append((y, y + h))
    best, best_med = None, -1.0
    for r in rows:
        if len(r) < 2:
            continue
        if max(k[0] + k[2] for k in r) - min(k[0] for k in r) < 0.1 * W:
            continue
        med = sorted(k[3] for k in r)[len(r) // 2]
        if med > best_med:
            best_med, best = med, r
    if best is None:
        return None
    return (min(k[0] for k in best), min(k[1] for k in best),
            max(k[0] + k[2] for k in best), max(k[1] + k[3] for k in best))


def _body_height(gray: np.ndarray, W: int, H: int, amount_bottom: int):
    """金额行下方正文文字的中位块高。块数不足返回 None, 由调用方弃权。"""
    y0 = amount_bottom + int(H * 0.01)
    y1 = min(H, int(H * 0.75))
    if y1 - y0 < 50:
        return None
    _, dark = cv2.threshold(gray[y0:y1], 139, 255, cv2.THRESH_BINARY_INV)
    hs = [k[3] for k in _label(dark, 20)
          if 0.008 * H < k[3] < 0.030 * H and k[2] < 0.4 * W]
    return float(np.median(hs)) if len(hs) >= 8 else None


def measure(path: str):
    """量一张图。返回 dict, 或 None(判不了)。

    整条流程与 FontCheck.cs 逐条对应, 改这里就要同步改那边, 否则两边结果对不上。
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    H, W = img.shape[:2]
    if W < 16 or H < 16:
        return None

    # 蓝底转账页排版不一样, 常态表不适用
    top = img[: max(1, H // 3)]
    b, g, r = (float(top[:, :, i].mean()) for i in range(3))
    if b > r + 25 and b > g + 15:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    box = _locate_amount(gray, W, H)
    if box is None:
        return None
    bx0, by0, bx1, by1 = box

    pad = max(2, int((by1 - by0) * 0.15))
    padx = max(pad, int((bx1 - bx0) * 0.30))
    cx0, cy0 = max(0, bx0 - padx), max(0, by0 - pad)
    cx1, cy1 = min(W, bx1 + pad), min(H, by1 + pad)
    if cx1 - cx0 < 8 or cy1 - cy0 < 8:
        return None

    sub = gray[cy0:cy1, cx0:cx1]
    _, fg = cv2.threshold(sub, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    if 255 * int((fg > 0).sum()) > 127 * fg.size:
        fg = 255 - fg

    glyphs = _label(fg, 6)
    if len(glyphs) < 4:
        return None
    hs = sorted(k[3] for k in glyphs)
    med_h = hs[len(hs) // 2]
    if med_h < 20:
        return None

    digits = [k for k in glyphs if k[3] > 0.75 * med_h and k[2] < 1.5 * k[3]]
    if len(digits) < 4:
        return None
    # 必须有负号 —— 见文件头"必须有负号才判"
    bars = [k for k in glyphs if k[2] >= 1.5 * k[3] and k[3] <= 0.45 * med_h]
    if not bars:
        return None

    dh = np.array([k[3] for k in digits], float)
    dw = np.array([k[2] for k in digits], float)
    if dh.std() / dh.mean() > 0.08:      # 闸 1: 数字高度不齐, 不像一行数字
        return None
    if dw.std() / dw.mean() > 0.30:      # 闸 2: 宽度差太多
        return None
    mw, mh = float(np.median(dw)), float(np.median(dh))
    if not (0.45 <= mw / mh <= 0.75):    # 闸 3: 宽高比不对
        return None

    body = _body_height(gray, W, H, by1)
    if body is None:
        return None

    bottoms = [k[1] + k[3] for k in digits]
    return dict(
        name=os.path.basename(path), W=W, H=H,
        amount_h=mh, body_h=body, amt_to_body=mh / body,
        n_digit=len(digits),
        baseline_spread=max(bottoms) - min(bottoms),
        height_cv=float(dh.std() / dh.mean()),
        month=(_TS.search(os.path.basename(path)).group(1)[:6]
               if _TS.search(os.path.basename(path)) else ""),
    )


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="金额字号异常扫描(自标定, 不写死常态)")
    ap.add_argument("input", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="每张图的测量值写到这个 CSV")
    ap.add_argument("--limit", type=int, default=0, help="只抽这么多张, 0 = 全扫")
    ap.add_argument("--min-group", type=int, default=300,
                    help="同分辨率少于这么多张就不判(标不出可信的常态)")
    ap.add_argument("--size-mult", type=float, default=1.03,
                    help="金额高超过常态的这个倍数算偏大")
    ap.add_argument("--ratio-mult", type=float, default=1.08,
                    help="金额/正文超过常态的这个倍数算偏大")
    ap.add_argument("--since", type=str, default=None, help="只看这个日期(含)之后的, YYYYMMDD")
    ap.add_argument("--until", type=str, default=None, help="只看这个日期(含)之前的, YYYYMMDD")
    ap.add_argument("--emit-table", action="store_true",
                    help="把标定出来的常态表打成 C# 字典, 贴进 FontCheck.cs")
    ap.add_argument("--sheet", type=Path, default=None,
                    help="把报出来的金额行拼成一张图, 好人工看")
    ap.add_argument("--replay", type=Path, default=None,
                    help="不重新扫图, 直接读之前 --out 存的 CSV 重做分析(秒级)")
    ap.add_argument("--seed", type=int, default=20260904)
    args = ap.parse_args()

    for _k, _v in (("--since", args.since), ("--until", args.until)):
        if _v is not None and not re.fullmatch(r"\d{8}", _v):
            ap.error(f"{_k} 要写成 YYYYMMDD, 收到的是 {_v!r}")

    if args.replay:
        with open(args.replay, encoding="utf-8-sig") as f:
            rows = [dict(r) for r in csv.DictReader(f)]
        for r in rows:
            for k in ("W", "H", "n_digit", "baseline_spread"):
                r[k] = int(float(r[k]))
            for k in ("amount_h", "body_h", "amt_to_body", "height_cv"):
                r[k] = float(r[k])
        print(f"从 {args.replay} 读回 {len(rows):,} 张的测量值, 不重新扫图")
        skipped = 0
        _analyse(args, rows, skipped, len(rows))
        return

    files = [p for p in args.input.rglob("*") if p.suffix.lower() in _EXTS]
    print(f"目录里 {len(files):,} 个文件")

    if args.since or args.until:
        kept, no_date = [], 0
        for p in files:
            m = _TS.search(p.name)
            if not m:
                no_date += 1
                continue
            d = m.group(1)
            if (args.since and d < args.since) or (args.until and d > args.until):
                continue
            kept.append(p)
        print(f"按日期筛({args.since or '不限'} ~ {args.until or '不限'}): "
              f"留下 {len(kept):,}, 取不到日期的 {no_date:,} 张已排除")
        files = kept

    if args.limit and len(files) > args.limit:
        import random
        random.Random(args.seed).shuffle(files)
        files = files[: args.limit]
        print(f"抽样 {len(files):,} 张")

    rows, skipped = [], 0
    for i, p in enumerate(files, 1):
        if i % 20000 == 0:
            print(f"  ...{i:,}/{len(files):,}, 量到 {len(rows):,}")
        try:
            r = measure(str(p))
        except Exception:
            r = None
        if r:
            rows.append(r)
        else:
            skipped += 1

    _analyse(args, rows, skipped, len(files))


def _analyse(args, rows, skipped, n_files) -> None:
    print(f"\n量到 {len(rows):,} 张, 判不了 {skipped:,} 张 "
          f"(判不了的占 {100.0 * skipped / max(1, n_files):.1f}%)")
    if not rows:
        print("一张都没量到, 检查目录对不对"); return

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"逐张测量值 -> {args.out}")

    # ---- 自标定: 每个分辨率的常态高 = 众数 ----
    by_res = defaultdict(list)
    for r in rows:
        by_res[(r["W"], r["H"])].append(r)
    groups = {k: v for k, v in by_res.items() if len(v) >= args.min_group}
    small = sum(len(v) for k, v in by_res.items() if len(v) < args.min_group)
    print(f"够 --min-group={args.min_group} 的分辨率 {len(groups)} 组, "
          f"覆盖 {sum(len(v) for v in groups.values()):,} 张; "
          f"不够的 {small:,} 张整组跳过(不判, 不是判为正常)")

    if not groups:
        print()
        print(f"没有任何分辨率够 {args.min_group} 张, 定不出常态, 到此为止。")
        print("试跑时可以把 --min-group 调小(比如 --min-group 50)先看看表的形状,")
        print("但正式标定不要这么干 —— 样本少定出来的常态不可信。")
        return

    norm = {k: Counter(int(round(x["amount_h"])) for x in v).most_common(1)[0][0]
            for k, v in groups.items()}
    ratio_all = np.array([r["amt_to_body"] for k in groups for r in groups[k]])
    norm_ratio = float(np.median(ratio_all))

    print(f"\n金额/正文 的真图常态(中位) = {norm_ratio:.3f}   "
          f"p1 {np.percentile(ratio_all, 1):.3f}  p99 {np.percentile(ratio_all, 99):.3f}")

    print(f"\n{'分辨率':>14} {'张数':>7} {'常态高':>6} {'众数占比':>8} {'高度分布(前3)':<22}")
    for k, v in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        c = Counter(int(round(x["amount_h"])) for x in v)
        dist = " ".join(f"{h}:{n}" for h, n in c.most_common(3))
        print(f"{k[0]}x{k[1]:<7} {len(v):>7,} {norm[k]:>6} "
              f"{100.0 * c.most_common(1)[0][1] / len(v):>7.1f}% {dist:<22}")

    # ---- 按月常态: app 改版的诊断 ----
    months = sorted({r["month"] for r in rows if r["month"]})
    if len(months) > 1:
        print(f"\n按月常态高(各月一样 = 字号没变过; 不一样 = app 改过版, 要分月各判各的)")
        top = [k for k, _ in sorted(groups.items(), key=lambda kv: -len(kv[1]))[:5]]
        print(f"{'分辨率':>14} " + " ".join(f"{m:>10}" for m in months))
        for k in top:
            cells = []
            for m in months:
                hs = [int(round(x["amount_h"])) for x in groups[k] if x["month"] == m]
                cells.append(f"{Counter(hs).most_common(1)[0][0]}({len(hs)})" if hs else "-")
            print(f"{k[0]}x{k[1]:<7} " + " ".join(f"{c:>10}" for c in cells))

    # ---- 阈值对照表: 给的是选项, 不是判定 ----
    print(f"\n阈值对照表 (两个条件都要满足才报; 报出的都是真图, 所以这一栏就是误报率)")
    print(f"{'字号倍数':>9} {'比值倍数':>9} {'报出':>13} {'万分之':>9}")
    for sm in (1.02, 1.03, 1.05, 1.08):
        for rm in (1.00, 1.05, 1.08, 1.12):
            hit = sum(1 for k in groups for r in groups[k]
                      if r["amount_h"] > sm * norm[k]
                      and r["amt_to_body"] > rm * norm_ratio)
            tot = sum(len(v) for v in groups.values())
            print(f"{sm:>9.2f} {rm:>9.2f} {hit:>6,}/{tot:<6,} {10000.0 * hit / tot:>8.2f}")

    hits = [r for k in groups for r in groups[k]
            if r["amount_h"] > args.size_mult * norm[(r['W'], r['H'])]
            and r["amt_to_body"] > args.ratio_mult * norm_ratio]
    tot = sum(len(v) for v in groups.values())
    print(f"\n当前参数 (--size-mult {args.size_mult} --ratio-mult {args.ratio_mult}): "
          f"报出 {len(hits)}/{tot:,} = {10000.0 * len(hits) / max(1, tot):.2f}/万")
    for r in sorted(hits, key=lambda x: -x["amount_h"] / norm[(x['W'], x['H'])])[:30]:
        n = norm[(r["W"], r["H"])]
        print(f"    {r['amount_h']:.0f} vs 常态 {n} ({r['amount_h'] / n:.3f}x)  "
              f"金额/正文 {r['amt_to_body']:.3f}  {r['name']}")

    if args.sheet and hits:
        _make_sheet(args.input, hits, norm, args.sheet)

    if args.emit_table:
        print("\n// 贴进 FontCheck.cs 的 NormalDigitHeight。"
              f"标定自 {args.input}, 日期 {args.since or '不限'}~{args.until or '不限'}, "
              f"共 {tot:,} 张")
        print("        static readonly Dictionary<(int, int), int> NormalDigitHeight = new()")
        print("        {")
        for k, v in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            print(f"            {{ ({k[0]}, {k[1]}), {norm[k]} }},   // n={len(v):,}")
        print("        };")
        print(f"        public const double NormalAmountToBody = {norm_ratio:.2f};")


def _make_sheet(input_dir, hits, norm, out_path):
    """把报出来的图各裁一条金额行, 上下叠成一张 PNG, 供人工看。

    标签只写 ASCII(cv2.putText 画不了中文): 实测高/常态高=倍数, r=金额比正文。
    """
    tiles = []
    for r in hits[:40]:
        hit = next(input_dir.rglob(r["name"]), None)
        if hit is None:
            continue
        img = cv2.imread(str(hit), cv2.IMREAD_COLOR)
        if img is None:
            continue
        H, W = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        box = _locate_amount(gray, W, H)
        if box is None:
            continue
        bx0, by0, bx1, by1 = box
        m = int((by1 - by0) * 0.6)
        crop = img[max(0, by0 - m):min(H, by1 + m), :]
        if crop.size == 0:
            continue
        scale = 900.0 / crop.shape[1]
        crop = cv2.resize(crop, (900, max(1, int(crop.shape[0] * scale))))
        bar = np.full((26, 900, 3), 240, np.uint8)
        n = norm[(r["W"], r["H"])]
        cv2.putText(bar, f'{r["amount_h"]:.0f}/{n}={r["amount_h"] / n:.3f}x  '
                         f'r={r["amt_to_body"]:.3f}  {r["name"][:52]}',
                    (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(np.vstack([bar, crop]))
    if not tiles:
        print("没有可拼的图(报出的文件在目录里找不到?)")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), np.vstack(tiles))
    print(f"拼图 -> {out_path}  (**一定要人工看一遍再下结论**)")


if __name__ == "__main__":
    main()
