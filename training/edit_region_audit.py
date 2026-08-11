r"""假图到底改在哪儿? —— 把假图和它的源图逐张比对, 看改动区落在不落在金额框里(只读)。

为什么必须有这个
----------------
`gen_api_local_edit.py:38` **只 import 了白图定位器** `locate_amount`, 没有蓝图定位器,
也没有版式分支。而 `locate_amount` 的判据是 `gray < 140` 的**深字浅底**
(`engine_b_tamper.py` 里那段 `dark = (gray[y0b:y1b] < 140)`)。

支付宝蓝底转账页的蓝大约 gray=106, **整个背景都会被判成"深"**, 金额是白字反而成了洞。
所以在真正的蓝图上, 这个定位器要么返回 None(该图被 `:184` 跳过), 要么锁到页面里
唯一的深字浅底元素 —— **红包促销卡**。

于是有个必须回答的问题: **那些"蓝图"API 假图集, 改动到底做在金额上, 还是做在红包卡上?**
如果是后者, 那么:
  - 这些假图的"召回率"测的根本不是改金额检测;
  - 用同一条路径存下来的**蓝图训练裁块**也是错区域的裁块。

★ 这条不能靠看代码推断, 只能**逐张比像素**。目录名不可信、参数名不可信, 只有像素可信。

判读逻辑
--------
1. 改动面积占比 > `--full-thresh`(默认 0.5) -> **整图重绘**, 不是局部编辑, 另算一类。
2. 否则取改动区的外接框, 和两个定位器给出的金额框比:
   - 落在**同版式**该用的那个金额框里  -> 正常
   - 落在另一个定位器的框里            -> **改错了地方**
   - 两个都不沾                        -> 改在别处(红包卡/广告位/头像等)

用法
----
  python training/edit_region_audit.py --fake D:\probe\api_wan_test ^
      --src-root D:\download\TempFakeImages D:\download2\OtherImages --limit 80

**只读**: 只读图片和 manifest, 不写任何文件。
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from engine_b_tamper import locate_amount
from locate_blue import is_blue_page, locate_amount_blue

_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_rgb(p: Path) -> np.ndarray | None:
    try:
        return np.asarray(Image.open(p).convert("RGB"))
    except Exception:
        return None


def diff_box(src: np.ndarray, fake: np.ndarray) -> tuple[tuple[int, int, int, int] | None, float]:
    """返回 (改动区外接框, 改动像素占比)。假图尺寸不同就先缩回源图尺寸。"""
    h, w = src.shape[:2]
    if fake.shape[:2] != (h, w):
        fake = cv2.resize(fake, (w, h), interpolation=cv2.INTER_AREA)
    a = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY).astype(np.int16)
    b = cv2.cvtColor(fake, cv2.COLOR_RGB2GRAY).astype(np.int16)
    d = np.abs(a - b).astype(np.uint8)
    # 阈值取"中位数 + 一点余量"的稳健形式: API 出图整体会有轻微重编码噪声, 不能用固定小阈值
    thr = max(18, int(np.median(d)) + 12)
    m = (d >= thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    ratio = float(m.sum()) / float(h * w)
    if m.sum() == 0:
        return None, 0.0
    nlab, _, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if nlab <= 1:
        return None, ratio
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, ww, hh = (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                    int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]))
    return (x, y, x + ww, y + hh), ratio


def contain(inner: tuple[int, int, int, int], outer: tuple[int, int, int, int], pad: int = 24) -> float:
    """inner 有多大比例落在 outer(外扩 pad)里。"""
    ox0, oy0, ox1, oy1 = outer[0] - pad, outer[1] - pad, outer[2] + pad, outer[3] + pad
    ix0, iy0, ix1, iy1 = inner
    w = max(0, min(ix1, ox1) - max(ix0, ox0))
    h = max(0, min(iy1, oy1) - max(iy0, oy0))
    area = max(1, (ix1 - ix0) * (iy1 - iy0))
    return w * h / area


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="查假图的改动区是不是落在金额框里(只读)")
    ap.add_argument("--fake", type=Path, required=True, help="假图目录(里面要有 manifest.csv)")
    ap.add_argument("--src-root", type=Path, nargs="+", required=True, help="源图池, 可多个")
    ap.add_argument("--limit", type=int, default=80, help="最多比对几张")
    ap.add_argument("--full-thresh", type=float, default=0.5, help="改动面积超过这个比例判为整图重绘")
    ap.add_argument("--hit", type=float, default=0.5, help="改动框有这么多比例落在金额框里就算命中")
    args = ap.parse_args()

    mp = args.fake / "manifest.csv"
    if not mp.exists():
        raise SystemExit(f"没有 manifest.csv, 无法回溯源图: {mp}")
    rows = list(csv.DictReader(open(mp, encoding="utf-8-sig")))

    # 源图池建索引(basename -> 路径), 一次扫完
    index: dict[str, Path] = {}
    for root in args.src_root:
        if not root.is_dir():
            print(f"  源图池不存在, 跳过: {root}")
            continue
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in _EXT:
                index.setdefault(p.name, p)
    print("=" * 100)
    print(f"假图目录 {args.fake}  manifest {len(rows)} 行 | 源图池索引 {len(index)} 张")
    print("=" * 100)

    tally: Counter[str] = Counter()
    page_tally: Counter[str] = Counter()
    n = miss = 0
    for r in rows:
        if n >= args.limit:
            break
        fn = (r.get("file") or "").strip()
        sn = Path((r.get("src") or "").strip()).name
        if not fn or not sn:
            continue
        fp, sp = args.fake / fn, index.get(sn)
        if not fp.exists() or sp is None:
            miss += 1
            continue
        fake, src = load_rgb(fp), load_rgb(sp)
        if fake is None or src is None:
            miss += 1
            continue
        n += 1

        blue = is_blue_page(src)
        page_tally["蓝图" if blue else "白图"] += 1
        box, ratio = diff_box(src, fake)
        if box is None:
            tally["没测到改动"] += 1
            continue
        if ratio > args.full_thresh:
            tally["整图重绘"] += 1
            continue

        lw = locate_amount(src)
        lb = locate_amount_blue(src)
        cw = contain(box, (lw[0], lw[1], lw[2], lw[3])) if lw else 0.0
        cb = contain(box, (lb[0], lb[1], lb[2], lb[3])) if lb else 0.0
        right, wrong = (cb, cw) if blue else (cw, cb)
        if right >= args.hit:
            tally["改在金额区(对)"] += 1
        elif wrong >= args.hit:
            tally["**改在另一个定位器的框里(错区域)**"] += 1
        else:
            tally["**改在别处(红包卡/广告位等)**"] += 1

    print(f"实际比对 {n} 张 (源图找不到/读不了 {miss} 张)")
    print(f"版式: " + "  ".join(f"{k} {v}" for k, v in page_tally.items()))
    print()
    for k, v in tally.most_common():
        print(f"  {k:36s} {v:5d}  {v * 100.0 / max(1, n):5.1f}%")
    print()
    print("判读:")
    print("  '改在金额区(对)' 占绝大多数    -> 这批假图是合格的改金额样本, 召回数有意义。")
    print("  '改在另一个定位器的框里'        -> **造样本时用错了定位器**, 这批的召回数测的不是改金额检测。")
    print("  '改在别处'                      -> 多半锁到了红包促销卡, 同上, 数不能用。")
    print("  '整图重绘' 占多数               -> 这是整图生成样本(线路A 的活), 不该拿来测线路B 的局部改金额。")


if __name__ == "__main__":
    main()
