r"""数一数每个目录里**蓝图和白图各占多少** —— 别再靠目录名猜版式。

为什么要有这个
--------------
我们一直用"目录名"当版式的依据(`download`=蓝图, `download2`=白图),
结果这个约定在后续操作里被悄悄破坏过, 也被我记错过一次, 直接导致
**所有"干净"的召回数其实都是跨版式测出来的**(白图上训练 -> 蓝图上测), 而当时没人意识到。

**目录名是约定, 像素不是。** 这个脚本直接看像素判版式, 只读, 随时可跑。

判定口径与 `predict_tiled` / `locate_blue` 完全一致:
取上三分之一的均值, `b > r+25 且 b > g+15` 即判蓝图。

用法
----
  python training/scan_pools.py --roots D:\download\TempFakeImages D:\download2\TempFakeImages
  python training/scan_pools.py --roots D:\probe\genuine_20k --sample 800
  python training/scan_pools.py --roots D:\download2 --recurse-one     # 逐个子目录分别统计
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def is_blue_page(path: Path) -> bool | None:
    """True=蓝图, False=白图, None=读不了。只解上三分之一, 快。"""
    try:
        im = Image.open(path)
        im.draft("RGB", (im.size[0] // 4, im.size[1] // 4))   # JPEG 可以按 1/4 解, 快很多
        a = np.asarray(im.convert("RGB"))
    except Exception:
        return None
    top = a[: max(1, a.shape[0] // 3)]
    r, g, b = (float(top[:, :, i].mean()) for i in range(3))
    return (b > r + 25) and (b > g + 15)


def scan(d: Path, n: int, seed: int) -> None:
    files = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in _EXT] if d.is_dir() else []
    total = len(files)
    if total == 0:
        print(f"  {str(d):58s} 没有图片")
        return
    random.seed(seed)
    samp = files if total <= n else random.sample(files, n)
    blue = white = err = 0
    for p in samp:
        v = is_blue_page(p)
        if v is None:
            err += 1
        elif v:
            blue += 1
        else:
            white += 1
    ok = blue + white
    if ok == 0:
        print(f"  {str(d):58s} 共 {total:7d} 张, 但抽样全部读失败")
        return
    kind = "**蓝图为主**" if blue > ok * 0.8 else ("**白图为主**" if white > ok * 0.8 else "**混合**")
    print(f"  {str(d):58s} 共 {total:7d} 张 | 抽 {ok:4d}: "
          f"蓝 {blue*100//ok:3d}%  白 {white*100//ok:3d}%  {kind}"
          + (f"  (读失败 {err})" if err else ""))


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="统计目录里蓝图/白图的比例(只读)")
    ap.add_argument("--roots", nargs="+", required=True, help="要统计的目录, 可多个")
    ap.add_argument("--sample", type=int, default=400, help="每个目录抽多少张判版式")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--recurse-one", action="store_true",
                    help="把每个 root 下的**一级子目录**也分别统计一遍")
    args = ap.parse_args()

    print("=" * 96)
    print("版式统计 (判定口径与 predict_tiled 一致: 上1/3均值 b>r+25 且 b>g+15 = 蓝图)")
    print("=" * 96)
    for r in args.roots:
        d = Path(r)
        if not d.exists():
            print(f"  {r:58s} !!! 目录不存在")
            continue
        scan(d, args.sample, args.seed)
        if args.recurse_one:
            for sub in sorted(x for x in d.iterdir() if x.is_dir()):
                scan(sub, args.sample, args.seed)
    print()
    print("怎么用: 造样本前先确认源目录是哪种版式;")
    print("        训练和测试要**同版式**才是同分布的召回, 跨版式测出来的是泛化能力, 两者不能混为一谈。")


if __name__ == "__main__":
    main()
