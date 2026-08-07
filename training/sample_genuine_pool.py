r"""从图库里抽一大批"干净真截图", 用来做大样本误杀标定(自动拒阈值要靠它定)。

为什么要单独抽: 现有的真图验证集只有 2700 张 -> 最细只能分辨到约 0.04% 的误杀,
而自动拒需要**万分之一**量级的把握, 必须用 2-3 万张才量得出来。

过滤规则与训练时一致(排除相机照片/翻拍, 只留手机截图比例的干净图), 保证口径可比。

用法:
  python training/sample_genuine_pool.py --roots D:\download\TempFakeImages D:\download2\TempFakeImages \
      --out D:\probe\genuine_20k --n 20000 --seed 4242
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

from PIL import Image

_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_OPT = (33434, 33437, 34855, 37386)   # 光学拍摄参数; 有=相机照片, 不当真截图


def _is_clean_screenshot(im: Image.Image) -> bool:
    w, h = im.size
    short = min(w, h)
    if not short:
        return False
    aspect = max(w, h) / short
    try:
        ifd = im.getexif().get_ifd(0x8769)
    except Exception:
        ifd = {}
    return (not any(t in ifd for t in _OPT)) and short <= 1500 and 1.6 <= aspect <= 2.6


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="抽大批干净真截图做误杀标定")
    ap.add_argument("--roots", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=4242,
                    help="用**和训练不同**的 seed, 尽量避开训练见过的那批真图")
    args = ap.parse_args()

    files: list[Path] = []
    for r in args.roots:
        files += [p for p in r.rglob("*") if p.suffix.lower() in _EXTS]
    print(f"图库共 {len(files)} 个文件, 打乱后逐个筛...", flush=True)
    random.Random(args.seed).shuffle(files)

    args.out.mkdir(parents=True, exist_ok=True)
    kept = skipped = 0
    for p in files:
        if kept >= args.n:
            break
        try:
            with Image.open(p) as im:
                if not _is_clean_screenshot(im):
                    skipped += 1
                    continue
            shutil.copy2(p, args.out / p.name)
            kept += 1
            if kept % 2000 == 0:
                print(f"  已抽 {kept}...", flush=True)
        except Exception:
            skipped += 1

    print(f"完成: 抽了 {kept} 张 -> {args.out}(跳过 {skipped} 张: 相机照片/比例不符/读不了)")
    if kept < args.n:
        print(f"注意: 只抽到 {kept} 张(不足 {args.n})—— 图库里符合条件的就这么多。")
    print("下一步: 用 v7 打分, 再跑 autoreject_threshold.py 定自动拒的分数线。")


if __name__ == "__main__":
    main()
