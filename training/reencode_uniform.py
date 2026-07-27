r"""把整个 SSP 数据集统一重存成同一 JPEG 质量,消除"压缩泄漏"。

关键坑:若 nature 还是 PNG/高质量、ai 是合成后存的 JPEG,SSP 会学成"JPEG=假"而不是学真伪。
所以训练前把 train/val 下 nature 和 ai 的每张图都用同一 q 重存一遍(默认 90),两类压缩一致。

用法:
  python training/reencode_uniform.py --root D:\ssp_alipay\imagenet_ai_0419_sdv4 --q 90
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageOps

_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="统一 JPEG 质量重存(防压缩泄漏)")
    ap.add_argument("--root", type=Path, required=True, help="数据集根,内含 train/val 的 nature/ai")
    ap.add_argument("--q", type=int, default=90)
    ap.add_argument("--max-side", type=int, default=768,
                    help="统一最长边像素(两类都缩到它,消除'低分辨率/模糊=假'的泄漏;0=不改尺寸)")
    args = ap.parse_args()

    n = 0
    for sp in ("train", "val"):
        for cl in ("nature", "ai"):
            d = args.root / sp / cl
            if not d.exists():
                print(f"缺目录: {d}")
                continue
            for p in list(d.iterdir()):
                if p.suffix.lower() not in _EXTS:
                    continue
                try:
                    with Image.open(p) as im:
                        rgb = ImageOps.exif_transpose(im).convert("RGB")
                    if args.max_side > 0:                         # 统一最长边 -> 两类分辨率/清晰度对齐
                        w, h = rgb.size
                        s = args.max_side / max(w, h)
                        if abs(s - 1.0) > 0.01:
                            rgb = rgb.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
                    dst = p.with_suffix(".jpg")
                    rgb.save(dst, "JPEG", quality=args.q)
                    if p.suffix.lower() != ".jpg":
                        p.unlink(missing_ok=True)   # 删掉原 png,只留统一 jpg
                    n += 1
                except Exception:
                    continue
            print(f"  {sp}/{cl}: {sum(1 for _ in d.iterdir())} 张")
    print(f"统一重存完成,共 {n} 张,全部 JPEG q={args.q}(两类压缩一致,无泄漏)。")


if __name__ == "__main__":
    main()
