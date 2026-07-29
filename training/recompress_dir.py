r"""把一个真图目录做"微信转发式"重压缩(低质量 JPEG, 可双压 + 可缩), 造硬负样本测误杀。

为什么: AI 检测器靠"文字被改软/糊"抓假图; 而**微信转发/重存的真图文字也会被压糊** ——
很可能被误判成假(误杀), 这是上线最大风险 + 判"模型学的是真指纹还是只看文字清晰度"。
所以要单独在这种硬真图上量误杀。

用法:
  python training/recompress_dir.py --src D:\ssp_test\hard_genuine --out D:\ssp_test\hard_genuine_wechat --q 65 --double
  # 想再模拟缩图: 加 --max-side 1080
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

from PIL import Image, ImageOps

_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _recompress(im: Image.Image, q: int) -> Image.Image:
    b = io.BytesIO()
    im.save(b, "JPEG", quality=q)
    b.seek(0)
    return Image.open(b).convert("RGB")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="真图微信式重压缩(硬负样本)")
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--q", type=int, default=65, help="JPEG 质量(微信偏低, 60-75)")
    ap.add_argument("--double", action="store_true", help="双次压缩(模拟多次转发)")
    ap.add_argument("--max-side", type=int, default=0, help="缩到最长边(0=不缩;微信约1080)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    files = [p for p in args.src.iterdir() if p.suffix.lower() in _EXTS]
    print(f"源 {len(files)} 张 -> 重压缩 q={args.q} double={args.double} max_side={args.max_side}", flush=True)
    n = 0
    for i, p in enumerate(files, 1):
        try:
            im = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
            if args.max_side > 0:
                w, h = im.size
                s = args.max_side / max(w, h)
                if s < 1.0:
                    im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
            im = _recompress(im, args.q)
            if args.double:
                im = _recompress(im, args.q)
            im.save(args.out / (p.stem + ".jpg"), "JPEG", quality=args.q)
            n += 1
            if i % 500 == 0:
                print(f"  {i}/{len(files)}...", flush=True)
        except Exception:
            continue
    print(f"完成: 重压缩 {n} 张 -> {args.out}")


if __name__ == "__main__":
    main()
