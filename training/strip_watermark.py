r"""去掉 AI 生成图的 AIGC 水印(豆包/千问等中国模型强制加的"AI生成"角标)。

为什么: 骗子交假图前会把水印裁掉; 若拿带水印的假图训, 模型会学"有水印=假"这种一裁就破的捷径,
所以喂检测器前必须去水印, 逼它学真的生成指纹(和 reencode 防"JPEG=假"、格式增广同一反捷径原则)。
测 v3 时也用去水印版, 更贴近真实欺诈、避免水印当混淆。

两种去法(看到样本水印在哪再调):
- mask-corner: 把某个角的一块区域 cv2 inpaint 抹掉(角落通常是背景, 抹了不伤版面)。中国AIGC角标多在右下/左下。
- crop-bottom: 直接裁掉底部一条(整条底部水印用)。

用法(先看一张样本水印在哪, 再选 mode/角/比例):
  python training/strip_watermark.py --src D:\dq_raw --out D:\dq_clean --mode mask-corner --corner br --corner-frac 0.16
  python training/strip_watermark.py --src D:\dq_raw --out D:\dq_clean --mode crop-bottom --bottom-frac 0.05
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _mask_corner(arr: np.ndarray, corner: str, frac: float) -> np.ndarray:
    h, w = arr.shape[:2]
    cw, ch = max(1, int(w * frac)), max(1, int(h * frac))
    m = np.zeros((h, w), np.uint8)
    if corner == "br":
        m[h - ch:h, w - cw:w] = 255
    elif corner == "bl":
        m[h - ch:h, 0:cw] = 255
    elif corner == "tr":
        m[0:ch, w - cw:w] = 255
    else:  # tl
        m[0:ch, 0:cw] = 255
    return cv2.inpaint(arr, m, 3, cv2.INPAINT_TELEA)


def _crop_bottom(arr: np.ndarray, frac: float) -> np.ndarray:
    h = arr.shape[0]
    return arr[0:max(1, int(h * (1.0 - frac))), :, :]


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="去 AIGC 水印(角落抹除或裁底)")
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mode", choices=["mask-corner", "crop-bottom", "both"], default="mask-corner")
    ap.add_argument("--corner", choices=["br", "bl", "tr", "tl"], default="br", help="角标位置(默认右下)")
    ap.add_argument("--corner-frac", type=float, default=0.16, help="角落抹除区域占比(宽高各)")
    ap.add_argument("--bottom-frac", type=float, default=0.05, help="裁掉底部比例")
    ap.add_argument("--q", type=int, default=95)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    files = [p for p in args.src.iterdir() if p.suffix.lower() in _EXTS]
    print(f"源 {len(files)} 张 -> 去水印 mode={args.mode} corner={args.corner}", flush=True)
    n = 0
    for p in files:
        try:
            arr = np.asarray(ImageOps.exif_transpose(Image.open(p)).convert("RGB"))
            if args.mode in ("mask-corner", "both"):
                arr = _mask_corner(arr, args.corner, args.corner_frac)
            if args.mode in ("crop-bottom", "both"):
                arr = _crop_bottom(arr, args.bottom_frac)
            Image.fromarray(arr).save(args.out / (p.stem + ".jpg"), "JPEG", quality=args.q)
            n += 1
        except Exception:
            continue
    print(f"完成: 去水印 {n} 张 -> {args.out}")
    print("提示: 先打开一张样本确认水印位置, 再调 --corner / --corner-frac / --bottom-frac; 抹完再核对水印真没了。")


if __name__ == "__main__":
    main()
