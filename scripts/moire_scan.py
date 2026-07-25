r"""相机子集翻拍扫描:先按 元数据+噪声 把图路由成 截图 vs 相机照片,只对相机照片跑
摩尔纹检测,报告 翻拍(photo-of-screen)占比,并把命中样本另存供人工核对/标定阈值。

模态路由(按 EXIF+噪声,不按长宽比——平板截图 4:3 会撞相机):
  有拍摄 EXIF(Exposure/ISO/FNumber/FocalLength/DateTimeOrig)      -> 相机(可信,零解码)
  手机竖屏精确分辨率(短边<=1500 且长宽比 1.6~2.6)且无拍摄 EXIF  -> 截图(零解码)
  其余(短边>1500 或 怪长宽比)                                     -> 解码看平坦区噪声:
      噪声高 -> 相机;噪声≈0 -> 截图(平板/大截图)

用法:
  python scripts\moire_scan.py --image-root C:\...\TempFakeImages --limit 3000 --out runs\moire.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from alipay_platform.moire import SCORE_THRESHOLD, moire_verdict  # noqa: E402
from alipay_platform.photo_detector import (  # noqa: E402
    ASPECT_MAX,
    ASPECT_MIN,
    NOISE_THRESHOLD,
    SHORT_SIDE_MAX,
    flat_region_noise,
)

_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_CAPTURE_TAGS = (36867, 33434, 33437, 34855, 37386)  # DateTimeOrig/Exposure/FNumber/ISO/FocalLen(不用 Make/Model:华为截图也带)


def _has_capture_tags(im: Image.Image) -> bool:
    try:
        exif = im.getexif()
        ifd = exif.get_ifd(0x8769) if exif else {}
        return any(t in ifd for t in _CAPTURE_TAGS)
    except Exception:
        return False


def main(argv: list[str] | None = None) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="相机子集翻拍(摩尔纹)扫描")
    ap.add_argument("--image-root", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0, help="最多扫多少张(0=全部)")
    ap.add_argument("--threshold", type=float, default=SCORE_THRESHOLD)
    ap.add_argument("--out", type=Path, default=None, help="逐图结果 jsonl")
    ap.add_argument("--save-hits", type=Path, default=None, help="把翻拍命中样本路径写到这个 txt 供人工核对")
    args = ap.parse_args(argv)

    files = sorted(p for p in args.image_root.rglob("*") if p.suffix.lower() in _EXTS)
    if args.limit:
        files = files[:: max(1, len(files) // args.limit)][: args.limit]
    if not files:
        print(f"没图片:{args.image_root}")
        return

    n = n_screenshot = n_camera = n_recap = 0
    hits: list[str] = []
    out_fh = args.out.open("w", encoding="utf-8") if args.out else None
    for p in files:
        try:
            with Image.open(p) as im:
                w, h = im.size
                cap = _has_capture_tags(im)
                short = min(w, h)
                aspect = max(w, h) / short if short else 0.0
                # ---- 模态路由 ----
                if cap:
                    modality, rgb = "camera", None
                elif short <= SHORT_SIDE_MAX and ASPECT_MIN <= aspect <= ASPECT_MAX:
                    modality, rgb = "screenshot", None
                else:
                    rgb = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
                    modality = "camera" if flat_region_noise(rgb) > NOISE_THRESHOLD else "screenshot"
        except Exception:
            continue
        n += 1
        if modality == "screenshot":
            n_screenshot += 1
            continue
        n_camera += 1
        try:
            if rgb is None:
                with Image.open(p) as im:
                    rgb = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
            v = moire_verdict(rgb, threshold=args.threshold)
        except Exception:
            continue
        if v.is_recapture:
            n_recap += 1
            hits.append(str(p))
        if out_fh:
            out_fh.write(json.dumps({"file": p.name, "modality": "camera", "w": w, "h": h,
                                     "moire_score": v.score, "n_peaks": v.n_peaks,
                                     "is_recapture": v.is_recapture}, ensure_ascii=False) + "\n")
    if out_fh:
        out_fh.close()
    if args.save_hits and hits:
        args.save_hits.write_text("\n".join(hits), encoding="utf-8")

    print(f"扫描 {n} 张")
    print(f"  路由为截图:  {n_screenshot} ({n_screenshot / max(1, n):.1%})")
    print(f"  路由为相机照片:{n_camera} ({n_camera / max(1, n):.1%})")
    print(f"  其中判为翻拍:  {n_recap} ({n_recap / max(1, n_camera):.1%} of 相机照片,摩尔纹显著度>={args.threshold:.0f})")
    print("\n注意:翻拍阈值在本机 10 张翻拍上标定(峰 52~83),样本小;上线前请在服务器真实翻拍子集上复标。"
          "摩尔纹在=高精度翻拍证据(触发复核);摩尔纹不在≠没翻拍(分辨率对齐/去摩尔纹会没峰),故只作复核触发不硬拒。")
    if args.save_hits and hits:
        print(f"命中样本路径 -> {args.save_hits}(请人工核对确认是翻拍再定阈值)")


if __name__ == "__main__":
    main()
