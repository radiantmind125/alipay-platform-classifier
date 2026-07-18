"""P0 阻塞门：量化整池“翻拍图”占比，并特别报告“自举标签的毒样本率”。

核心问题不是“池里有多少翻拍”，而是“被分辨率打了 ios/android 标签的图里，有多少其实是翻拍”
——这才是会污染免费标签的部分。

先零解码跑元数据信号（EXIF + 短边尺寸 + 长宽比），再可选对“被标注且元数据像截图”的样本
抽样解码查平坦区噪声（抓 EXIF 被抹但仍有噪声/摩尔纹的翻拍）。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from alipay_platform.photo_detector import flat_region_noise, photo_verdict_from_meta  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", type=Path, default=Path("runs/pool_20260701/inspect.jsonl"))
    ap.add_argument("--bootstrap", type=Path, default=Path("runs/pool_20260701/bootstrap_labels.jsonl"))
    ap.add_argument("--image-root", type=Path, default=Path("../raw_images"))
    ap.add_argument("--pixel-sample", type=int, default=60, help="对已标注样本抽样解码查噪声的数量（0=跳过）")
    args = ap.parse_args(argv)

    recs = [json.loads(l) for l in args.inspect.read_text(encoding="utf-8").splitlines() if l.strip()]
    ok = [r for r in recs if "error" not in r]
    boot = {json.loads(l)["file"]: json.loads(l)["label"] for l in args.bootstrap.read_text(encoding="utf-8").splitlines() if l.strip()}

    reasons: Counter[str] = Counter()
    photo_files: list[str] = []
    for r in ok:
        cap = bool(r.get("has_capture_tags"))
        v = photo_verdict_from_meta(r["width"], r["height"], has_capture_tags=cap)
        if v.is_photo:
            photo_files.append(r["file"])
            reasons[v.reasons[0].split("(")[0].split("，")[0]] += 1
    n = len(ok)
    print(f"总计 {n} 张（元数据零解码扫描）")
    print(f"  元数据判为翻拍/非截图：{len(photo_files)} = {len(photo_files)/n*100:.2f}%")
    for k, v in reasons.most_common():
        print(f"    - {k}: {v}")

    # 毒样本率：被分辨率标了 ios/android，却又被判为翻拍。
    labeled = {f for f, lab in boot.items() if lab in ("ios", "android")}
    poison = [f for f in photo_files if f in labeled]
    print(f"\n自举已标注(ios/android) {len(labeled)} 张；其中元数据判为翻拍(毒样本) {len(poison)} = "
          f"{len(poison)/max(1,len(labeled))*100:.2f}%")
    if poison:
        for f in poison[:10]:
            print(f"    毒样本示例：{f}")

    # 像素兜底：对“已标注且元数据像截图”的抽样查噪声（抓 EXIF 被抹的翻拍）。
    if args.pixel_sample > 0:
        from PIL import Image, ImageOps

        cand = [r for r in ok if r["file"] in labeled and r["file"] not in set(photo_files)]
        step = max(1, len(cand) // args.pixel_sample)
        sample = cand[::step][: args.pixel_sample]
        noises = []
        for r in sample:
            p = args.image_root / r["file"]
            try:
                with Image.open(p) as im:
                    rgb = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
                noises.append(flat_region_noise(rgb))
            except Exception:
                continue
        if noises:
            a = np.array(noises)
            print(f"\n已标注样本像素噪声抽查 n={len(a)}：中位={np.median(a):.2f} p90={np.percentile(a,90):.2f} "
                  f"max={a.max():.2f}（真实截图基线≤0.13；若 max 明显偏高说明有 EXIF 被抹的翻拍）")

    print("\n注意：本样本是小样本且验证过基本全是直接截图。上线前必须在服务器全量池重跑本脚本，"
          "并对判为翻拍的子集人工核对，标定像素噪声/摩尔纹阈值。")


if __name__ == "__main__":
    main()
