r"""批量造翻拍训练正样本:拿干净截图合成"相机拍屏"的翻拍图。

用途:
- 给 SSP 的 head_screen(翻拍检测)造带标注正样本;
- 把 --list 指向"假页/篡改页"就能造"翻拍假页"洗白类(screen-yes 且 gen-yes 的难类);
- 输出每张的摩尔纹分,便于筛掉太弱(<阈值)或太强(不真实)的。

用法:
  python scripts\make_recapture.py --image-root C:\...\TempFakeImages --n 500 --per 2 --out-dir runs\sim_recapture
  python scripts\make_recapture.py --list fake_pages.txt --per 3 --out-dir runs\sim_launder
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from alipay_platform.moire import moire_verdict  # noqa: E402
from alipay_platform.recapture_sim import simulate_recapture  # noqa: E402

_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_OPTICAL = (33434, 33437, 34855, 37386)


def _is_clean_screenshot(im: Image.Image) -> bool:
    """只对干净手机截图造翻拍(有光学 EXIF 或大尺寸的多半已是相机照片,不适合当源)。"""
    w, h = im.size
    short = min(w, h)
    aspect = max(w, h) / short if short else 0.0
    try:
        ifd = im.getexif().get_ifd(0x8769)
    except Exception:
        ifd = {}
    has_optical = any(t in ifd for t in _OPTICAL)
    return (not has_optical) and short <= 1500 and 1.6 <= aspect <= 2.6


def main(argv: list[str] | None = None) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="批量造翻拍训练正样本")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image-root", type=Path, help="截图目录(会筛干净手机截图当源)")
    src.add_argument("--list", type=Path, help="源图路径清单(每行一个,不筛,用于假页洗白类)")
    ap.add_argument("--n", type=int, default=500, help="用多少张源图")
    ap.add_argument("--per", type=int, default=2, help="每张源图造几张翻拍")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    if args.list:
        sources = [Path(l.strip()) for l in args.list.read_text(encoding="utf-8").splitlines() if l.strip()]
    else:
        cand = sorted(p for p in args.image_root.rglob("*") if p.suffix.lower() in _EXTS)
        cand = cand[:: max(1, len(cand) // (args.n * 3 or 1))]     # 先稀疏取,再筛
        sources = []
        for p in cand:
            if len(sources) >= args.n:
                break
            try:
                with Image.open(p) as im:
                    if args.list or _is_clean_screenshot(im):
                        sources.append(p)
            except Exception:
                continue
    if not sources:
        print("没有可用源图")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = (args.out_dir / "manifest.jsonl").open("w", encoding="utf-8")
    made = 0
    scores: list[float] = []
    for si, sp in enumerate(sources):
        try:
            with Image.open(sp) as im:
                rgb = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
        except Exception:
            continue
        for k in range(args.per):
            seed = args.seed + si * 100 + k
            from alipay_platform.recapture_sim import _rand_params  # noqa: PLC0415

            p = _rand_params(np.random.default_rng(seed))
            sim = simulate_recapture(rgb, params=p)
            v = moire_verdict(sim)
            out = args.out_dir / f"recap_{si:05d}_{k}.jpg"
            Image.fromarray(sim).save(out, quality=92)
            manifest.write(json.dumps({"src": sp.name, "out": out.name, "seed": seed,
                                       "moire_score": v.score, "is_recapture": v.is_recapture,
                                       "params": asdict(p)}, ensure_ascii=False) + "\n")
            scores.append(v.score)
            made += 1
    manifest.close()

    a = np.array(scores) if scores else np.array([0.0])
    print(f"造了 {made} 张翻拍 -> {args.out_dir}")
    print(f"  摩尔纹分: 中位={np.median(a):.0f} p25={np.percentile(a,25):.0f} p75={np.percentile(a,75):.0f} "
          f"| 达阈值(>=50)占比={float((a>=50).mean()):.0%}(其余是弱摩尔纹的难样本,正好训模型)")
    print(f"  清单 -> {args.out_dir / 'manifest.jsonl'}")


if __name__ == "__main__":
    main()
