"""把设备识别结果标注到图片上(红色字体),随机抽一批供给运营看。

在每张图左上角用红色大字标出设备类型(苹果/安卓),下面小字标置信度和判定来源。
默认随机抽 300 张;换 --seed 或加大 --limit 可换一批/多标一些。只用 PIL。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

DEVICE_CN = {"ios": "苹果", "android": "安卓", "uncertain": "不确定", "unknown": "未知"}
DEVICE_EN = {"ios": "Apple/iOS", "android": "Android", "uncertain": "Uncertain", "unknown": "Unknown"}
DEVICE_COLOR = {"ios": (235, 20, 20), "android": (235, 20, 20), "uncertain": (255, 140, 0), "unknown": (160, 160, 160)}
SOURCE_CN = {"resolution": "分辨率", "cnn": "模型", "none": ""}


def _font(size: int):
    """优先中文字体(黑体/雅黑);找不到就退回默认字体并改用英文标签。"""
    for p in ("C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simsun.ttc"):
        try:
            return ImageFont.truetype(p, size), True
        except Exception:
            pass
    return ImageFont.load_default(), False


def main(argv: list[str] | None = None) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("runs/pool_full/final_device_v2.jsonl"))
    ap.add_argument("--image-root", type=Path, default=Path("D:/download/TempFakeImages"))
    ap.add_argument("--output", type=Path, default=Path("runs/pool_full/annotated"))
    ap.add_argument("--limit", type=int, default=300, help="随机抽多少张标注")
    ap.add_argument("--seed", type=int, default=0, help="换一批就换个数")
    ap.add_argument("--device", default="", help="只标某类:ios|android|uncertain(默认全部)")
    args = ap.parse_args(argv)

    rows = [json.loads(l) for l in args.results.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.device:
        rows = [r for r in rows if r.get("device") == args.device]
    random.seed(args.seed)
    random.shuffle(rows)
    rows = rows[: args.limit]

    args.output.mkdir(parents=True, exist_ok=True)
    done = 0
    for r in rows:
        src = args.image_root / r["file"]
        try:
            im = Image.open(src).convert("RGB")
        except Exception:
            continue
        w, h = im.size
        fs = max(40, w // 13)
        font, cjk = _font(fs)
        small, _ = _font(max(24, fs // 2))
        dev = r.get("device", "unknown")
        label = (DEVICE_CN if cjk else DEVICE_EN).get(dev, dev)
        color = DEVICE_COLOR.get(dev, (235, 20, 20))
        d = ImageDraw.Draw(im)
        stroke = max(2, fs // 12)
        x, y = int(w * 0.03), int(h * 0.02)
        d.text((x, y), label, fill=color, font=font, stroke_width=stroke, stroke_fill=(0, 0, 0))
        # 小字:置信度 + 来源(分辨率/模型)
        conf = r.get("confidence")
        src_cn = SOURCE_CN.get(r.get("source", ""), r.get("source", "")) if cjk else r.get("source", "")
        parts = []
        if conf is not None:
            parts.append((f"置信度{conf}" if cjk else f"conf {conf}"))
        if src_cn:
            parts.append((f"{src_cn}判定" if cjk else src_cn))
        if parts:
            d.text((x, y + fs + int(fs * 0.15)), "  ".join(parts), fill=color, font=small,
                   stroke_width=max(1, stroke // 2), stroke_fill=(0, 0, 0))
        out = args.output / (Path(r["file"]).stem + "_labeled.jpg")
        im.save(out, quality=90)
        done += 1
    print(f"标注完成 {done} 张 -> {args.output}（换一批:改 --seed;多标:加大 --limit;只看某类:--device ios）")


if __name__ == "__main__":
    main()
