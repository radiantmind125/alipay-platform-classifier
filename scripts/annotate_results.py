"""把设备识别结果标注到图片上,风格对齐检测那边的输出(右侧“识别结果”面板 + 红色高亮设备)。

左边原图(顶部状态栏用红框标出——判定依据),右边白色面板列出结果,设备一行用红字/红框高亮。
随机抽一批供给运营看。默认 300 张;--seed 换一批,--limit 加大,--device 只看某类。只用 PIL。
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
SOURCE_CN = {"resolution": "分辨率", "cnn": "状态栏模型", "none": "-"}
FRAUD_CN = {"pass": "通过", "review": "复核", "reject": "拒绝"}
RED = (235, 20, 20)
DARK = (40, 44, 52)
GRAY = (110, 116, 128)


def _font(size: int):
    for p in ("C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simsun.ttc"):
        try:
            return ImageFont.truetype(p, size), True
        except Exception:
            pass
    return ImageFont.load_default(), False


def annotate(im: Image.Image, r: dict) -> Image.Image:
    w, h = im.size
    pw = int(w * 0.55)                     # 右侧面板宽度
    canvas = Image.new("RGB", (w + pw, h), (247, 249, 252))
    canvas.paste(im, (0, 0))
    d = ImageDraw.Draw(canvas)

    dev = r.get("device", "unknown")
    label, cjk = (DEVICE_CN, True) if _font(20)[1] else (DEVICE_EN, False)
    dev_txt = (DEVICE_CN if cjk else DEVICE_EN).get(dev, dev)
    conf = r.get("confidence")
    conf_txt = f"（{int(round(conf * 100))}%）" if isinstance(conf, (int, float)) else ""
    src = SOURCE_CN.get(r.get("source", ""), r.get("source", "")) if cjk else r.get("source", "")
    fraud = FRAUD_CN.get(r.get("fraud_verdict", ""), r.get("fraud_verdict", "")) if cjk else r.get("fraud_verdict", "")

    # 图上:顶部状态栏红框(判定依据)
    lw = max(3, w // 320)
    d.rectangle([lw, lw, w - lw, int(h * 0.058)], outline=RED, width=lw)
    tag_font, _ = _font(max(22, w // 42))
    d.text((lw + 6, int(h * 0.058) + 4), ("状态栏(判定依据)" if cjk else "status bar"), fill=RED, font=tag_font,
           stroke_width=2, stroke_fill=(255, 255, 255))

    # 右侧面板
    px = w + int(pw * 0.06)
    pr = w + pw - int(pw * 0.06)
    title_font, _ = _font(max(30, pw // 15))
    y = int(h * 0.035)
    d.text((px, y), ("识别结果" if cjk else "Result"), fill=DARK, font=title_font)
    y += int(pw // 15) + int(h * 0.03)

    row_font, _ = _font(max(28, pw // 17))
    bh = int(pw // 17) + int(h * 0.028)

    def box(y0, text, color, thick):
        d.rectangle([px, y0, pr, y0 + bh], outline=color, width=thick)
        d.text((px + int(pw * 0.03), y0 + int(bh * 0.22)), text, fill=color, font=row_font)
        return y0 + bh + int(h * 0.018)

    # 设备行:红色高亮
    y = box(y, f"设备：{dev_txt} {conf_txt}" if cjk else f"Device: {dev_txt} {conf_txt}", RED, max(3, pw // 130))
    # 依据 + 风险(普通)
    y = box(y, (f"判定依据：{src}" if cjk else f"Basis: {src}"), (90, 130, 210), max(2, pw // 200))
    if fraud:
        y = box(y, (f"风险标签：{fraud}" if cjk else f"Risk: {fraud}"), (70, 160, 90), max(2, pw // 200))
    return canvas


def main(argv: list[str] | None = None) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("runs/pool_full/final_device_v2.jsonl"))
    ap.add_argument("--image-root", type=Path, default=Path("D:/download/TempFakeImages"))
    ap.add_argument("--output", type=Path, default=Path("runs/pool_full/annotated"))
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="", help="只标某类:ios|android|uncertain")
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
        try:
            with Image.open(args.image_root / r["file"]) as im:
                out_im = annotate(im.convert("RGB"), r)
        except Exception:
            continue
        out_im.save(args.output / (Path(r["file"]).stem + "_labeled.jpg"), quality=90)
        done += 1
    print(f"标注完成 {done} 张 -> {args.output}（换一批:改 --seed;多标:加大 --limit;只看某类:--device ios）")


if __name__ == "__main__":
    main()
