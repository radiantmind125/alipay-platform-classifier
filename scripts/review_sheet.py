"""从 CNN 预测里抽“边界样本”拼成联系表，人工核对，估计 ambiguous 桶里 iOS 的精确率/召回率。

当前金标太小（对抗集只有 2 张 iOS），无法测硬 iOS。这里抽三组各若干：
  预测=ios（估精确率）、uncertain（边界）、预测=android 的抽样（找漏掉的 iOS=估召回率）。
每行标注模型预测与 p_ios，人工看图判真值即可。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _crop_path(crops: Path, file: str) -> Path:
    return crops / f"{Path(file).with_suffix('').as_posix().replace('/', '__')}.png"


def _font(sz: int):
    for p in ("C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/arial.ttf"):
        try:
            return ImageFont.truetype(p, sz)
        except Exception:
            pass
    return ImageFont.load_default()


def _spread(lst: list, n: int) -> list:
    if len(lst) <= n:
        return lst
    step = len(lst) / n
    return [lst[int(i * step)] for i in range(n)]


def main(argv: list[str] | None = None) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnn", type=Path, default=Path("runs/pool_full/cnn_device.jsonl"))
    ap.add_argument("--crops", type=Path, default=Path("runs/pool_full/status_bar"))
    ap.add_argument("--output", type=Path, default=Path("runs/pool_full/review_sheet.png"))
    ap.add_argument("--n", type=int, default=14, help="每组张数")
    ap.add_argument("--width", type=int, default=1000)
    args = ap.parse_args(argv)

    rows = [json.loads(l) for l in args.cnn.read_text(encoding="utf-8").splitlines() if l.strip()]
    ios = [r for r in rows if r["device"] == "ios"]
    unc = [r for r in rows if r["device"] == "uncertain"]
    andr = [r for r in rows if r["device"] == "android"]
    sample = ([("pred=ios", r) for r in _spread(ios, args.n)]
              + [("uncertain", r) for r in _spread(unc, args.n)]
              + [("pred=android", r) for r in _spread(andr, args.n)])

    band = 240
    imgs: list[np.ndarray] = []
    mapping: list[dict] = []
    f1, f2 = _font(34), _font(20)
    for i, (grp, r) in enumerate(sample, 1):
        sp = _crop_path(args.crops, r["file"])
        if not sp.exists():
            continue
        strip = np.asarray(Image.open(sp).convert("RGB"))
        pil = Image.fromarray(strip)
        sc = args.width / pil.width
        pil = pil.resize((args.width, max(1, int(pil.height * sc))))
        cv = Image.new("RGB", (args.width + band, pil.height), (20, 20, 20))
        cv.paste(pil, (band, 0))
        d = ImageDraw.Draw(cv)
        d.text((8, 6), str(i), fill=(255, 230, 0), font=f1)
        d.text((8, 44), grp, fill=(120, 220, 255), font=f2)
        d.text((8, 68), f"p_ios={r['p_ios']}", fill=(180, 180, 180), font=f2)
        imgs.append(np.asarray(cv))
        mapping.append({"row": i, "group": grp, "file": r["file"], "p_ios": r["p_ios"]})

    if not imgs:
        raise SystemExit("没有可用的状态栏条（检查 --crops 路径）")
    maxw = max(a.shape[1] for a in imgs)
    sep = np.full((4, maxw, 3), 255, np.uint8)
    stk: list[np.ndarray] = []
    for a in imgs:
        if a.shape[1] < maxw:
            a = np.concatenate([a, np.full((a.shape[0], maxw - a.shape[1], 3), 20, np.uint8)], axis=1)
        stk.extend([a, sep])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(stk, axis=0)).save(args.output)
    print(f"review sheet: {args.output}  ({len(imgs)} 行)")
    for m in mapping:
        print(f"  行{m['row']:>2} {m['group']:<14} p_ios={m['p_ios']} {m['file']}")


if __name__ == "__main__":
    main()
