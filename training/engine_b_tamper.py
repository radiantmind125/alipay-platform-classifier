r"""engine B:程序化篡改真图金额,造带标注(掩码)的假图 —— 喂 SSP 的 ai 类 + Head B。

针对经理的 03/04 例子(白底账单详情,深色负号金额)。定位金额 = 上半部分最大的深色文字块。
三种改法(都产出 假图 + 掩码,对应真实"改字/换字体/加负号"欺诈):
- refont:把金额抹掉,用系统字体(近但不同的字体)重画一个金额 —— 造假 App 重渲染字段的典型 tell。
- minus:把已有负号拉长/变细,或给正数金额前插一个负号 —— 正是 03/04 那种。
- copymove:把金额里某个数字复制盖到另一个数字上(同字体、真像素,最难察觉的改值)。
掩码=被改区域(自监督定位标签,免人工)。

依赖:numpy + Pillow + cv2(都本地有)。只针对白底深色金额页(蓝底白字金额的另处理)。
用法:
  python training/engine_b_tamper.py --src-root C:\...\white\TempFakeImages --out runs\tamper --n 300 --save-mask
"""

from __future__ import annotations

import argparse
import glob
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\tahoma.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for f in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(f, size)
        except Exception:
            continue
    return ImageFont.load_default()


def locate_amount(rgb: np.ndarray):
    """白底深色金额:上半部分最大的深色文字块。返回 (x0,y0,x1,y1, glyphs) 或 None。"""
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    y0b, y1b = int(h * 0.08), int(h * 0.55)
    dark = (gray[y0b:y1b] < 140).astype(np.uint8)
    nlab, _, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    comps = []
    for i in range(1, nlab):
        x, y, ww, hh, area = stats[i]
        if hh < 0.02 * h or hh > 0.22 * h or area < 20 or ww > 0.5 * w:
            continue
        comps.append([int(x), int(y + y0b), int(ww), int(hh), int(area)])
    if not comps:
        return None
    maxh = max(c[3] for c in comps)
    amt = [c for c in comps if c[3] >= 0.6 * maxh]        # 大字块 = 金额(数字/负号/点)
    amt.sort(key=lambda c: c[0])
    if len(amt) < 2:                                       # 至少要有几位数字
        return None
    x0 = min(c[0] for c in amt); x1 = max(c[0] + c[2] for c in amt)
    ya = min(c[1] for c in amt); yb = max(c[1] + c[3] for c in amt)
    if (x1 - x0) < 0.1 * w:
        return None
    return x0, ya, x1, yb, amt


def _colors(rgb: np.ndarray, box):
    x0, y0, x1, y1 = box
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    roi = rgb[y0:y1, x0:x1].reshape(-1, 3)
    g = gray[y0:y1, x0:x1].reshape(-1)
    fg = np.median(roi[g < 140], axis=0) if (g < 140).any() else np.array([30, 30, 30])
    pad = rgb[max(0, y0 - 8):y0, x0:x1].reshape(-1, 3)     # 上方一条当背景
    bg = np.median(pad, axis=0) if pad.size else np.array([255, 255, 255])
    return tuple(int(v) for v in fg), tuple(int(v) for v in bg)


def edit_refont(img: Image.Image, box, glyphs, rng) -> np.ndarray:
    x0, y0, x1, y1 = box
    fg, bg = _colors(np.asarray(img), box)
    d = ImageDraw.Draw(img)
    d.rectangle([x0 - 2, y0 - 2, x1 + 2, y1 + 2], fill=bg)  # 抹掉原金额
    amt = f"-{rng.randint(1, 99)}{rng.choice(['0','1','8','9'])}{rng.choice(['0','8','9'])}.{rng.randint(10,99)}"
    font = _load_font(max(10, int((y1 - y0) * 1.15)))
    d.text((x0, y0 - int((y1 - y0) * 0.1)), amt, font=font, fill=fg)  # 用系统(错)字体重画
    mask = np.zeros(np.asarray(img).shape[:2], np.uint8)
    mask[y0 - 2:y1 + 2, x0 - 2:x1 + 2] = 255
    return mask


def edit_minus(img: Image.Image, box, glyphs, rng) -> np.ndarray:
    x0, y0, x1, y1 = box
    fg, bg = _colors(np.asarray(img), box)
    left = glyphs[0]                                        # 最左字块
    lx, ly, lw, lh = left[0], left[1], left[2], left[3]
    d = ImageDraw.Draw(img)
    mask = np.zeros(np.asarray(img).shape[:2], np.uint8)
    if lw > 1.6 * lh:                                       # 已是负号 -> 拉长/加宽(变"长负号")
        ext = int(lw * rng.uniform(0.3, 0.7))
        cy = ly + lh // 2
        th = max(2, int(lh * 0.9))
        d.rectangle([lx - ext, cy - th // 2, lx + lw, cy + th // 2], fill=fg)
        mask[cy - th:cy + th, lx - ext - 2:lx + lw + 2] = 255
    else:                                                   # 正数 -> 前面插一个负号
        cy = ly + lh // 2
        th = max(3, int(lh * 0.16))
        mw = int(lh * 0.5)
        d.rectangle([lx - mw - 6, cy - th // 2, lx - 6, cy + th // 2], fill=fg)
        mask[cy - th:cy + th, lx - mw - 8:lx - 4] = 255
    return mask


def edit_copymove(img: Image.Image, box, glyphs, rng) -> np.ndarray:
    arr = np.asarray(img)
    digits = [g for g in glyphs if g[2] < 1.4 * g[3]]      # 近方形 = 数字(排除负号/点)
    if len(digits) < 2:
        return edit_minus(img, box, glyphs, rng)
    src, dst = rng.sample(digits, 2)
    sx, sy, sw, sh = src[0], src[1], src[2], src[3]
    dx, dy, dw, dh = dst[0], dst[1], dst[2], dst[3]
    patch = arr[sy:sy + sh, sx:sx + sw]
    patch = cv2.resize(patch, (dw, dh))
    arr = arr.copy()
    arr[dy:dy + dh, dx:dx + dw] = patch                    # 用一个数字盖另一个(同字体真像素)
    img.paste(Image.fromarray(arr))
    mask = np.zeros(arr.shape[:2], np.uint8)
    mask[dy - 2:dy + dh + 2, dx - 2:dx + dw + 2] = 255
    return mask


_EDITS = {"refont": edit_refont, "minus": edit_minus, "copymove": edit_copymove}


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="engine B 金额篡改合成")
    ap.add_argument("--src-root", type=Path, required=True, help="白底账单详情源(如 white 池)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--edits", nargs="+", default=["refont", "minus", "copymove"], choices=list(_EDITS))
    ap.add_argument("--save-mask", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    files = [f for f in glob.glob(str(args.src_root / "*")) if f.lower().endswith(_EXTS)]
    rng = random.Random(args.seed)
    rng.shuffle(files)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.save_mask:
        (args.out / "masks").mkdir(exist_ok=True)

    made = 0
    tried = 0
    per_edit = {e: 0 for e in args.edits}
    for f in files:
        if made >= args.n:
            break
        tried += 1
        try:
            img = Image.open(f).convert("RGB")
            loc = locate_amount(np.asarray(img))
            if not loc:
                continue
            box = loc[:4]; glyphs = loc[4]
            edit = rng.choice(args.edits)
            mask = _EDITS[edit](img, box, glyphs, rng)
            out = args.out / f"tamper_{edit}_{made:06d}.jpg"
            img.save(out, quality=92)
            if args.save_mask:
                Image.fromarray(mask).save(args.out / "masks" / f"tamper_{edit}_{made:06d}.png")
            per_edit[edit] += 1
            made += 1
        except Exception:
            continue

    print(f"造了 {made} 张篡改假图(定位成功率 ~{made/max(1,tried):.0%})-> {args.out}")
    print(f"  分类型: {per_edit}")
    print("  这些 = 模板/字段类假图,喂 SSP 的 ai 类(贴近真实欺诈)+ Head B 训练/评估;掩码是免费定位标签。")


if __name__ == "__main__":
    main()
