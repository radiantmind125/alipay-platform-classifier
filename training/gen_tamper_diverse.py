r"""多样化金额篡改生成器(domain randomization)—— 给金额篡改 CNN 造覆盖全"真实度谱"的训练/评估假图。

思路:标签稀缺时,用**大量不同的篡改手法 + 重随机化**逼模型学"这块不是原生渲染"的可迁移信号,
而不是某一种合成 artifact。每张产出 假图 + 掩码 + 手法名(手法名用于 留一手法 泛化评估)。

CPU 手法(马虎→中等,本地可跑):
- copymove   复制一个数字盖另一个(同字体真像素)
- splice     从**另一张真图**取数字贴过来(真像素但来源不同,渲染/缩放略不匹配)
- retype     擦掉金额用系统字体(多种)重打(错字体)
- minus      插入/加长负号
- scaleshift 把某个数字轻微缩放/移位

每个手法后接**随机后处理**:羽化边界 + 随机 JPEG 质量 + 可选重采样/轻微模糊
—— 关键:随机化压缩/分辨率轴,免得模型偷学固定压缩指纹(之前探针发现的混淆)。

GPU 手法(可选 --ai-harmonize,服务器):对合成后的金额区做**低强度 img2img**融合接缝,
让贴上去的数字看着更原生 = 最"高质量/真实"的一端。需 diffusers + GPU。

用法(服务器):
  python training/gen_tamper_diverse.py --src-root D:\download2\TempFakeImages --out D:\tamper_div --n 6000 --save-mask
  # 加 GPU 高真实度一类:  --ai-harmonize --device cuda --model stabilityai/stable-diffusion-2-1
"""

from __future__ import annotations

import argparse
import csv
import glob
import io
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

import engine_b_tamper as ebt   # 同目录,复用 locate_amount / edit_copymove / edit_minus / _colors

_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
_FONTS = [f for f in (
    r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\segoeuib.ttf", r"C:\Windows\Fonts\tahoma.ttf", r"C:\Windows\Fonts\verdana.ttf",
    r"C:\Windows\Fonts\calibri.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
) if os.path.exists(f)]


def _digits(glyphs):
    return [g for g in glyphs if g[2] < 1.4 * g[3]]   # 近方形 = 数字


def t_splice(img, box, glyphs, rng, bank):
    arr = np.asarray(img); digs = _digits(glyphs)
    if not digs or not bank:
        return ebt.edit_copymove(img, box, glyphs, rng)
    dx, dy, dw, dh = rng.choice(digs)[:4]
    patch = cv2.resize(rng.choice(bank), (max(3, dw), max(4, dh)))
    a = arr.copy(); a[dy:dy + dh, dx:dx + dw] = patch
    img.paste(Image.fromarray(a))
    m = np.zeros(arr.shape[:2], np.uint8); m[dy - 2:dy + dh + 2, dx - 2:dx + dw + 2] = 255
    return m


def t_scaleshift(img, box, glyphs, rng, bank):
    arr = np.asarray(img); digs = _digits(glyphs)
    if not digs:
        return ebt.edit_minus(img, box, glyphs, rng)
    x, y, w, h = rng.choice(digs)[:4]
    fg, bg = ebt._colors(arr, box)
    f = rng.uniform(0.78, 1.28); nw, nh = max(3, int(w * f)), max(4, int(h * f))
    patch = cv2.resize(arr[y:y + h, x:x + w], (nw, nh))
    a = arr.copy(); a[max(0, y - 1):y + h + 1, max(0, x - 1):x + w + 1] = bg
    sx, sy = x + rng.randint(-3, 3), y + rng.randint(-2, 2)
    H, W = a.shape[:2]; sx = max(0, min(W - nw, sx)); sy = max(0, min(H - nh, sy))
    a[sy:sy + nh, sx:sx + nw] = patch
    img.paste(Image.fromarray(a))
    m = np.zeros(arr.shape[:2], np.uint8)
    m[max(0, y - 1):y + h + 1, max(0, x - 1):x + w + 1] = 255
    m[sy:sy + nh, sx:sx + nw] = 255
    return m


def t_retype(img, box, glyphs, rng, bank):
    arr = np.asarray(img); fg, bg = ebt._colors(arr, box)
    x0, y0, x1, y1 = box
    d = ImageDraw.Draw(img)
    d.rectangle([x0 - 2, y0 - 2, x1 + 2, y1 + 2], fill=tuple(int(v) for v in bg))
    amt = f"-{rng.randint(1, 99)}{rng.choice('0189')}{rng.choice('089')}.{rng.randint(10, 99)}"
    if _FONTS:
        font = ImageFont.truetype(rng.choice(_FONTS), max(10, int((y1 - y0) * rng.uniform(1.0, 1.2))))
    else:
        font = ImageFont.load_default()
    d.text((x0, y0 - int((y1 - y0) * 0.1)), amt, font=font, fill=tuple(int(v) for v in fg))
    m = np.zeros(arr.shape[:2], np.uint8); m[y0 - 2:y1 + 2, x0 - 2:x1 + 2] = 255
    return m


_TECHS = {
    "copymove": lambda img, b, g, rng, bank: ebt.edit_copymove(img, b, g, rng),
    "minus": lambda img, b, g, rng, bank: ebt.edit_minus(img, b, g, rng),
    "retype": t_retype,
    "splice": t_splice,
    "scaleshift": t_scaleshift,
}


def _feather(img, mask, rng):
    """沿掩码边界轻微羽化,弱化硬接缝(随机化 tell 强度)。"""
    if rng.random() > 0.6:
        return img
    a = np.asarray(img).astype(np.float32)
    blur = np.asarray(img.filter(ImageFilter.GaussianBlur(rng.uniform(0.6, 1.4)))).astype(np.float32)
    edge = cv2.dilate(mask, np.ones((5, 5), np.uint8)) - cv2.erode(mask, np.ones((5, 5), np.uint8))
    w = (edge > 0)[:, :, None]
    out = np.where(w, blur, a).astype(np.uint8)
    return Image.fromarray(out)


def _post(img, rng):
    """随机后处理:羽化交给上面;这里随机 JPEG 质量 + 可选重采样/轻模糊。破坏固定压缩指纹。"""
    if rng.random() < 0.3:
        w, h = img.size; s = rng.uniform(0.82, 0.97)
        img = img.resize((max(8, int(w * s)), max(8, int(h * s))), Image.LANCZOS).resize((w, h), Image.LANCZOS)
    if rng.random() < 0.22:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.8)))
    b = io.BytesIO(); img.save(b, "JPEG", quality=rng.randint(55, 95))
    return Image.open(b).convert("RGB")


def _build_bank(files, rng, k=40):
    bank = []
    for f in files[:400]:
        if len(bank) >= k:
            break
        try:
            arr = np.asarray(Image.open(f).convert("RGB"))
        except Exception:
            continue
        loc = ebt.locate_amount(arr)
        if not loc:
            continue
        for g in _digits(loc[4])[:3]:
            x, y, w, h = g[:4]
            if h >= 8 and 0.30 <= w / max(1, h) <= 0.92:   # 数字形状(排除 ¥/点/方形符号)
                bank.append(arr[y:y + h, x:x + w].copy())
    return bank


def _ai_harmonize(img, box, pipe, gen, rng):
    """GPU 可选:对金额区做低强度 img2img,融合接缝 -> 更真实。需 diffusers。"""
    x0, y0, x1, y1 = box
    pad = 8
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    crop = img.crop((x0, y0, x1 + pad, y1 + pad))
    w, h = crop.size
    cw, ch = max(64, (w // 8) * 8), max(64, (h // 8) * 8)
    out = pipe(prompt="a clean mobile bill screenshot number", image=crop.resize((cw, ch)),
               strength=rng.uniform(0.12, 0.22), guidance_scale=2.5, generator=gen).images[0].resize((w, h))
    img = img.copy(); img.paste(out, (x0, y0))
    return img


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="多样化金额篡改生成器")
    ap.add_argument("--src-root", type=Path, required=True, help="白底账单详情源(white 池)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--techs", nargs="+", default=list(_TECHS), choices=list(_TECHS))
    ap.add_argument("--save-mask", action="store_true")
    ap.add_argument("--save-clean", action="store_true", help="同时存一份未篡改原图(负样本对照)")
    ap.add_argument("--ai-harmonize", action="store_true", help="GPU:对金额区低强度img2img融合(需diffusers)")
    ap.add_argument("--ai-frac", type=float, default=0.35, help="多大比例走 AI 融合")
    ap.add_argument("--model", default="stabilityai/stable-diffusion-2-1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    files = [f for f in glob.glob(str(args.src_root / "*")) if f.lower().endswith(_EXTS)]
    rng = random.Random(args.seed); rng.shuffle(files)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.save_mask:
        (args.out / "masks").mkdir(exist_ok=True)
    if args.save_clean:
        (args.out / "clean").mkdir(exist_ok=True)

    pipe = gen = None
    if args.ai_harmonize:
        import torch
        from diffusers import StableDiffusionImg2ImgPipeline
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(args.model, torch_dtype=torch.float16,
                                                              safety_checker=None).to(args.device)
        pipe.set_progress_bar_config(disable=True)
        gen = torch.Generator(device=args.device).manual_seed(args.seed)

    print("建 splice 数字库...")
    bank = _build_bank(files, rng)
    print(f"  数字库 {len(bank)} 个")

    made = tried = 0
    per = {t: 0 for t in args.techs}
    manifest = open(args.out / "manifest.csv", "w", newline="", encoding="utf-8-sig")
    mw = csv.writer(manifest); mw.writerow(["file", "technique", "ai_harmonized", "src"])
    for f in files:
        if made >= args.n:
            break
        tried += 1
        try:
            img = Image.open(f).convert("RGB")
            arr = np.asarray(img)
            loc = ebt.locate_amount(arr)
            if not loc:
                continue
            box, glyphs = loc[:4], loc[4]
            if args.save_clean:
                img.copy().save(args.out / "clean" / f"clean_{made:06d}.jpg", quality=92)
            tech = rng.choice(args.techs)
            mask = _TECHS[tech](img, box, glyphs, rng, bank)
            img = _feather(img, mask, rng)
            ai = False
            if pipe is not None and rng.random() < args.ai_frac:
                try:
                    img = _ai_harmonize(img, box, pipe, gen, rng); ai = True
                except Exception as exc:
                    if made < 3:
                        print("ai_harmonize 失败", exc)
            img = _post(img, rng)
            name = f"tamper_{tech}_{'ai' if ai else 'cpu'}_{made:06d}.jpg"
            img.save(args.out / name, quality=rng.randint(88, 95))
            if args.save_mask:
                Image.fromarray(mask).save(args.out / "masks" / name.replace(".jpg", ".png"))
            mw.writerow([name, tech, int(ai), os.path.basename(f)])
            per[tech] += 1
            made += 1
        except Exception:
            continue
    manifest.close()
    print(f"造了 {made} 张(定位成功率 ~{made/max(1,tried):.0%})-> {args.out}")
    print(f"  分手法: {per}")
    print("  留一手法评估: 训练时留下 1-2 个 technique 只用于测试, 看模型能否抓没见过的手法(泛化真信号的证据)。")
    print("  诚实: 合成只训, 真召回未知; 误杀在真图上单独量; review-only 部署顺带收真标签。")


if __name__ == "__main__":
    main()
