r"""在服务器 GPU 上,用我们自己的真图造 AI 假图(SSP 重训缺的 'ai' 类)—— 不用向经理要样本。

思路:真图海量、假图为零。拿真图过生成模型,给它盖上"生成指纹",就得到带标注的 ai 类。
- **VAE 往返(主力,最稳最便宜):** 真图过 Stable Diffusion 的 AutoencoderKL 编码再解码(不做扩散步)。
  版面几乎不变,但解码器把上采样/高频指纹盖上去 —— 正是 SSP 的 SRM 残差要抓的东西。
- **img2img 低强度(增多样性):** 真图过 img2img,strength 0.2~0.5,轻微重画,指纹更重、内容略漂。
- 多个模型(sd1.5 / sd2 / sdxl 的 VAE)混用 —— SSP 本就设计成"训一个生成器、泛化到没见过的",
  多模型让它学通用指纹而非单一水印。

诚实前提:这造的是"AI 触碰过的页"(生成指纹),对应"AI 生成/AI 洗白"类欺诈。若真实欺诈主要是
模板改字(03/04 负号那种),那不是生成、SSP 抓不到,要用 engine B 篡改合成 + 字形头。所以这份数据
喂 SSP;篡改那类另走 Head B。上线前务必在从图库里挖出的真假图上复核(见 mine 脚本)。

依赖(服务器):pip install torch torchvision diffusers transformers accelerate ;需要 GPU。
用法:
  python training/gen_ai_fakes.py --genuine-roots D:\download\TempFakeImages D:\download2\TempFakeImages \
    --out D:\ssp_alipay\imagenet_ai_0419_sdv4\train\ai --n 4000 --methods vae img2img \
    --models stabilityai/sd-vae-ft-mse runwayml/stable-diffusion-v1-5 --device cuda
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_OPT = (33434, 33437, 34855, 37386)  # 光学拍摄参数;有=相机照片,不拿来当真图源


def _is_clean_screenshot(im: Image.Image) -> bool:
    w, h = im.size
    short = min(w, h)
    if not short:
        return False
    aspect = max(w, h) / short
    try:
        ifd = im.getexif().get_ifd(0x8769)
    except Exception:
        ifd = {}
    return (not any(t in ifd for t in _OPT)) and short <= 1500 and 1.6 <= aspect <= 2.6


def _collect(roots: list[Path], n: int, seed: int) -> list[Path]:
    files: list[Path] = []
    for r in roots:
        files += [p for p in r.rglob("*") if p.suffix.lower() in _EXTS]
    random.Random(seed).shuffle(files)
    out: list[Path] = []
    for p in files:
        if len(out) >= n:
            break
        try:
            with Image.open(p) as im:
                if _is_clean_screenshot(im):
                    out.append(p)
        except Exception:
            continue
    return out


def _to_multiple_of_8(im: Image.Image, cap: int = 768) -> Image.Image:
    w, h = im.size
    s = 1.0 if cap <= 0 else min(1.0, cap / max(w, h))   # cap<=0 = 原生分辨率(不缩,保住 AI 指纹)
    w, h = max(8, int(w * s) // 8 * 8), max(8, int(h * s) // 8 * 8)
    return im.resize((w, h))


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="用真图造 AI 假图(SSP 的 ai 类)")
    ap.add_argument("--genuine-roots", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=4000, help="用多少张真图当源")
    ap.add_argument("--methods", nargs="+", default=["vae"], choices=["vae", "img2img"])
    ap.add_argument("--models", nargs="+", default=["stabilityai/sd-vae-ft-mse"],
                    help="vae 用 AutoencoderKL 模型id;img2img 用完整 SD 模型id")
    ap.add_argument("--strength", type=float, default=0.35, help="img2img 强度(越大改得越多)")
    ap.add_argument("--cap", type=int, default=768, help="最长边上限;**0=原生分辨率**(重训 AI 检测器用 0, 保住指纹别缩)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"需要 torch:{exc}")
    try:
        from diffusers import AutoencoderKL
        if "img2img" in args.methods:
            from diffusers import StableDiffusionImg2ImgPipeline
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"需要 diffusers(pip install diffusers transformers accelerate):{exc}")

    import csv
    args.out.mkdir(parents=True, exist_ok=True)
    srcs = _collect(args.genuine_roots, args.n, args.seed)
    if not srcs:
        print("没采到真图源"); return
    print(f"真图源 {len(srcs)} 张,方法 {args.methods},模型 {args.models}")
    mf = open(args.out / "manifest.csv", "w", newline="", encoding="utf-8-sig")
    mw = csv.writer(mf); mw.writerow(["file", "method", "model"])   # 记生成器身份, 供 held-out 生成器切分

    made = 0
    for method in args.methods:
        for mid in args.models:
            tag = mid.split("/")[-1]
            if method == "vae":
                vae = AutoencoderKL.from_pretrained(mid, torch_dtype=torch.float16).to(args.device).eval()
                for i, sp in enumerate(srcs):
                    try:
                        im = _to_multiple_of_8(ImageOps.exif_transpose(Image.open(sp)).convert("RGB"), args.cap)
                        x = torch.from_numpy(np.array(im)).permute(2, 0, 1).unsqueeze(0)  # np.array 复制=可写,消除警告
                        x = (x.float() / 127.5 - 1.0).to(args.device, torch.float16)
                        with torch.no_grad():
                            lat = vae.encode(x).latent_dist.sample()
                            rec = vae.decode(lat).sample
                        rec = ((rec.clamp(-1, 1) + 1) * 127.5).round().byte().squeeze(0).permute(1, 2, 0).cpu().numpy()
                        fn = f"aivae_{tag}_{i:06d}.jpg"
                        Image.fromarray(rec).save(args.out / fn, quality=95)
                        mw.writerow([fn, "vae", mid]); made += 1
                    except Exception as exc:  # noqa: BLE001
                        if i < 3:
                            print("vae 失败", sp.name, exc)
                del vae
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
            elif method == "img2img":
                pipe = StableDiffusionImg2ImgPipeline.from_pretrained(mid, torch_dtype=torch.float16,
                                                                      safety_checker=None).to(args.device)
                pipe.set_progress_bar_config(disable=True)
                gen = torch.Generator(device=args.device).manual_seed(args.seed)
                for i, sp in enumerate(srcs):
                    try:
                        im = _to_multiple_of_8(ImageOps.exif_transpose(Image.open(sp)).convert("RGB"), args.cap)
                        out = pipe(prompt="a mobile app payment screenshot", image=im,
                                   strength=args.strength, guidance_scale=3.0, generator=gen).images[0]
                        fn = f"aii2i_{tag}_{i:06d}.jpg"
                        out.save(args.out / fn, quality=95)
                        mw.writerow([fn, "img2img", mid]); made += 1
                    except Exception as exc:  # noqa: BLE001
                        if i < 3:
                            print("img2img 失败", sp.name, exc)
                del pipe
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()

    mf.close()
    print(f"造了 {made} 张 AI 假图 -> {args.out}(manifest.csv 已记生成器身份)")
    print("重训 AI 检测器要点:--cap 0 原生分辨率保指纹;多个生成器混造增多样;留一个生成器只进 val 测泛化。")


if __name__ == "__main__":
    main()
