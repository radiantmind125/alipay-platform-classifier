r"""造"只有金额那一小块被 AI 改过"的假图 —— 测 SSP/v6 的结构性盲区。

为什么要测这个:
真实欺诈里最省事的做法不是整张重新生成, 而是**拿真截图只改金额那几个数字**。
而 SSP 的判法是: 把图切成很多 32x32 小块, 只挑**纹理最丰富的那一块**去看指纹。
金额那一小块很可能**不是**被挑中的那块 -> 改动看不见 -> 可能整张被判成真图(漏检)。
这是结构性弱点(不是训练不够), 所以必须实测。抓到=稳; 漏=本项目最重要的发现, 要另配局部篡改检测。

做法(真实模拟"AI局部重绘"):
1. locate_amount 定位金额区(复用 engine_b_tamper 的定位, 白底深色金额页)。
2. 只把这一小块过 VAE 往返(= AI 重绘该块, 盖上生成指纹), 其余像素**原封不动**。
3. 边缘羽化贴回, 避免留下明显拼接缝(骗子也会尽量不留痕)。
产出 = 99% 真像素 + 1% AI 像素, 正是"局部 AI 编辑"这类假图。

对照组(--mode full)可造整图 VAE 往返的同源假图, 用来对比"整图改 vs 只改金额"的召回差 ——
这样能干净地把"漏检"归因到**改动面积小**而不是别的因素。

用法(服务器 GPU):
  python training/gen_local_ai_edit.py --src-root D:\download2\TempFakeImages --out D:\probe\localedit --n 300 --device cuda
  python training/gen_local_ai_edit.py --src-root D:\download2\TempFakeImages --out D:\probe\fulledit  --n 300 --mode full --device cuda
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from engine_b_tamper import locate_amount

_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_OPT = (33434, 33437, 34855, 37386)   # 光学拍摄参数; 有=相机照片, 不当真图源


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


def _collect(root: Path, n: int, seed: int) -> list[Path]:
    files = [p for p in root.rglob("*") if p.suffix.lower() in _EXTS]
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


def _vae_roundtrip(vae, arr: np.ndarray, device: str, dt) -> np.ndarray:
    """把一块 RGB uint8 过 VAE 编解码, 返回同尺寸 uint8(尺寸补到 8 的倍数再裁回)。"""
    import torch
    h, w = arr.shape[:2]
    ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8
    if ph or pw:
        arr = cv2.copyMakeBorder(arr, 0, ph, 0, pw, cv2.BORDER_REPLICATE)
    x = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).unsqueeze(0)
    x = (x.float() / 127.5 - 1.0).to(device, dt)
    with torch.no_grad():
        rec = vae.decode(vae.encode(x).latent_dist.sample()).sample
    rec = ((rec.clamp(-1, 1) + 1) * 127.5).round().byte().squeeze(0).permute(1, 2, 0).cpu().numpy()
    return rec[:h, :w]


def _feather_paste(base: np.ndarray, patch: np.ndarray, box, feather: int = 3) -> np.ndarray:
    """把改过的块羽化贴回原图, 不留硬边(骗子也会尽量不留拼接痕迹)。"""
    x0, y0, x1, y1 = box
    m = np.zeros((y1 - y0, x1 - x0), np.float32)
    m[feather:-feather or None, feather:-feather or None] = 1.0
    if feather > 0:
        m = cv2.GaussianBlur(m, (feather * 2 + 1, feather * 2 + 1), 0)
    m = m[..., None]
    out = base.copy()
    out[y0:y1, x0:x1] = (patch.astype(np.float32) * m +
                         base[y0:y1, x0:x1].astype(np.float32) * (1 - m)).round().astype(np.uint8)
    return out


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="造局部 AI 改金额的假图(测 SSP 结构盲区)")
    ap.add_argument("--src-root", type=Path, required=True, help="真图源(白底账单详情那类)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--mode", default="local", choices=["local", "full"],
                    help="local=只重绘金额块(主测); full=整图往返(对照组)")
    ap.add_argument("--pad", type=int, default=6, help="金额框外扩像素(让重绘块自然点)")
    ap.add_argument("--model", default="stabilityai/sd-vae-ft-mse")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    try:
        import torch
        from diffusers import AutoencoderKL
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"需要 torch + diffusers:{exc}")
    dt = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]

    args.out.mkdir(parents=True, exist_ok=True)
    srcs = _collect(args.src_root, args.n * 3, args.seed)   # 多取些, 定位不到金额的会跳过
    if not srcs:
        print("没采到真图源"); return
    print(f"真图源 {len(srcs)} 张, 模式 {args.mode}, 加载 VAE...", flush=True)
    vae = AutoencoderKL.from_pretrained(args.model, torch_dtype=dt).to(args.device).eval()

    made = skipped = 0
    for sp in srcs:
        if made >= args.n:
            break
        try:
            im = ImageOps.exif_transpose(Image.open(sp)).convert("RGB")
            arr = np.asarray(im)
            if args.mode == "full":
                out = _vae_roundtrip(vae, arr, args.device, dt)
            else:
                loc = locate_amount(arr)
                if not loc:
                    skipped += 1
                    continue
                x0, y0, x1, y1 = loc[0], loc[1], loc[2], loc[3]
                h, w = arr.shape[:2]
                x0, y0 = max(0, x0 - args.pad), max(0, y0 - args.pad)
                x1, y1 = min(w, x1 + args.pad), min(h, y1 + args.pad)
                if x1 - x0 < 16 or y1 - y0 < 16:
                    skipped += 1
                    continue
                patch = _vae_roundtrip(vae, arr[y0:y1, x0:x1], args.device, dt)
                out = _feather_paste(arr, patch, (x0, y0, x1, y1))
            Image.fromarray(out).save(args.out / f"ailocal_{args.mode}_{made:06d}.jpg", quality=95)
            made += 1
            if made % 50 == 0:
                print(f"  已造 {made} 张...", flush=True)
        except Exception as exc:  # noqa: BLE001
            if skipped < 3:
                print("失败", sp.name, exc)
            skipped += 1

    print(f"造了 {made} 张({args.mode}) -> {args.out}; 跳过 {skipped} 张(多为定位不到金额)")
    if args.mode == "local":
        print("提示: 只改了金额那一小块, 其余像素原封不动 —— 这就是 SSP 最可能漏的那类。")
        print("对照组: 同命令加 --mode full --out 另一个目录, 比较召回差 = 改动面积的影响。")


if __name__ == "__main__":
    main()
