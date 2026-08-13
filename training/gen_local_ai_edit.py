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
from locate_blue import is_blue_page, locate_amount_blue

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


def _used_srcs(paths: list[Path]) -> set[str]:
    """已经被别的批次用过的源图文件名。传 manifest.csv 或含 <批次>/manifest.csv 的目录都行。

    **2026-08-12 加**: 这条路以前既不写 manifest 也不排除已用源图, 结果是
    VAE 那几组裁块的同源情况**根本没法查**(`matrix_audit` 第5节对 `white`/`blue`
    两格一直只能打"无 manifest, 查不了")。API 那条路早就补上了, 这条也要有,
    否则新造的训练裁块可能和 `localedit_heldout` 共用源图, 而且查都查不出来。
    """
    import csv as _csv
    used: set[str] = set()
    for p in paths:
        for mp in (sorted(p.glob("*/manifest.csv")) if p.is_dir() else [p]):
            if not mp.exists():
                continue
            try:
                for r in _csv.DictReader(open(mp, encoding="utf-8-sig")):
                    s = (r.get("src") or "").strip()
                    if s:
                        used.add(Path(s).name)
            except Exception as exc:  # noqa: BLE001
                print(f"  读不了 {mp}: {exc}", flush=True)
    return used


def _ext(fmt: str, src: Path) -> str:
    """输出用什么扩展名。auto = **跟着源图走**(源是 png 就存 png)。

    为什么要有这个开关: 以前这里写死 .jpg quality=95, 于是**我们造的每一张假图都被 JPEG 压过一道**,
    而真图标定池里 55.6% 是从没压过的 png。压缩史在两类之间是不对称的, 召回表因此量的不是纯粹的
    "改没改过"。要把这个混淆拆开, 就得能造出**和 png 真图同压缩史**的假图。
    """
    if fmt == "auto":
        return ".png" if src.suffix.lower() == ".png" else ".jpg"
    return "." + fmt


def _save_im(im: Image.Image, dst: Path) -> None:
    """png 走无损, jpg 固定 q95(和以前逐字一致, 不改已有批次的含义)。"""
    if dst.suffix.lower() == ".png":
        im.save(dst, "PNG", optimize=True)
    else:
        im.convert("RGB").save(dst, "JPEG", quality=95)


def _collect(root: Path, n: int, seed: int, exclude: set[str] | None = None,
             exts: set[str] | None = None) -> list[Path]:
    files = [p for p in root.rglob("*") if p.suffix.lower() in (exts or _EXTS)]
    random.Random(seed).shuffle(files)
    ex = exclude or set()
    out: list[Path] = []
    skipped_used = 0
    for p in files:
        if len(out) >= n:
            break
        if p.name in ex:
            skipped_used += 1
            continue                      # 别的批次用过, 换一张, 不给自己埋同源的雷
        try:
            with Image.open(p) as im:
                if _is_clean_screenshot(im):
                    out.append(p)
        except Exception:
            continue
    if ex:
        print(f"排除已用源图 {len(ex)} 个, 本次跳过其中 {skipped_used} 张", flush=True)
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
    ap.add_argument("--page-type", default="white", choices=["white", "blue", "auto"],
                    help="白底账单详情用 white(默认); 蓝底转账页用 blue(白字蓝底的定位器); auto=按页型自动选")
    ap.add_argument("--save-crops", type=Path, default=None,
                    help="额外产出**配对训练块**: <dir>/ai=改动区裁块, <dir>/nature=同一张原图同一位置的裁块。"
                         "内容完全相同、只差有没有被 AI 动过 -> 训练局部篡改检测最干净的正负样本(标注免费)")
    ap.add_argument("--crop-min", type=int, default=0,
                    help="0(默认)=**紧贴改动区**裁, 裁块内每个像素都被AI动过(训练必须这样, 否则标签噪声大到学不动); "
                         ">0 = 以改动区为中心外扩到这个最小边长(会混进没动过的真像素, 只在想给模型上下文时用)")
    ap.add_argument("--model", default="stabilityai/sd-vae-ft-mse")
    ap.add_argument("--tag", default="",
                    help="输出文件名里的标记(默认取模型名)。**多个生成器往同一目录累积时必须区分**, "
                         "否则第二次跑会把第一次的覆盖掉(计数从0重来)")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-format", default="jpg", choices=["jpg", "png", "auto"],
                    help="假图存成什么格式。**默认 jpg 就是老行为, 已有批次的含义不变。**"
                         "auto=跟着源图走; png=强制无损。"
                         "为什么需要: 我们所有假图都是 jpg q95, 而真图池 55.6% 是从没压过的 png —— "
                         "**从来没有量过一张 png 假图**。而'打开 png 截图 改金额 存 png'是最自然的一种做法。")
    ap.add_argument("--src-ext", nargs="*", default=None, metavar="EXT",
                    help="只从这些扩展名的真图里挑源(如 --src-ext .png)。"
                         "配合 --save-format png 用, 才能造出**全程没被 JPEG 压过**的假图: "
                         "源是 jpg 的话, 压缩痕迹已经烙在像素里了, 存成 png 也洗不掉。")
    ap.add_argument("--exclude-src", type=Path, nargs="*", default=None, metavar="PATH",
                    help=r"造之前就排除已经用过的源图, 避免训练测试同源。可给 manifest.csv, "
                         r"也可给包含 <批次>/manifest.csv 的目录(如 D:\probe)。"
                         r"**这条路以前不写 manifest 也不排除**, 所以 VAE 那几组的同源情况一直查不了。")
    args = ap.parse_args()

    try:
        import torch
        from diffusers import AutoencoderKL
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"需要 torch + diffusers:{exc}")
    dt = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]

    args.out.mkdir(parents=True, exist_ok=True)
    if args.save_crops:
        (args.save_crops / "ai").mkdir(parents=True, exist_ok=True)
        (args.save_crops / "nature").mkdir(parents=True, exist_ok=True)
    excl = _used_srcs(list(args.exclude_src)) if args.exclude_src else set()
    _se = {e.lower() if e.startswith(".") else "." + e.lower() for e in args.src_ext} if args.src_ext else None
    srcs = _collect(args.src_root, args.n * 3, args.seed, excl, _se)   # 多取些, 定位不到金额的会跳过
    if not srcs:
        print("没采到真图源"); return
    tag = args.tag or args.model.rstrip("/").split("/")[-1]     # 文件名标记, 防多生成器互相覆盖
    import csv as _csv
    _mp = args.out / "manifest.csv"; _new = not _mp.exists()
    _mf = open(_mp, "a", newline="", encoding="utf-8-sig"); _mw = _csv.writer(_mf)
    if _new:
        _mw.writerow(["file", "model", "src"])       # 表头与 gen_api_local_edit 一致, 下游工具通用
    print(f"真图源 {len(srcs)} 张, 模式 {args.mode}, 标记 {tag}, 加载 VAE...", flush=True)
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
                blue = is_blue_page(arr)
                if args.page_type == "blue" or (args.page_type == "auto" and blue):
                    loc = locate_amount_blue(arr)
                elif args.page_type == "white" and blue:
                    skipped += 1          # 指定只做白图, 遇到蓝图跳过(白图定位器在蓝图上会误锁红包卡)
                    continue
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
                if args.save_crops:      # 配对裁块: 同一位置 改过的 vs 原始的, 内容一致只差 AI 动没动过
                    if args.crop_min <= 0:
                        # **紧贴改动区**(往里缩 4px 躲开羽化边)-> 裁块里**每个像素都被 AI 动过**。
                        # 这一条很关键: 之前默认往外扩到 128, 结果裁块里八成是没动过的真像素,
                        # patch_img 随机选的那 32x32 多半落在真像素上 -> "ai" 样本其实是真的 ->
                        # 标签噪声过半 -> 训练卡在瞎猜(loss 0.69, ai/nature 准确率反向摆动)。
                        a0, b0, a1, b1 = y0 + 4, x0 + 4, y1 - 4, x1 - 4
                    else:                # 旧行为: 带上下文(想让模型看周边时才用)
                        cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
                        half = max(args.crop_min, x1 - x0, y1 - y0) // 2
                        a0, b0 = max(0, cy - half), max(0, cx - half)
                        a1, b1 = min(h, cy + half), min(w, cx + half)
                    if a1 - a0 >= 32 and b1 - b0 >= 32:
                        # ai 和 nature 必须**用同一个扩展名**: 两类的压缩史一旦不一样,
                        # 模型就能靠"压没压过"走捷径, 训练集自己就废了。
                        _ce = _ext(args.save_format, sp)
                        _save_im(Image.fromarray(out[a0:a1, b0:b1]),
                                 args.save_crops / "ai" / f"crop_{tag}_{made:06d}{_ce}")
                        _save_im(Image.fromarray(arr[a0:a1, b0:b1]),
                                 args.save_crops / "nature" / f"crop_{tag}_{made:06d}{_ce}")
            _fn = f"ailocal_{args.mode}_{tag}_{made:06d}{_ext(args.save_format, sp)}"
            _save_im(Image.fromarray(out), args.out / _fn)
            _mw.writerow([_fn, args.model, sp.name]); _mf.flush()
            made += 1
            if made % 50 == 0:
                print(f"  已造 {made} 张...", flush=True)
        except Exception as exc:  # noqa: BLE001
            if skipped < 3:
                print("失败", sp.name, exc)
            skipped += 1

    _mf.close()
    print(f"造了 {made} 张({args.mode}) -> {args.out}; 跳过 {skipped} 张(多为定位不到金额)")
    if args.save_crops:
        print(f"配对训练块 -> {args.save_crops}\\ai 与 {args.save_crops}\\nature(同位置 同内容 只差有没有被AI动过)")
    if args.mode == "local":
        print("提示: 只改了金额那一小块, 其余像素原封不动 —— 这就是 SSP 最可能漏的那类。")
        print("对照组: 同命令加 --mode full --out 另一个目录, 比较召回差 = 改动面积的影响。")


if __name__ == "__main__":
    main()
