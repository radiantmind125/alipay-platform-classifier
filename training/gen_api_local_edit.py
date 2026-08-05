r"""用**真实服务(豆包 Seedream)**造"只有金额那块是 AI 的"假图 —— 最贴近真实骗子手法的测试集。

为什么需要它:
`gen_local_ai_edit.py` 是用 SD 的 VAE 往返模拟局部编辑, 指纹是 **SD VAE** 的。
但真骗子用的是豆包这类服务, 指纹是 **Seedream** 的 —— 两者不是一回事。
千问那次的教训: 拿相似模型估, 结果真模型漏了七成。所以要用真服务造一批来测。

真实手法(本脚本复刻):
1. 把真收据发给豆包, 让它重绘并把金额改掉 -> 得到整张重绘图(尺寸/内容都变了, 这个 v6 能抓)。
2. **但骗子不会直接交这张** —— 他会把重绘图里**金额那一块**抠出来, 贴回原始真图。
3. 结果 = 99% 真像素 + 金额区是 Seedream 生成的 -> 整图检测器基本看不见(这正是我们的盲区)。

用法(先设 key):
  PowerShell:  $env:DMX_KEY = "sk-xxxx"
  python training/gen_api_local_edit.py --src-root D:\download2\TempFakeImages --out D:\probe\localedit_seedream --n 100
  # 先跑 --n 2 看一眼质量再放量(每张要调一次接口, 慢且花钱)
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
import random
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from api_image import generate_image
from engine_b_tamper import locate_amount

_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_OPT = (33434, 33437, 34855, 37386)


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


def _post(url: str, key: str, payload: dict, timeout: int) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), method="POST",
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + key})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode("utf-8", "replace")[:400]
        except Exception:
            detail = ""
        raise RuntimeError(f"HTTP {e.code}: {detail}") from None


def _feather_paste(base: np.ndarray, patch: np.ndarray, box, feather: int = 3) -> np.ndarray:
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
    ap = argparse.ArgumentParser(description="用真实服务(Seedream)造局部改金额假图")
    ap.add_argument("--src-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=100, help="**先用 2 试跑看质量**, 再放量(每张一次接口调用)")
    ap.add_argument("--model", default="doubao-seedream-4-5-251128",
                    help="实测能图生图的: doubao-seedream-*(豆包/即梦) / wan2.7-image / wan2.7-image-pro(阿里万相)。"
                         "可灵在这平台只有视频接口, 用不了")
    ap.add_argument("--prompt-mode", default="redraw", choices=["redraw", "amount"],
                    help="redraw(默认)=让它照原样重画; amount=让它把金额改成 --amount。"
                         "**默认用 redraw**: 要求改支付凭证金额会触发风控('input image may contain sensitive information')"
                         "而我们只需要它**重新生成**金额那块以带上 Seedream 指纹 —— 改不改数字指纹完全一样。")
    ap.add_argument("--amount", default="8888.88", help="--prompt-mode amount 时把金额改成多少")
    ap.add_argument("--prompt", default="", help="完全自定义提示词(盖过 --prompt-mode)")
    ap.add_argument("--send", default="crop", choices=["crop", "full"],
                    help="crop(默认)=**只把金额那一小块发出去**重绘 —— 小块里没有姓名/账号/订单号, "
                         "不触发风控('input image may contain sensitive information'), 且免去版面对齐; "
                         "full=发整张(会被风控拒 且要重新定位对齐)")
    ap.add_argument("--send-pad", type=float, default=0.5,
                    help="crop 模式下发出去的小块比金额框上下左右各多留几倍框高(给模型一点上下文)")
    ap.add_argument("--save-crops", type=Path, default=None, help="同时存配对裁块(ai/ 与 nature/), 喂线路B训练")
    ap.add_argument("--save-full", type=Path, default=None,
                    help="**同时把整张重绘图也存下来**(仅 --send full 有效), 喂线路A训练。"
                         "一次接口调用同时产出两条线的训练数据, 省一半 API 时间")
    ap.add_argument("--tag", default="",
                    help="裁块/整图文件名里的标记(默认取模型名)。多个模型往同一目录累积时必须区分, 否则互相覆盖")
    ap.add_argument("--crop-min", type=int, default=0,
                    help="0=紧贴改动区裁(训练用, 裁块内全是AI像素); >0=外扩到该最小边长(会混进真像素)")
    ap.add_argument("--base", default="https://www.dmxapi.cn")
    ap.add_argument("--size", default="2K")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--retries", type=int, default=2,
                    help="超时/网络错时重试几次(风控拒图那种永久性错误不重试, 重试也没用)")
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    key = os.environ.get("DMX_KEY", "").strip()
    if not key:
        raise SystemExit('没读到 DMX_KEY。PowerShell: $env:DMX_KEY = "sk-xxxx" (同一窗口再跑本脚本)')

    args.out.mkdir(parents=True, exist_ok=True)
    tag = args.tag or args.model.replace(".", "-").split("/")[-1]   # 防多模型累积时文件名撞车
    if args.save_crops:
        (args.save_crops / "ai").mkdir(parents=True, exist_ok=True)
        (args.save_crops / "nature").mkdir(parents=True, exist_ok=True)
    if args.save_full:
        if args.send != "full":
            raise SystemExit("--save-full 需要配 --send full(crop 模式下我们只拿到小块, 没有整张重绘图)")
        args.save_full.mkdir(parents=True, exist_ok=True)

    srcs = _collect(args.src_root, args.n * 6, args.seed)   # 多备货: 部分会被风控拒 + 部分定位失败
    if not srcs:
        print("没采到真图源"); return
    prompt = args.prompt or (
        "照着这张图 重新画一张一模一样的图片 所有文字 数字和排版都保持不变"
        if args.prompt_mode == "redraw" else
        f"把这张图里的金额数字改成 {args.amount} 其他所有内容 排版 颜色 字体都完全保持不变")
    print(f"真图源 {len(srcs)} 张 | 模型 {args.model} | 目标造 {args.n} 张", flush=True)
    print(f"提示词: {prompt}", flush=True)

    mp = args.out / "manifest.csv"; new = not mp.exists()
    mf = open(mp, "a", newline="", encoding="utf-8-sig"); mw = csv.writer(mf)
    if new:
        mw.writerow(["file", "model", "src"])

    made = failed = skipped = 0
    for sp in srcs:
        if made >= args.n:
            break
        fn = f"apilocal_{tag}_{made:05d}.jpg"
        if (args.out / fn).exists():
            made += 1
            continue
        try:
            src_im = ImageOps.exif_transpose(Image.open(sp)).convert("RGB")
            src = np.asarray(src_im)
            loc_s = locate_amount(src)
            if not loc_s:
                skipped += 1
                continue                      # 原图定位不到金额, 没法做局部替换

            sx0, sy0, sx1, sy1 = loc_s[0], loc_s[1], loc_s[2], loc_s[3]
            H, W = src.shape[:2]

            if args.send == "crop":
                # 只发金额那一小块: 里面没有姓名/账号/订单号 -> 不触发风控; 而且不用再做版面对齐
                pad = int(max(8, (sy1 - sy0) * args.send_pad))
                cx0, cy0 = max(0, sx0 - pad), max(0, sy0 - pad)
                cx1, cy1 = min(W, sx1 + pad), min(H, sy1 + pad)
                buf = io.BytesIO()
                Image.fromarray(src[cy0:cy1, cx0:cx1]).save(buf, "JPEG", quality=95)
                send_bytes = buf.getvalue()
            else:
                send_bytes = sp.read_bytes()

            gen_im = None
            for attempt in range(args.retries + 1):     # 超时/网络抖动重试; 风控拒图不重试(永久错)
                try:
                    # 各厂商请求格式不同(豆包 images/generations vs 万相 responses), 统一走 api_image
                    out_bytes = generate_image(args.model, prompt, send_bytes, key,
                                               base=args.base, size=args.size, timeout=args.timeout)
                    gen_im = Image.open(io.BytesIO(out_bytes)).convert("RGB")
                    break
                except Exception as e:  # noqa: BLE001
                    if "sensitive" in str(e) or attempt >= args.retries:
                        raise
                    time.sleep(2.0 * (attempt + 1))
            if gen_im is None:
                raise RuntimeError("重试后仍失败")

            if args.save_full:      # 整张重绘图本身就是线路A 的训练样本, 顺手存下来
                gen_im.save(args.save_full / f"apifull_{tag}_{made:05d}.jpg", quality=95)

            if args.send == "crop":
                # 重绘结果缩回小块原尺寸, 再从中取出"金额框"那部分贴回原图(位置天然对齐)
                gen = np.asarray(gen_im.resize((cx1 - cx0, cy1 - cy0), Image.LANCZOS))
                ox, oy = sx0 - cx0, sy0 - cy0
                piece = gen[oy:oy + (sy1 - sy0), ox:ox + (sx1 - sx0)]
            else:
                # 整图模式: 服务按自己分辨率重绘 -> 缩回原尺寸再重新定位金额才能对齐
                gen = np.asarray(gen_im.resize(src_im.size, Image.LANCZOS))
                loc_g = locate_amount(gen)
                if not loc_g:
                    skipped += 1
                    continue                  # 重绘图里定位不到金额(版面漂了), 这张放弃
                gx0, gy0, gx1, gy1 = loc_g[0], loc_g[1], loc_g[2], loc_g[3]
                piece = cv2.resize(gen[gy0:gy1, gx0:gx1], (sx1 - sx0, sy1 - sy0), interpolation=cv2.INTER_LANCZOS4)

            out = _feather_paste(src, piece, (sx0, sy0, sx1, sy1))   # 只把金额那块换成 Seedream 生成的

            Image.fromarray(out).save(args.out / fn, quality=95)
            mw.writerow([fn, args.model, sp.name]); mf.flush()

            if args.save_crops:               # 配对裁块: 同位置 改过 vs 原始
                if args.crop_min <= 0:        # 紧贴改动区(躲开羽化边)-> 裁块内全是被AI动过的像素
                    a0, b0, a1, b1 = sy0 + 4, sx0 + 4, sy1 - 4, sx1 - 4
                else:
                    cy, cx = (sy0 + sy1) // 2, (sx0 + sx1) // 2
                    half = max(args.crop_min, sx1 - sx0, sy1 - sy0) // 2
                    a0, b0 = max(0, cy - half), max(0, cx - half)
                    a1, b1 = min(H, cy + half), min(W, cx + half)
                if a1 - a0 >= 32 and b1 - b0 >= 32:
                    Image.fromarray(out[a0:a1, b0:b1]).save(
                        args.save_crops / "ai" / f"crop_{tag}_{made:05d}.jpg", quality=95)
                    Image.fromarray(src[a0:a1, b0:b1]).save(
                        args.save_crops / "nature" / f"crop_{tag}_{made:05d}.jpg", quality=95)
            made += 1
            if made % 10 == 0:
                print(f"  已造 {made} 张...", flush=True)
            time.sleep(args.sleep)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            if failed <= 5:
                print(f"  失败 {sp.name}: {str(exc)[:180]}", flush=True)
            if failed == 5 and "sensitive" in str(exc):
                print("  提示: 风控在拒图。若你用的是 --prompt-mode amount, 换成默认的 redraw 会好很多"
                      "(指纹一样, 但不涉及'改支付金额'这种敏感请求)。", flush=True)
            if failed >= 10 and made == 0:
                print("连续失败太多, 先停下检查(key/额度/模型名/网络)。"); break

    mf.close()
    print(f"完成: {made} 张 -> {args.out}(跳过 {skipped} 张定位失败, 失败 {failed} 次)")
    print("**先开 2-3 张看一眼**: 应该整张就是原来那张真图, 只有金额数字变了、且看不出拼接痕迹。")
    print("若金额那块明显歪/糊/有方框, 告诉我(说明版面对齐没对上, 我调对齐方式)。")


if __name__ == "__main__":
    main()
