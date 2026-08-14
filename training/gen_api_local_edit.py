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
from locate_blue import locate_amount_auto

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


def _used_srcs(paths: list[Path]) -> set[str]:
    """把已经用过的源图文件名收集起来 —— 传 manifest.csv 或含 manifest 的目录都行。

    **为什么必须有这个**: 这个脚本挑源图是 `rglob + 随机打乱`, 两次运行只要种子和源目录
    一样, 挑到的就是同一批图。历史上因此出过大事故:
    豆包白测试集 397 张里有 394 张和训练用的是同一批原图(99.2%), 整个召回数作废;
    万相白 6 张、千问白 3 张也有重叠。

    **事后靠比对 manifest 去剔, 是补救; 造之前就排除掉, 才是解决。**
    """
    used: set[str] = set()
    for p in paths:
        mps = sorted(p.glob("*/manifest.csv")) if p.is_dir() else [p]
        for mp in mps:
            if not mp.exists():
                continue
            try:
                for r in csv.DictReader(open(mp, encoding="utf-8-sig")):
                    s = (r.get("src") or "").strip()
                    if s:
                        used.add(Path(s).name)
            except Exception as exc:  # noqa: BLE001
                print(f"  读不了 {mp}: {exc}", flush=True)
    return used


def _collect(root: Path, n: int, seed: int, exclude: set[str] | None = None) -> list[Path]:
    files = [p for p in root.rglob("*") if p.suffix.lower() in _EXTS]
    random.Random(seed).shuffle(files)
    ex = exclude or set()
    out: list[Path] = []
    skipped_used = 0
    for p in files:
        if len(out) >= n:
            break
        if p.name in ex:
            skipped_used += 1
            continue                      # 这张已经被别的批次用过, 换一张, 不给自己埋同源的雷
        try:
            with Image.open(p) as im:
                if _is_clean_screenshot(im):
                    out.append(p)
        except Exception:
            continue
    if ex:
        print(f"排除已用源图 {len(ex)} 个, 本次跳过其中 {skipped_used} 张", flush=True)
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
    ap.add_argument("--exclude-src", type=Path, nargs="*", default=None, metavar="PATH",
                    help="**造之前就把已经用过的源图排除掉**, 避免训练测试同源。"
                         "可以给 manifest.csv, 也可以给包含若干 <批次>/manifest.csv 的目录(如 D:\\probe)。"
                         "教训: 豆包白测试集 397 张里 394 张和训练同源, 整个召回数作废 —— "
                         "事后剔是补救, 造之前排除才是解决。")
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

    excl = _used_srcs(list(args.exclude_src)) if args.exclude_src else set()
    srcs = _collect(args.src_root, args.n * 6, args.seed, excl)   # 多备货: 部分会被风控拒 + 部分定位失败
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
    _t0 = time.time()
    page_n = {"blue": 0, "white": 0}          # 造出来的图各是什么版式, 收工时报一下, 免得再靠目录名猜
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
            # ★ 必须按版式分派。**2026-08-11 修正**: 这里原来写死用白图定位器 locate_amount,
            # 而它的判据是 gray<140 的深字浅底; 支付宝蓝底转账页的蓝约 gray=106, 整个背景都被判成"深",
            # 金额是白字反而成了洞 —> 要么定位不到(整张被跳过), 要么锁到页面里唯一的深字浅底元素:
            # **红包促销卡**。于是造出来的"蓝图假图"改的是红包卡不是金额, 用同一个框存下的
            # 蓝图训练裁块也是红包卡的裁块。实测: 万相蓝 38/40、豆包蓝 12/12 全改错了地方。
            loc_s, page_s = locate_amount_auto(src)
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
                    # 各厂商请求格式不同(豆包 images/generations / 万相 responses / 千问 images端点但DashScope形状), 统一走 api_image
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
                loc_g, _ = locate_amount_auto(gen)
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
            page_n[page_s] = page_n.get(page_s, 0) + 1
            # 小批量时**每张都报**: 原来写死每 10 张一报, `--n 10` 就变成全程无输出、
            # 最后才蹦一行 —— 而 gpt-image 这类每张要 30~60 秒, 看上去就是卡死了。
            _step = 1 if args.n <= 20 else 10
            if made % _step == 0:
                print(f"  已造 {made}/{args.n} 张  (用时 {time.time() - _t0:.0f}s, "
                      f"平均 {(time.time() - _t0) / made:.1f}s/张)", flush=True)
            time.sleep(args.sleep)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            if failed <= 5:
                print(f"  失败 {sp.name}: {str(exc)[:180]}", flush=True)
            if failed == 5 and "sensitive" in str(exc):
                print("  提示: 风控在拒图。若你用的是 --prompt-mode amount, 换成默认的 redraw 会好很多"
                      "(指纹一样, 但不涉及'改支付金额'这种敏感请求)。", flush=True)
            if failed >= 10 and made == 0:
                try:
                    from api_image import SUPPORTED, provider_of
                    prov, known = provider_of(args.model), args.model in SUPPORTED
                except Exception:
                    prov, known = "?", False
                print(f"连续失败太多, 先停下检查(key/额度/模型名/网络)。")
                print(f"  当前模型 {args.model} 被路由成 **{prov}** 格式"
                      f"({'表里有登记' if known else '**表里没登记, 是按前缀猜的**'})。")
                if "404" in str(exc):
                    # 实测踩过: 新版千问搬到了 /v1/responses, 但代码没更新, provider_of 按前缀
                    # 猜成老千问 -> 打到 /v1/images/generations -> 每张都 404 openai_error。
                    print("  **全是 404 多半是端点不对** —— 这个模型在平台上走的不是这条路。")
                    print("  1) 先 `git pull`(端点表可能刚更新过)")
                    print("  2) 再 `probe_models.py --models <模型名>` 探它到底走哪个端点")
                    print("  3) 确认 `api_image.SUPPORTED` 里登记了 —— 没登记就会按前缀瞎猜, 而且猜错不报警")
                break

    mf.close()
    print(f"完成: {made} 张 -> {args.out}(跳过 {skipped} 张定位失败, 失败 {failed} 次)")
    print(f"版式: 蓝图 {page_n.get('blue', 0)} 张, 白图 {page_n.get('white', 0)} 张 "
          f"(按像素判, 与 predict_tiled 同口径 —— 别再靠目录名猜版式)")
    print("**先开 2-3 张看一眼**: 应该整张就是原来那张真图, 只有金额数字变了、且看不出拼接痕迹。")
    print("若金额那块明显歪/糊/有方框, 告诉我(说明版面对齐没对上, 我调对齐方式)。")


if __name__ == "__main__":
    main()
