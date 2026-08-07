r"""探一探聚合平台上到底哪些图生图模型能用 —— 经理要"不断新增生成器", 这是第一步。

为什么要单独做这个
------------------
之前判定"qwen-image-edit 在这平台端点不通", 但**只试过豆包那个端点**
(`/v1/images/generations`)。而万相走的是另一条 DashScope 格式的路
(`/v1/responses`, 多模态 messages)。**千问和万相同属阿里 DashScope, 很可能走的是万相那条路。**
所以要**每个模型 x 每个端点**都试一遍, 别拿一次失败就下结论。

一张图一个模型, 花不了几个钱; 但**这是真的会调用付费接口**, 心里有数再跑。

用法(key 只从环境变量读, 不写进仓库):
    $env:DMX_KEY = "sk-..."
    python training/probe_models.py --src D:\download\TempFakeImages\某张真图.jpg --out D:\probe\modelprobe

想加候选模型直接改 --models 或 _CANDIDATES。
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://www.dmxapi.cn"

# 候选模型。分三类:
#   已知能用的留着当"对照组" —— 如果连它们都失败, 那就是 key/网络的问题, 不是模型的问题。
_CANDIDATES = [
    # --- 对照组(已知能用) ---
    "doubao-seedream-4-5-251128",
    "wan2.7-image",
    # --- 千问(经理点名: 国内用的人很多) ---
    "qwen-image-edit",
    "qwen-image-edit-plus",
    "qwen-image",
    "qwen-image-plus",
    "wan2.5-i2i-preview",
    # --- 字节系: 抖音/即梦 跟豆包同源(经理说"抖音那个好像也是基于豆包的") ---
    "doubao-seededit-3-0-i2i-250628",       # 豆包 SeedEdit 图像编辑
    "seededit-3-0-i2i-250628",
    "jimeng-image-edit",
    "jimeng-i2i",
    "seedream-4-0-250828",
]

# 每个模型都把两条端点都试一遍
_ENDPOINTS = ("images", "responses")

_PROMPT = "把图中的金额数字改成 8888.00, 其余部分保持完全不变"


def _post(url: str, key: str, payload: dict, timeout: int):
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), method="POST",
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + key})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return True, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8", "replace")
        except Exception:
            body = ""
        return False, f"HTTP {e.code}: {body[:220]}"
    except Exception as e:                      # 超时/网络
        return False, f"{type(e).__name__}: {str(e)[:180]}"


def _find_url(r) -> str:
    """两家的出图位置不一样, 都找一遍。"""
    if not isinstance(r, dict):
        return ""
    d = r.get("data")
    if isinstance(d, list) and d and isinstance(d[0], dict):
        u = d[0].get("url") or d[0].get("b64_json")
        if u:
            return str(u)[:120]
    try:
        return str(r["output"][0]["content"][0]["text"])[:120]
    except Exception:
        pass
    try:
        return str(r["output"]["results"][0]["url"])[:120]
    except Exception:
        return ""


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="探测聚合平台上哪些图生图模型真的能用")
    ap.add_argument("--src", type=Path, required=True, help="拿来当输入的一张真截图")
    ap.add_argument("--out", type=Path, default=None, help="把成功出图的存这里(可选)")
    ap.add_argument("--models", nargs="*", default=None, help="不给就用内置候选表")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--size", default="2K")
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--min-side", type=int, default=512,
                    help="短边不足先放大(万相对小图会 InvalidParameter)")
    args = ap.parse_args()

    key = os.environ.get("DMX_KEY", "").strip()
    if not key:
        raise SystemExit("请先设环境变量 DMX_KEY —— **别把 key 写进仓库, 这个仓库是公开的**")

    raw = args.src.read_bytes()
    try:                                        # 短边补足, 否则阿里那边会拒
        import io
        from PIL import Image
        im = Image.open(io.BytesIO(raw)).convert("RGB")
        if min(im.size) < args.min_side:
            r = args.min_side / min(im.size)
            im = im.resize((int(im.size[0] * r + .5), int(im.size[1] * r + .5)), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=95)
        raw = buf.getvalue()
    except Exception as e:
        print(f"(缩放跳过: {e})")
    data_uri = "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")

    models = args.models or _CANDIDATES
    print(f"输入图 {args.src.name} ({len(raw)/1024:.0f} KB), 探 {len(models)} 个模型 x 2 个端点")
    print("**这会真的调用付费接口**, 每个成功的模型出一张图。\n")
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)

    ok: list[tuple[str, str]] = []
    for m in models:
        for ep in _ENDPOINTS:
            if ep == "images":
                url = args.base.rstrip("/") + "/v1/images/generations"
                payload = {"model": m, "prompt": _PROMPT, "image": data_uri,
                           "response_format": "url", "size": args.size, "watermark": False}
            else:
                url = args.base.rstrip("/") + "/v1/responses"
                payload = {"model": m,
                           "input": {"messages": [{"role": "user",
                                                   "content": [{"image": data_uri},
                                                               {"text": _PROMPT}]}]},
                           "parameters": {"size": args.size, "n": 1}}
            good, r = _post(url, key, payload, args.timeout)
            if not good:
                print(f"  ✗ {m:34s} [{ep:9s}] {r}")
                continue
            img = _find_url(r)
            if not img:
                print(f"  ? {m:34s} [{ep:9s}] 200 但没找到出图: {json.dumps(r)[:160]}")
                continue
            print(f"  ✓ {m:34s} [{ep:9s}] 出图 {img[:80]}")
            ok.append((m, ep))
            if args.out and img.startswith("http"):
                try:
                    with urllib.request.urlopen(img, timeout=args.timeout) as resp:
                        (args.out / f"{m.replace('/', '_')}__{ep}.png").write_bytes(resp.read())
                except Exception as e:
                    print(f"      (下载失败 {e})")
            break                               # 这个模型已经通了, 不用再试另一个端点

    print(f"\n能用的模型 {len(ok)} 个:")
    for m, ep in ok:
        print(f"  {m:34s} 走 {'/v1/images/generations' if ep == 'images' else '/v1/responses'}")
    print("\n把能用的填进 api_image.py 的 SUPPORTED, 然后就能批量造样本了。")
    print("**先去 --out 目录肉眼看几张**: 金额有没有真的被改、页面有没有崩、右下角有没有 AIGC 水印。")


if __name__ == "__main__":
    main()
