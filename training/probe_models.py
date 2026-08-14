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

# 每个模型都把这几条端点试一遍
#   images    /v1/images/generations  —— 豆包/万相 走这条
#   responses /v1/responses           —— 阿里 DashScope 那套多模态 messages
#   edits     /v1/images/edits        —— **OpenAI 系的图像编辑在这条**, 而且是 multipart 不是 JSON。
#             实测 gpt-image-2 在 generations 上报 `Unknown parameter: 'image'`, 就是因为走错端点了。
_ENDPOINTS = ("images", "responses", "edits")

# 尺寸写法两家不一样: 豆包吃 "2K", 阿里新版要 "宽*高"。
# 实测 qwen-image-edit-plus-20260226 只差这一个参数: `Invalid size format: 2K, expected width*height`。
_ALT_SIZE = "1024*1024"

_PROMPT = "把图中的金额数字改成 8888.00, 其余部分保持完全不变"


def _post_multipart(url: str, key: str, fields: dict, img: bytes, timeout: int):
    """OpenAI 的 /v1/images/edits 只吃 multipart/form-data, 不吃 JSON。"""
    b = "----dmxprobe7f3a9c1e"
    parts = []
    for k, v in fields.items():
        parts.append(f"--{b}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
    parts.append((f"--{b}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"a.jpg\"\r\n"
                  f"Content-Type: image/jpeg\r\n\r\n").encode())
    parts.append(img)
    parts.append(f"\r\n--{b}--\r\n".encode())
    body = b"".join(parts)
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={b}",
                                          "Authorization": "Bearer " + key})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return True, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            body_s = e.read().decode("utf-8", "replace")
        except Exception:
            body_s = ""
        return False, f"HTTP {e.code}: {body_s[:220]}"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {str(e)[:180]}"


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
    """各家出图的位置不一样, 都找一遍。

    **返回完整值, 绝不截断** —— 以前这里 `[:120]` 是为了打印好看,
    但同一个字符串又被拿去下载, 于是下的是**半截 URL**, 阿里 OSS 回 403 / 火山回 404,
    看着像"平台不让下", 其实是我们自己把地址截了。截断只该发生在打印那一行。
    """
    if not isinstance(r, dict):
        return ""
    d = r.get("data")
    if isinstance(d, list) and d and isinstance(d[0], dict):
        u = d[0].get("url") or d[0].get("b64_json")
        if u:
            return str(u)
    try:
        return str(r["output"][0]["content"][0]["text"])
    except Exception:
        pass
    try:
        return str(r["output"]["results"][0]["url"])
    except Exception:
        return ""


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="探测聚合平台上哪些图生图模型真的能用")
    ap.add_argument("--list", action="store_true",
                    help="**先问平台自己有哪些模型**(`/v1/models`), 不出图不花钱。"
                         "别靠记忆写模型名 —— 各家版本换得很快, 聚合平台上的命名也和官方不一样。"
                         "拿这份清单去挑候选, 比猜准得多。")
    ap.add_argument("--grep", nargs="*", default=None,
                    help="配合 --list: 只显示名字里含这些子串的(如 image edit i2i flux gemini)")
    ap.add_argument("--src", type=Path, default=None, help="拿来当输入的一张真截图(--list 时不需要)")
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

    if args.list:
        req = urllib.request.Request(f"{args.base}/v1/models",
                                     headers={"Authorization": f"Bearer {key}"})
        try:
            with urllib.request.urlopen(req, timeout=args.timeout) as r:
                body = json.loads(r.read().decode("utf-8", "replace"))
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"取模型清单失败: {exc}\n(有的聚合平台不开 /v1/models, 那就只能拿候选名去试)")
        ids = sorted({(m.get("id") or "").strip() for m in (body.get("data") or []) if m.get("id")})
        if not ids:
            raise SystemExit(f"清单是空的, 原始返回前 300 字:\n{json.dumps(body)[:300]}")
        if args.grep:
            pats = [g.lower() for g in args.grep]
            ids = [i for i in ids if any(p in i.lower() for p in pats)]
        print(f"平台共 {len(body.get('data') or [])} 个模型"
              + (f", 匹配 {len(ids)} 个" if args.grep else "") + ":\n")
        for i in ids:
            print(f"  {i}")
        print("\n挑候选时的两条:")
        print("  1) **要真正没训过的**才有意义。我们训练里已经有阿里(千问/万相)和字节(豆包/即梦),")
        print("     还有本地开源 VAE: SD / SDXL / FLUX-VAE / TAESD / ostris / Qwen-Image。")
        print("     **共用同一个 VAE 的模型不算没训过** —— 比如 FLUX 系的编辑模型和我们训过的 FLUX-VAE 同源。")
        print("  2) 最干净的留一组是**自研闭源解码器**的那些(谷歌 / OpenAI / 腾讯 / 智谱 之类)。")
        return

    if not args.src:
        raise SystemExit("除了 --list 之外都要给 --src(一张真截图当输入)")
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

    def _build(ep: str, size: str):
        if ep == "images":
            return (args.base.rstrip("/") + "/v1/images/generations",
                    {"model": m, "prompt": _PROMPT, "image": data_uri,
                     "response_format": "url", "size": size, "watermark": False})
        return (args.base.rstrip("/") + "/v1/responses",
                {"model": m,
                 "input": {"messages": [{"role": "user",
                                         "content": [{"image": data_uri}, {"text": _PROMPT}]}]},
                 "parameters": {"size": size, "n": 1}})

    ok: list[tuple[str, str]] = []
    for m in models:
        for ep in _ENDPOINTS:
            if ep == "edits":                    # OpenAI 系: multipart, 且尺寸用 宽x高
                url = args.base.rstrip("/") + "/v1/images/edits"
                good, r = _post_multipart(url, key,
                                          {"model": m, "prompt": _PROMPT, "size": "1024x1024"},
                                          raw, args.timeout)
            else:
                url, payload = _build(ep, args.size)
                good, r = _post(url, key, payload, args.timeout)
                # 尺寸格式两家不一样 —— 报的就是尺寸问题时, 换一种写法再试一次, 别就此判死
                if not good and isinstance(r, str) and "size" in r.lower():
                    url, payload = _build(ep, _ALT_SIZE)
                    good2, r2 = _post(url, key, payload, args.timeout)
                    if good2 or len(str(r2)) < len(str(r)):
                        print(f"    (尺寸换成 {_ALT_SIZE} 重试)")
                        good, r = good2, r2
            if not good:
                print(f"  ✗ {m:34s} [{ep:9s}] {r}")
                continue
            img = _find_url(r)
            if not img:
                print(f"  ? {m:34s} [{ep:9s}] 200 但没找到出图: {json.dumps(r)[:160]}")
                continue
            print(f"  ✓ {m:34s} [{ep:9s}] 出图 {img[:80]}")
            ok.append((m, ep))
            if args.out:
                try:
                    if img.startswith("http"):
                        rq = urllib.request.Request(img, headers={"User-Agent": "Mozilla/5.0"})
                        with urllib.request.urlopen(rq, timeout=args.timeout) as resp:
                            blob = resp.read()
                    else:
                        # OpenAI 走 b64_json 直接返回图片字节, 不给 URL ——
                        # 以前只处理 http 开头的, 这一路的图**一张都没存下来**
                        blob = base64.b64decode(img.split(",", 1)[-1])
                    dst = args.out / f"{m.replace('/', '_')}__{ep}.png"
                    dst.write_bytes(blob)
                    print(f"      -> 已存 {dst.name} ({len(blob) / 1024:.0f} KB)")
                except Exception as e:  # noqa: BLE001
                    print(f"      (存图失败 {type(e).__name__}: {str(e)[:120]})")
            break                               # 这个模型已经通了, 不用再试另一个端点

    # 端点名 -> 真实路径。以前这里写的是二选一的三元表达式, 加了 edits 之后
    # **走 edits 通的会被标成 /v1/responses** —— 照着配下游就是错的, 而且错得很像对的。
    _EP_URL = {"images": "/v1/images/generations",
               "responses": "/v1/responses",
               "edits": "/v1/images/edits(multipart)"}
    print(f"\n能用的模型 {len(ok)} 个:")
    for m, ep in ok:
        print(f"  {m:34s} 走 {_EP_URL.get(ep, ep)}")
    print("\n把能用的填进 api_image.py 的 SUPPORTED, 然后就能批量造样本了。")
    print("**先去 --out 目录肉眼看几张**: 金额有没有真的被改、页面有没有崩、右下角有没有 AIGC 水印。")


if __name__ == "__main__":
    main()
