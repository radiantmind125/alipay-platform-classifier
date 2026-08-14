r"""聚合平台(DMXAPI)上不同厂商的图生图接口 —— 各家格式不一样, 这里统一成一个函数。

为什么要这个: 同一个平台上, 各家模型的请求格式完全不同, 直接用一套 body 只有豆包能通:
- **豆包 Seedream**: `POST /v1/images/generations`, body `{model, prompt, image, response_format, size, watermark}`,
  出图在 `data[0].url`。
- **阿里万相 Wan**: `POST /v1/responses`, DashScope 多模态格式
  `{model, input:{messages:[{role:"user", content:[{image}, {text}]}]}, parameters:{size,n}}`,
  出图在 `output[0].content[0].text`。
  (注: 官方文档写的是"仅文生图", 但实测 content 里塞 image 是能做图生图的。)
- **千问 qwen-image-edit**: `POST /v1/images/generations`, 但 body 是 **DashScope 格式**
  `{model, input:{messages:[{role:"user", content:[{image}, {text}]}]}, parameters:{n}}`。
  **注意它和万相端点不同**: 万相走 `/v1/responses`, 千问走 `/v1/images/generations`,
  但两家的 body 都是 DashScope 的 `input.messages` 形状。
- **可灵 Kling**: 在这个平台上**只有视频**(模型列表里全是 image2video/text2video; 平台导航也把它归在 AI视频)。
  图生图的 doc 页存在但 `/kling/v1/images/generations` 实际没路由(返回网站 HTML)。**对我们(伪造收据截图)没用。**
- **即梦 jimeng / SeedEdit**: 这个平台上**没有渠道**(`model_not_found 无可用渠道`)。
  但**即梦和豆包同属 Seedream 系**, 已有的 doubao-seedream 就是同一族。

**2026-08-08 更正**: 之前记的"qwen-image-edit 在这平台端点不通"**是错的**。
当时只试了豆包那种 `{model, prompt, image}` 的 body, 平台回的是
`For image editing, the message must contain 1~3 image content items. Got 0 image items.`
—— **那不是"模型不存在", 是"图片没放对地方"**。换成 DashScope 的 `input.messages` 就通了。
**教训: 400 带具体报错 ≠ 模型不可用, 要把报错读完再下结论。**
同理 `qwen-image` / `qwen-image-plus` 当时报的是 `Invalid size format: 2K`, 也只是 size 格式问题;
但实测这两个是**文生图**(完全不参考输入图, 出来的是数字显示屏照片), **对我们没用**, 所以没接进来。

用法:
    from api_image import generate_image, SUPPORTED
    png_bytes = generate_image("wan2.7-image", prompt, img_bytes, key)
"""

from __future__ import annotations

import base64
import json
import urllib.request

BASE = "https://www.dmxapi.cn"

# 实测能做"图生图"的模型(能拿真收据当输入重绘)
SUPPORTED = {
    "doubao-seedream-5-0-pro-260628": "seedream",
    "doubao-seedream-4-5-251128": "seedream",
    "doubao-seedream-4-0-250828": "seedream",
    "doubao-seedream-5.0-lite": "seedream",
    "wan2.7-image": "wan",
    "wan2.7-image-pro": "wan",
    "wan2.6-image": "wan",
    "qwen-image-edit": "qwen",
    # ★ 2026 年的新版千问编辑模型**换端点了**: 走 /v1/responses 而不是 /v1/images/generations,
    #   而且尺寸只认 "宽*高", 给 "2K" 会报 `Invalid size format`。
    #   **必须显式登记** —— 否则 provider_of 的 startswith("qwen") 兜底会把它们按老千问路由, 直接 404。
    "qwen-image-edit-plus-20260226": "qwen_resp",
    "qwen-image-edit-max-2026-01-16": "qwen_resp",
    "qwen-image-edit-plus": "qwen_resp",
    # ★ OpenAI 的图像编辑在 /v1/images/edits, 且是 multipart 不是 JSON, 返回 b64 不是 url。
    #   实测在 /v1/images/generations 上会报 `Unknown parameter: 'image'`。
    "gpt-image-2": "openai",
    "gpt-image-1.5": "openai",
    "gpt-image-1": "openai",
}


def provider_of(model: str) -> str:
    """按模型名判定厂商格式(不在表里的按前缀猜, 猜不到就当 seedream)。"""
    if model in SUPPORTED:
        return SUPPORTED[model]
    if model.startswith("wan"):
        return "wan"
    if model.startswith("qwen"):
        return "qwen"
    if model.startswith("kling"):
        raise ValueError("可灵在这个平台上只有视频接口, 做不了图生图 —— 对伪造收据截图没用")
    return "seedream"


def _ensure_min_side(image_bytes: bytes, min_side: int = 512) -> bytes:
    """短边不足就等比放大再发。

    实测万相对小图报 `InvalidParameter: Error validating image`:
    400x120 / 480x160 / 600x200 都被拒, 800x256 / 1024x320 通过 -> 短边下限在 200~256 之间。
    我们发的是"金额那一小条", 天然很小, 所以必须先放大。
    放大无害: 调用方拿到出图后本来就要缩回原尺寸再贴回。
    """
    import io
    from PIL import Image
    im = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = im.size
    if min(w, h) >= min_side:
        return image_bytes
    r = min_side / max(1, min(w, h))
    im = im.resize((max(min_side, int(w * r + 0.5)), max(min_side, int(h * r + 0.5))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=95)
    return buf.getvalue()


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


def _check_aspect(out_bytes: bytes, in_bytes: bytes, model: str, tol: float = 0.25) -> bytes:
    """出图的长宽比必须和输入大致一致, 否则**直接报错**。

    为什么要有这一道
    ----------------
    上一轮把 `qwen-image-edit-plus` 的尺寸写死成方形, 而 `--send crop` 发的是
    **金额那一条**(长宽比约 3)。模型按方形出图, 调用方再 resize 回扁框 ->
    数字被垂直压扁压糊。**150 张全废, 而且一路上没有任何报错** ——
    生成日志写着"失败 0 次", 定位率却从 96% 掉到 9.3%, 直到人工看拼图才发现。

    比例对不上是**机械性错误**, 不是模型能力问题, 应该当场停, 不该产出一堆坏样本。
    上层 `gen_api_local_edit` 连续失败十次会自动早停, 所以系统性的比例问题会立刻暴露。
    """
    try:
        import io as _io
        from PIL import Image as _Im
        with _Im.open(_io.BytesIO(in_bytes)) as a:
            iw, ih = a.size
        with _Im.open(_io.BytesIO(out_bytes)) as b:
            ow, oh = b.size
    except Exception:
        return out_bytes                        # 读不出来就不拦, 交给下游
    ri, ro = iw / max(1, ih), ow / max(1, oh)
    if ri > 0 and abs(ro - ri) / ri > tol:
        raise RuntimeError(
            f"{model} 出图长宽比对不上: 输入 {iw}x{ih}(比 {ri:.2f}) -> 出图 {ow}x{oh}(比 {ro:.2f})。"
            f"贴回去会把金额压变形 —— 先把 size 参数调对再造, 别产出坏样本。")
    return out_bytes


def _post_multipart(url: str, key: str, fields: dict, img: bytes, timeout: int) -> dict:
    """OpenAI 的 /v1/images/edits 只吃 multipart/form-data, 不吃 JSON。"""
    b = "----dmxgen4b1c7e2f"
    parts = []
    for k, v in fields.items():
        parts.append(f"--{b}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
    parts.append((f"--{b}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"a.jpg\"\r\n"
                  f"Content-Type: image/jpeg\r\n\r\n").encode())
    parts.append(img)
    parts.append(f"\r\n--{b}--\r\n".encode())
    req = urllib.request.Request(url, data=b"".join(parts), method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={b}",
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


def _dig_url(r) -> str:
    """各家把出图 url 放的位置都不一样, 挨个位置找一遍。

    **别截断 url** —— DashScope 的地址带签名参数, 截了就 403。
    """
    if not isinstance(r, dict):
        return ""
    d = r.get("data")
    if isinstance(d, list) and d and isinstance(d[0], dict) and d[0].get("url"):
        return d[0]["url"]
    for path in (lambda x: x["output"][0]["content"][0]["text"],
                 lambda x: x["output"]["choices"][0]["message"]["content"][0]["image"],
                 lambda x: x["output"]["results"][0]["url"]):
        try:
            v = path(r)
            if isinstance(v, str) and v.startswith("http"):
                return v
        except Exception:
            pass
    return ""


def generate_image(model: str, prompt: str, image_bytes: bytes, key: str,
                   base: str = BASE, size: str = "2K", timeout: int = 300,
                   watermark: bool = False) -> bytes:
    """把 image_bytes 交给指定模型按 prompt 重绘, 返回出图的原始字节。"""
    prov = provider_of(model)
    if prov in ("wan", "qwen", "qwen_resp"):   # 阿里几家对小图都会 InvalidParameter, 先把短边补到 512
        image_bytes = _ensure_min_side(image_bytes, 512)
    data_uri = "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode("ascii")

    if prov == "openai":
        # 端点/编码/返回三样都和别家不同: /v1/images/edits + multipart + b64_json(没有 url)。
        # ★ size 用 "auto": OpenAI 只收固定几档(1024x1024 / 1536x1024 / 1024x1536),
        #   而我们发的金额条长宽比约 3, 任何一档都对不上 —— 写死方形就是上一轮把 150 张
        #   千问样本压变形的那个错。下面 `_check_aspect` 会兜住这种情况。
        r = _post_multipart(base.rstrip("/") + "/v1/images/edits", key,
                            {"model": model, "prompt": prompt, "size": "auto"},
                            image_bytes, timeout)
        b64 = (r.get("data") or [{}])[0].get("b64_json")
        if not b64:
            raise RuntimeError("OpenAI 响应里没找到 b64_json: " + json.dumps(r)[:300])
        return _check_aspect(base64.b64decode(b64), image_bytes, model)

    if prov == "qwen_resp":
        # 新版千问: 端点同万相(/v1/responses), 但尺寸只认 "宽*高", 给 "2K" 会被拒。
        #
        # ★ 这里踩过一次大坑: 最初写死成 "1024*1024"(那是探端点时随手填的方形),
        #   而我们 --send crop 发出去的是**金额那一条**, 又扁又宽。模型按方形出图,
        #   调用方再 resize 回原来的扁框 -> **数字被垂直压扁压糊**。
        #   实测 150 张里只有 14 张还能定位到金额(9.3%), 而正常应该在 96% 以上。
        #   探端点时只要求"能返回一张图", 尺寸对不对根本没检查, 于是这个值被原样带进了批量生成。
        #
        # 现在按**输入图的真实尺寸**要图, 保持长宽比。
        if "*" in size:
            sz = size
        else:
            import io as _io
            from PIL import Image as _Im
            with _Im.open(_io.BytesIO(image_bytes)) as _im:
                _w, _h = _im.size
            # 取 64 的整数倍, 并夹在接口能接受的范围内
            _r = lambda v: max(512, min(2048, int(round(v / 64)) * 64))
            sz = f"{_r(_w)}*{_r(_h)}"
        payload = {"model": model,
                   "input": {"messages": [{"role": "user",
                                           "content": [{"image": data_uri}, {"text": prompt}]}]},
                   "parameters": {"size": sz, "n": 1}}
        try:
            r = _post(base.rstrip("/") + "/v1/responses", key, payload, timeout)
        except RuntimeError as exc:
            if "size" not in str(exc).lower():
                raise
            payload["parameters"].pop("size", None)     # 尺寸被拒就干脆不指定, 让它跟随输入
            r = _post(base.rstrip("/") + "/v1/responses", key, payload, timeout)
        url = _dig_url(r)
        if not url:
            raise RuntimeError("新版千问响应里没找到图片 url: " + json.dumps(r)[:300])
    elif prov == "qwen":
        # 千问: 端点是豆包那个, body 却是 DashScope 的 input.messages 形状
        payload = {"model": model,
                   "input": {"messages": [{"role": "user",
                                           "content": [{"image": data_uri}, {"text": prompt}]}]},
                   "parameters": {"n": 1}}
        r = _post(base.rstrip("/") + "/v1/images/generations", key, payload, timeout)
        url = _dig_url(r)
        if not url:
            raise RuntimeError("千问响应里没找到图片 url: " + json.dumps(r)[:300])
    elif prov == "wan":
        payload = {"model": model,
                   "input": {"messages": [{"role": "user",
                                           "content": [{"image": data_uri}, {"text": prompt}]}]},
                   "parameters": {"size": size, "n": 1}}
        r = _post(base.rstrip("/") + "/v1/responses", key, payload, timeout)
        try:
            url = r["output"][0]["content"][0]["text"]
        except Exception:
            raise RuntimeError("万相响应里没找到图片 url: " + json.dumps(r)[:300]) from None
    else:                                  # seedream
        payload = {"model": model, "prompt": prompt, "image": data_uri,
                   "response_format": "url", "size": size, "watermark": watermark}
        r = _post(base.rstrip("/") + "/v1/images/generations", key, payload, timeout)
        url = (r.get("data") or [{}])[0].get("url")
        if not url:
            raise RuntimeError("豆包响应里没找到图片 url: " + json.dumps(r)[:300])

    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return _check_aspect(resp.read(), image_bytes, model)
