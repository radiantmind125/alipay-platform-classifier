r"""聚合平台(DMXAPI)上不同厂商的图生图接口 —— 各家格式不一样, 这里统一成一个函数。

为什么要这个: 同一个平台上, 各家模型的请求格式完全不同, 直接用一套 body 只有豆包能通:
- **豆包 Seedream**: `POST /v1/images/generations`, body `{model, prompt, image, response_format, size, watermark}`,
  出图在 `data[0].url`。
- **阿里万相 Wan**: `POST /v1/responses`, DashScope 多模态格式
  `{model, input:{messages:[{role:"user", content:[{image}, {text}]}]}, parameters:{size,n}}`,
  出图在 `output[0].content[0].text`。
  (注: 官方文档写的是"仅文生图", 但实测 content 里塞 image 是能做图生图的。)
- **可灵 Kling**: 在这个平台上**只有视频**(模型列表里全是 image2video/text2video; 平台导航也把它归在 AI视频)。
  图生图的 doc 页存在但 `/kling/v1/images/generations` 实际没路由(返回网站 HTML)。**对我们(伪造收据截图)没用。**

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
}


def provider_of(model: str) -> str:
    """按模型名判定厂商格式(不在表里的按前缀猜, 猜不到就当 seedream)。"""
    if model in SUPPORTED:
        return SUPPORTED[model]
    if model.startswith("wan"):
        return "wan"
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


def generate_image(model: str, prompt: str, image_bytes: bytes, key: str,
                   base: str = BASE, size: str = "2K", timeout: int = 300,
                   watermark: bool = False) -> bytes:
    """把 image_bytes 交给指定模型按 prompt 重绘, 返回出图的原始字节。"""
    prov = provider_of(model)
    if prov == "wan":                      # 万相对小图会 InvalidParameter, 先把短边补到 512
        image_bytes = _ensure_min_side(image_bytes, 512)
    data_uri = "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode("ascii")

    if prov == "wan":
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
        return resp.read()
