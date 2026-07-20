r"""设备识别·可复用组件(供合并进检测/OCR 主流水线,替换掉基于对号的旧设备模型)。

一句话:给一张图,返回设备判定。内部就是我们那套两层 + 交叉核查:
  分辨率(Tier-0,零解码,~78%) → 判不了再看状态栏 CNN(Tier-1) → 命中分辨率也顺手核一眼状态栏
  (发现"缩放到 iPhone 分辨率"的伪造 → device_prior_conflict)。

**和 mate 那个 --status-style-checkpoint 的区别**:那版看"对号"判设备;这版看**状态栏**(更准),
输入/预处理/结构都不同,所以不是换个 .pt 路径就行,而是把设备这段换成 import 这个类来调。

用法(在 infer.py 里):
    from device_classifier import DeviceClassifier
    dc = DeviceClassifier("checkpoints/statusbar_v2/best.pt", device="cuda")
    r = dc.predict(pil_or_path)      # {device, device_cn, confidence, source, device_prior_conflict, ...}
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))                     # training/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))         # src/
from preprocess import crop_status_strip, strip_to_canvas, normalize  # noqa: E402
from model import build_model  # noqa: E402
from alipay_platform.bootstrap import resolution_platform  # noqa: E402
from alipay_platform.fusion import device_prior_conflict  # noqa: E402

DEVICE_CN = {"ios": "苹果", "android": "安卓", "uncertain": "不确定", "unknown": "未知"}


class DeviceClassifier:
    """设备识别组件。一次加载,多次 predict。线程内复用。"""

    def __init__(self, checkpoint: str | Path, *, device: str = "cuda",
                 conf_uncertain: float = 0.75, crosscheck: bool = True) -> None:
        self.device = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
        payload = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self.model = build_model(2, width=float(payload.get("width", 1.0))).to(self.device)
        self.model.load_state_dict(payload["model_state"])
        self.model.eval()
        self.conf_uncertain = conf_uncertain   # <此值 → 标"不确定";要强制判 0/1 传 0.5
        self.crosscheck = crosscheck           # 命中分辨率时也跑一遍 CNN 查缩放伪造

    @torch.no_grad()
    def _p_ios(self, rgb: np.ndarray) -> float:
        x = torch.from_numpy(normalize(strip_to_canvas(crop_status_strip(rgb)))).unsqueeze(0).to(self.device)
        return float(torch.softmax(self.model(x), 1)[0, 1])

    def predict(self, image: Image.Image | str | Path) -> dict:
        """image: PIL.Image 或 路径。返回设备判定 dict。"""
        pil = image if isinstance(image, Image.Image) else Image.open(image)
        pil = ImageOps.exif_transpose(pil).convert("RGB")
        w, h = pil.size
        plat = resolution_platform(w, h)

        if plat in ("ios", "android"):
            out = {"device": plat, "source": "resolution", "confidence": 0.99, "device_prior_conflict": False}
            if self.crosscheck:
                pv = self._p_ios(np.asarray(pil))
                cdev = "ios" if pv > 0.5 else "android"
                if device_prior_conflict(plat, cdev, max(pv, 1 - pv)):
                    out.update(confidence=0.5, device_prior_conflict=True, cnn_device=cdev,
                               conflict_detail=f"分辨率判{plat}、状态栏判{cdev}(疑似缩放伪造)")
        else:  # 分辨率判不了 → 状态栏 CNN 定夺
            pv = self._p_ios(np.asarray(pil))
            conf = max(pv, 1 - pv)
            dev = "uncertain" if conf < self.conf_uncertain else ("ios" if pv > 0.5 else "android")
            out = {"device": dev, "source": "cnn", "confidence": round(conf, 3),
                   "p_ios": round(pv, 4), "device_prior_conflict": False}

        out["device_cn"] = DEVICE_CN.get(out["device"], out["device"])
        return out
