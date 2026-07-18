"""训练与推理共用的唯一预处理路径（消除 train/serve 偏差 + 机械性杜绝分辨率泄漏）。

规范：原图 -> 裁顶部 STATUS_STRIP_FRACTION 的整宽状态栏条 -> 缩放到固定 512x64 画布
-> 归一化。固定画布让“绝对分辨率”被彻底抹掉——分类器只可能学到状态栏的外观，
这是防“循环学习”（标签来自分辨率）的核心约束。

只用 numpy + PIL，可在 numpy 训练机和 CPU 服务端逐字节一致地运行。
"""

from __future__ import annotations

from typing import Final

import numpy as np
from PIL import Image

# 规范常量：缓存状态栏条、训练、上线三处必须一致。
STATUS_STRIP_FRACTION: Final[float] = 0.08
CANVAS_W: Final[int] = 512
CANVAS_H: Final[int] = 64
_MEAN: Final = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD: Final = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def crop_status_strip(original_rgb: np.ndarray, fraction: float = STATUS_STRIP_FRACTION) -> np.ndarray:
    """从 EXIF 摆正后的原图裁出顶部整宽状态栏条。"""
    if original_rgb.ndim != 3 or original_rgb.shape[2] != 3:
        raise ValueError("需要 HxWx3 的 RGB 图")
    h = original_rgb.shape[0]
    return original_rgb[: max(1, int(round(h * fraction)))]


def strip_to_canvas(strip_rgb: np.ndarray) -> np.ndarray:
    """把状态栏条缩放到固定 512x64（抹掉绝对分辨率）。返回 uint8 HxWx3。"""
    pil = Image.fromarray(strip_rgb).convert("RGB").resize((CANVAS_W, CANVAS_H))
    return np.asarray(pil)


def normalize(canvas_uint8: np.ndarray) -> np.ndarray:
    """归一化到 CHW float32（ImageNet 均值方差）。"""
    a = canvas_uint8.astype(np.float32) / 255.0
    a = (a - _MEAN) / _STD
    return np.ascontiguousarray(a.transpose(2, 0, 1))


def preprocess_strip(strip_rgb: np.ndarray) -> np.ndarray:
    """状态栏条 -> 归一化 CHW（训练读缓存条时用）。"""
    return normalize(strip_to_canvas(strip_rgb))


def preprocess_original(original_rgb: np.ndarray, fraction: float = STATUS_STRIP_FRACTION) -> np.ndarray:
    """原图 -> 状态栏条 -> 归一化 CHW（上线端从原图直接推理时用）。"""
    return normalize(strip_to_canvas(crop_status_strip(original_rgb, fraction)))
