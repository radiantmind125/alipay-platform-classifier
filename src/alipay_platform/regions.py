"""基于几何位置的区域裁剪，取两个判别区域。

所有裁剪都从 EXIF 摆正后的“原图全分辨率”数组上取（对应检测器工程里的
``geometry.load_upright_rgb``）——绝不从纠正后、缩放后或重新编码后的图上取。这是刻意为之：
需求关注的是状态栏时钟/图标和成功对勾的像素级抗锯齿（“像素差”），而三次插值缩放 + JPEG
重编码恰好会把这些证据抹掉。

传进来的框是原图坐标系下的 ``[x1, y1, x2, y2]``。如果框来自检测器（检测器跑在纠正图上），
调用方必须先用 ``geometry.transform_points(box, rectified_to_original)`` 反投影回原图。

状态栏用几何位置定位（取顶部一条），不依赖检测器，因为检测器的透视纠正可能把状态栏裁掉
——尤其是深色的 iOS / 安卓深色模式状态栏。检测器的 ``time`` 框如果有，只用来收紧这条带的
上下范围。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

BBox = Sequence[float]


@dataclass(frozen=True)
class Crop:
    """一块裁剪，附带它在原图像素坐标下的框 (x1,y1,x2,y2)。"""

    image: np.ndarray
    box: tuple[int, int, int, int]


def _check_rgb(image: np.ndarray) -> tuple[int, int]:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("需要 H x W x 3 的 RGB 图")
    height, width = image.shape[:2]
    if height < 2 or width < 2:
        raise ValueError("图太小")
    return height, width


def crop_xyxy(image: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> Crop | None:
    """把 xyxy 框裁剪到图内并返回裁剪；退化（过小）时返回 None。"""
    height, width = _check_rgb(image)
    left = max(0, int(np.floor(min(x1, x2))))
    top = max(0, int(np.floor(min(y1, y2))))
    right = min(width, int(np.ceil(max(x1, x2))))
    bottom = min(height, int(np.ceil(max(y1, y2))))
    if right - left < 2 or bottom - top < 2:
        return None
    return Crop(image[top:bottom, left:right], (left, top, right, bottom))


def status_bar_strip(image: np.ndarray, *, fraction: float = 0.08, min_px: int = 48) -> Crop:
    """整幅宽度的顶部一条 = 整个状态栏。

    高度取图高的 ``fraction``（下限 ``min_px``，上限为图高一半）。这是不依赖检测器的稳妥默认。
    """
    if not 0 < fraction <= 0.5:
        raise ValueError("fraction 必须在 (0, 0.5] 内")
    height, width = _check_rgb(image)
    strip = min(height // 2, max(min_px, int(round(height * fraction))))
    crop = crop_xyxy(image, 0, 0, width, strip)
    if crop is None:  # pragma: no cover - 最小高度已在 _check_rgb 保证
        raise ValueError("无法生成状态栏条")
    return crop


def _bbox_ints(box: BBox) -> tuple[float, float, float, float]:
    if len(box) != 4:
        raise ValueError("框必须是四个值 [x1, y1, x2, y2]")
    return float(box[0]), float(box[1]), float(box[2]), float(box[3])


def status_bar_strip_from_time_box(
    image: np.ndarray, time_bbox: BBox, *, pad_frac: float = 0.6, min_fraction: float = 0.04
) -> Crop:
    """由检测到的 ``time`` 框决定高度的整宽状态栏条。

    从 y=0 一直取到框底再加上框高的 ``pad_frac``，这样即便时钟框本身很窄，整条状态栏
    （右侧图标、刘海带）也都能包含进来。同时兜底到至少图高的 ``min_fraction``，避免框太小
    时只裁到一条细缝。
    """
    height, width = _check_rgb(image)
    _, y1, _, y2 = _bbox_ints(time_bbox)
    box_h = max(1.0, y2 - y1)
    bottom = max(y2 + pad_frac * box_h, height * min_fraction)
    bottom = min(bottom, height / 2)
    crop = crop_xyxy(image, 0, 0, width, bottom)
    if crop is None:
        return status_bar_strip(image)
    return crop


def clock_crop_from_time_box(image: np.ndarray, time_bbox: BBox, *, pad_frac: float = 0.25) -> Crop | None:
    """时钟数字本身（带边距）——用作 SF Pro vs Roboto 判别特征的输入。"""
    x1, y1, x2, y2 = _bbox_ints(time_bbox)
    pad_x = (x2 - x1) * pad_frac
    pad_y = (y2 - y1) * pad_frac
    return crop_xyxy(image, x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y)


def checkmark_crop_from_status_box(
    image: np.ndarray, transfer_bbox: BBox, *, left_fraction: float = 0.22, pad_frac: float = 0.18
) -> Crop | None:
    """取 ``transfer_status`` 框的左侧一段 = 成功对勾。

    ``transfer_status`` 框是“对勾 + 转账成功文字”合在一起的一个区域；对勾只占最左约 10~15%。
    这里取框宽的 ``left_fraction``，并在最容易抖动的左/上边留足边距，从而把对勾和文字分开
    （否则文字会稀释信号），又不至于把对勾裁掉。
    """
    if not 0 < left_fraction <= 1.0:
        raise ValueError("left_fraction 必须在 (0, 1] 内")
    x1, y1, x2, y2 = _bbox_ints(transfer_bbox)
    box_w = x2 - x1
    box_h = y2 - y1
    if box_w < 2 or box_h < 2:
        return None
    pad_x = box_w * pad_frac
    pad_y = box_h * pad_frac
    right = x1 + box_w * left_fraction
    return crop_xyxy(image, x1 - pad_x, y1 - pad_y, right + pad_x, y2 + pad_y)
