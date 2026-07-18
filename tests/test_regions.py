"""几何区域裁剪的测试。"""

from __future__ import annotations

import numpy as np
import pytest

from alipay_platform.regions import (
    checkmark_crop_from_status_box,
    clock_crop_from_time_box,
    crop_xyxy,
    status_bar_strip,
    status_bar_strip_from_time_box,
)


def _img(h: int = 2532, w: int = 1170) -> np.ndarray:
    # 确定性的横向渐变，便于区分不同裁剪。
    ramp = np.linspace(0, 255, w, dtype=np.uint8)
    return np.repeat(np.tile(ramp, (h, 1))[:, :, None], 3, axis=2)


def test_status_bar_strip_is_full_width_top_band() -> None:
    img = _img()
    crop = status_bar_strip(img, fraction=0.08)
    assert crop.image.shape[1] == img.shape[1]          # 整幅宽度
    assert crop.box[1] == 0                              # 从顶部开始
    assert crop.image.shape[0] == max(48, int(round(2532 * 0.08)))


def test_status_bar_strip_respects_min_px() -> None:
    small = _img(h=400, w=300)
    crop = status_bar_strip(small, fraction=0.01, min_px=48)
    assert crop.image.shape[0] == 48


def test_status_bar_strip_bad_fraction() -> None:
    with pytest.raises(ValueError):
        status_bar_strip(_img(), fraction=0.9)


def test_crop_xyxy_clamps_and_orders() -> None:
    img = _img(h=100, w=100)
    # 坐标反了、又越界，仍应得到裁剪到边界内的目标区域。
    crop = crop_xyxy(img, 90, 10, -20, 40)
    assert crop is not None
    left, top, right, bottom = crop.box
    assert (left, top, right, bottom) == (0, 10, 90, 40)


def test_crop_xyxy_degenerate_returns_none() -> None:
    img = _img(h=100, w=100)
    assert crop_xyxy(img, 10, 10, 11, 50) is None          # 宽度 1 < 2


def test_status_bar_strip_from_time_box() -> None:
    img = _img()
    # 靠近左上角的时钟框。
    crop = status_bar_strip_from_time_box(img, (40, 60, 240, 110), pad_frac=0.6)
    assert crop.box[0] == 0 and crop.box[2] == img.shape[1]  # 整幅宽度
    # bottom = 110 + 0.6*50 = 140
    assert crop.box[3] == 140


def test_clock_crop_from_time_box_pads() -> None:
    img = _img()
    crop = clock_crop_from_time_box(img, (100, 60, 200, 100), pad_frac=0.25)
    assert crop is not None
    left, top, right, bottom = crop.box
    # 宽 100 -> 两侧各留 25；高 40 -> 两侧各留 10
    assert (left, top, right, bottom) == (75, 50, 225, 110)


def test_checkmark_crop_is_left_fraction() -> None:
    img = _img()
    # transfer_status 框占据大部分宽度，靠页面下方。
    box = (300, 1000, 900, 1080)   # w=600, h=80
    crop = checkmark_crop_from_status_box(img, box, left_fraction=0.22, pad_frac=0.0)
    assert crop is not None
    left, top, right, bottom = crop.box
    # 左边在 x1，右边在 x1 + 0.22*600 = 300 + 132 = 432
    assert left == 300 and right == 432
    assert top == 1000 and bottom == 1080


def test_checkmark_crop_degenerate_box_returns_none() -> None:
    img = _img()
    assert checkmark_crop_from_status_box(img, (10, 10, 11, 11)) is None
