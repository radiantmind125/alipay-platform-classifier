"""翻拍模拟器的自足测试:确定(同 seed 一致)、输出有效、且能造出可检测的摩尔纹。"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")

from alipay_platform.moire import moire_verdict
from alipay_platform.recapture_sim import RecaptureParams, simulate_recapture


def _fake_screenshot(h: int = 480, w: int = 240) -> np.ndarray:
    """合成一张有平滑色块的"截图"(给子像素渲染提供颜色)。"""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = 128 + 100 * np.sin(xx / w * 3)
    g = 128 + 80 * np.cos(yy / h * 2)
    b = 200 - 60 * (xx / w)
    return np.clip(np.stack([r, g, b], axis=2), 0, 255).astype(np.uint8)


def test_deterministic_same_seed():
    img = _fake_screenshot()
    a = simulate_recapture(img, seed=7)
    b = simulate_recapture(img, seed=7)
    assert np.array_equal(a, b)


def test_output_is_valid_rgb():
    out = simulate_recapture(_fake_screenshot(), seed=1)
    assert out.ndim == 3 and out.shape[2] == 3 and out.dtype == np.uint8


def test_differs_from_input():
    img = _fake_screenshot()
    out = simulate_recapture(img, seed=2)
    assert out.shape[:2] != img.shape[:2] or not np.array_equal(out, img)


def test_produces_detectable_moire():
    """强参数(锐利屏栅 + 适度透视)应造出比原图明显强的摩尔纹。"""
    img = _fake_screenshot(720, 360)
    strong = RecaptureParams(theta_deg=1.6, cam_ratio=1.15, keystone=0.04, grid_gap=0.3,
                             psf_sub=0.8, blur_sigma=0.4, glare=0.0, vignette=0.15,
                             noise_sigma=2.0, jpeg_q=90)
    base = moire_verdict(img).score
    sim = moire_verdict(simulate_recapture(img, params=strong)).score
    assert sim > base + 15


def test_rejects_non_rgb():
    with pytest.raises(ValueError):
        simulate_recapture(np.zeros((10, 10), np.uint8), seed=0)
