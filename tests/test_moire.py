"""摩尔纹翻拍检测的自足测试:合成"屏栅周期图"应触发,平滑渐变不应触发,且结果确定。"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")

from alipay_platform.moire import moire_verdict


def _screen_grid(n: int = 480, freq: float = 0.12, amp: float = 10.0) -> np.ndarray:
    """合成一张带中频**色度**周期条纹的图,模拟相机拍屏的摩尔纹拍频。"""
    yy, xx = np.mgrid[0:n, 0:n]
    patt = amp * np.cos(2 * np.pi * freq * (xx + yy))          # 离轴中频周期
    rng = np.random.default_rng(0)
    base = 128 + rng.normal(0, 2, (n, n))                       # 轻噪声,避免带内方差为 0
    rgb = np.stack([base + patt, base, base - patt], axis=2)    # 加到 R、减到 B => 色度调制
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _smooth_gradient(n: int = 480) -> np.ndarray:
    ramp = np.linspace(40, 210, n, dtype=np.float32)
    g = np.repeat(ramp[None, :], n, axis=0)
    return np.clip(np.stack([g, g, g], axis=2), 0, 255).astype(np.uint8)


def test_screen_grid_flags_recapture():
    v = moire_verdict(_screen_grid())
    assert v.is_recapture
    assert v.score >= 50.0


def test_smooth_gradient_not_recapture():
    v = moire_verdict(_smooth_gradient())
    assert not v.is_recapture


def test_grid_scores_far_above_gradient():
    assert moire_verdict(_screen_grid()).score > moire_verdict(_smooth_gradient()).score + 20


def test_deterministic():
    img = _screen_grid()
    a, b = moire_verdict(img), moire_verdict(img)
    assert (a.score, a.n_peaks, a.is_recapture) == (b.score, b.n_peaks, b.is_recapture)


def test_rejects_non_rgb():
    with pytest.raises(ValueError):
        moire_verdict(np.zeros((10, 10), dtype=np.uint8))
