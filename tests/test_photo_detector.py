"""翻拍图检测的测试。"""

from __future__ import annotations

import numpy as np

from alipay_platform.photo_detector import (
    flat_region_noise,
    photo_verdict,
    photo_verdict_from_meta,
)


def test_exif_capture_tags_is_photo() -> None:
    v = photo_verdict_from_meta(1170, 2532, has_capture_tags=True)
    assert v.is_photo and v.confidence >= 0.9


def test_large_short_side_is_photo() -> None:
    # 相机照片：短边 3024。
    v = photo_verdict_from_meta(3024, 4032, has_capture_tags=False)
    assert v.is_photo and "短边" in v.reasons[0]


def test_normal_screenshot_not_photo() -> None:
    v = photo_verdict_from_meta(1170, 2532, has_capture_tags=False)
    assert not v.is_photo


def test_weird_aspect_flagged() -> None:
    v = photo_verdict_from_meta(1200, 1400, has_capture_tags=False)  # h/w≈1.17，非手机竖屏
    assert not (1.60 <= v.aspect <= 2.60)


def test_flat_region_noise_low_on_clean_gradient() -> None:
    # 平滑蓝色渐变（截图头部）——逐像素噪声应很低。
    h, w = 512, 256
    ramp = np.linspace(30, 70, h)[:, None]
    img = np.zeros((h, w, 3), np.uint8)
    img[:, :, 2] = np.clip(np.broadcast_to(ramp, (h, w)) + 180, 0, 255).astype(np.uint8)
    assert flat_region_noise(img) < 1.0


def test_flat_region_noise_high_on_noisy() -> None:
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (512, 256, 3), dtype=np.uint8)  # 纯噪声
    assert flat_region_noise(img) > 5.0


def test_photo_verdict_pixel_path_screenshot() -> None:
    # 干净的手机尺寸渐变图 -> 不判翻拍。
    img = np.zeros((2532, 1170, 3), np.uint8)
    img[:, :, 2] = 220
    assert not photo_verdict(img).is_photo
