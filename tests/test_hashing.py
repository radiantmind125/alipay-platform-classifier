"""感知哈希（dHash）的测试，用于分组 / 去重。"""

from __future__ import annotations

import numpy as np
from PIL import Image

from alipay_platform.hashing import dhash, hamming_distance, is_near_duplicate


def _gradient_image(seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    base = np.linspace(0, 255, 64, dtype=np.uint8)
    field = np.tile(base, (64, 1)) + rng.integers(0, 5, (64, 64), dtype=np.uint8)
    return Image.fromarray(np.clip(field, 0, 255).astype(np.uint8))


def test_identical_images_hash_equal() -> None:
    img = _gradient_image()
    assert dhash(img) == dhash(img.copy())
    assert hamming_distance(dhash(img), dhash(img.copy())) == 0


def test_brightness_shift_is_near_duplicate() -> None:
    img = _gradient_image()
    brighter = Image.fromarray(np.clip(np.asarray(img).astype(np.int16) + 20, 0, 255).astype(np.uint8))
    assert is_near_duplicate(dhash(img), dhash(brighter), threshold=8)


def test_different_images_are_far_apart() -> None:
    a = _gradient_image(seed=1)
    b = Image.fromarray(np.asarray(_gradient_image(seed=2))[:, ::-1].copy())  # 内容左右镜像
    assert hamming_distance(dhash(a), dhash(b)) > 8


def test_hash_is_64_bits_for_default_size() -> None:
    value = dhash(_gradient_image())
    assert 0 <= value < (1 << 64)
