"""状态栏特征提取的测试。"""

from __future__ import annotations

import numpy as np

from alipay_platform.strip_features import FEATURE_NAMES, extract_features


def _blue_strip() -> np.ndarray:
    # 蓝底（模拟支付宝头部）。
    a = np.zeros((60, 1000, 3), dtype=np.uint8)
    a[:, :, 2] = 240  # 高蓝
    a[:, :, 1] = 120
    return a


def test_feature_vector_length_matches_names() -> None:
    feats = extract_features(_blue_strip())
    assert feats.shape == (len(FEATURE_NAMES),)
    assert feats.dtype == np.float32


def test_blank_strip_has_low_content() -> None:
    feats = extract_features(_blue_strip())
    assert feats[0] < 0.05   # content_frac 很低


def test_white_content_raises_density() -> None:
    a = _blue_strip()
    a[20:40, 700:950] = 255   # 右侧一块白色内容
    feats = extract_features(a)
    idx_right = list(FEATURE_NAMES).index("right_third")
    assert feats[0] > 0.02
    assert feats[idx_right] > 0.05   # 右三分之一有内容


def test_deterministic() -> None:
    a = _blue_strip()
    a[10:30, 100:200] = 255
    assert np.allclose(extract_features(a), extract_features(a))


def test_rejects_non_rgb() -> None:
    import pytest

    with pytest.raises(ValueError):
        extract_features(np.zeros((10, 10), dtype=np.uint8))
