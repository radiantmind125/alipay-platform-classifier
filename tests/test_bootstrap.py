"""基于分辨率的自举标注的测试。"""

from __future__ import annotations

from alipay_platform.bootstrap import (
    BOOTSTRAP_ABSTAIN,
    BOOTSTRAP_ANDROID,
    BOOTSTRAP_IOS,
    resolution_platform,
)


def test_iphone_resolution_is_ios_both_orientations() -> None:
    assert resolution_platform(1179, 2556) == BOOTSTRAP_IOS
    assert resolution_platform(2556, 1179) == BOOTSTRAP_IOS
    assert resolution_platform(1320, 2868) == BOOTSTRAP_IOS


def test_android_panel_width_is_android() -> None:
    assert resolution_platform(1080, 2400) == BOOTSTRAP_ANDROID
    assert resolution_platform(720, 1600) == BOOTSTRAP_ANDROID
    assert resolution_platform(1440, 3200) == BOOTSTRAP_ANDROID
    assert resolution_platform(2400, 1080) == BOOTSTRAP_ANDROID   # 横向也算


def test_ambiguous_widths_abstain() -> None:
    # 1200~1290 的非常规宽度（多为裁剪/缩放的 iPhone 或少见安卓）-> 弃权。
    for w, h in [(1200, 2664), (1236, 2676), (1260, 2750), (1320, 2856), (1280, 2772)]:
        assert resolution_platform(w, h) == BOOTSTRAP_ABSTAIN


def test_no_iphone_resolution_collides_with_android_widths() -> None:
    # iPhone 分辨率的短边绝不落在安卓面板宽度里，规则之间不会冲突。
    from alipay_platform.bootstrap import ANDROID_PANEL_WIDTHS
    from alipay_platform.metadata_seed import IPHONE_RESOLUTIONS

    for w, h in IPHONE_RESOLUTIONS:
        assert min(w, h) not in ANDROID_PANEL_WIDTHS


def test_invalid_dimensions_abstain() -> None:
    assert resolution_platform(0, 100) == BOOTSTRAP_ABSTAIN
    assert resolution_platform(-1, -1) == BOOTSTRAP_ABSTAIN
