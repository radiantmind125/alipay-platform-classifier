"""平台标签固定定义的测试。"""

from __future__ import annotations

import pytest

from alipay_platform.platform_labels import (
    ID_TO_LABEL,
    LABEL_TO_ID,
    NUM_CLASSES,
    PLATFORM_CLASSES,
    validate_checkpoint_classes,
    validate_label,
    validate_vote,
)


def test_class_order_is_frozen() -> None:
    assert PLATFORM_CLASSES == ("android", "ios", "inconsistent")
    assert NUM_CLASSES == 3
    assert LABEL_TO_ID == {"android": 0, "ios": 1, "inconsistent": 2}
    assert ID_TO_LABEL[1] == "ios"


def test_validate_label_roundtrip_and_reject() -> None:
    assert validate_label("ios") == "ios"
    with pytest.raises(ValueError):
        validate_label("windows")


def test_validate_vote() -> None:
    for vote in ("android", "ios", "abstain"):
        assert validate_vote(vote) == vote
    with pytest.raises(ValueError):
        validate_vote("maybe")


def test_checkpoint_guard_accepts_matching() -> None:
    validate_checkpoint_classes({"classes": ["android", "ios", "inconsistent"]})


def test_checkpoint_guard_rejects_reordered() -> None:
    # 三个名字相同、顺序不同 => 必须拒绝（防止含义被悄悄调换）。
    with pytest.raises(ValueError):
        validate_checkpoint_classes({"classes": ["ios", "android", "inconsistent"]})


def test_checkpoint_guard_rejects_missing_or_wrong_type() -> None:
    with pytest.raises(ValueError):
        validate_checkpoint_classes({"classes": ["android", "ios"]})
    with pytest.raises(ValueError):
        validate_checkpoint_classes({})
    with pytest.raises(ValueError):
        validate_checkpoint_classes("not a dict")
