"""弃权式投票聚合策略的测试。"""

from __future__ import annotations

import pytest

from alipay_platform.labeling import (
    ACCEPT_ANDROID,
    ACCEPT_IOS,
    REVIEW_CONFLICT,
    REVIEW_INSUFFICIENT,
    aggregate,
)
from alipay_platform.metadata_seed import Vote


def _v(label: str, conf: float = 0.9, reason: str = "t") -> Vote:
    return Vote(label, conf, reason)


def test_two_agreeing_votes_accept() -> None:
    d = aggregate([_v("ios"), _v("ios"), _v("abstain")])
    assert d.outcome == ACCEPT_IOS and d.label == "ios" and d.accepted


def test_two_agreeing_android_votes_accept() -> None:
    d = aggregate([_v("android"), _v("android")])
    assert d.outcome == ACCEPT_ANDROID and d.label == "android"


def test_conflict_routes_to_review_as_tamper_candidate() -> None:
    d = aggregate([_v("ios"), _v("ios"), _v("android")])
    assert d.outcome == REVIEW_CONFLICT and d.label is None and d.needs_review


def test_single_vote_is_insufficient() -> None:
    d = aggregate([_v("ios"), _v("abstain"), _v("abstain")])
    assert d.outcome == REVIEW_INSUFFICIENT and d.label is None


def test_all_abstain_is_insufficient() -> None:
    d = aggregate([_v("abstain"), _v("abstain")])
    assert d.outcome == REVIEW_INSUFFICIENT


def test_one_confident_lf_cannot_manufacture_a_label() -> None:
    # 置信度很高，但只有一个独立标注函数 => 仍然转人工。
    d = aggregate([_v("ios", conf=100.0)])
    assert d.needs_review


def test_min_agree_is_configurable() -> None:
    d = aggregate([_v("ios")], min_agree=1)
    assert d.outcome == ACCEPT_IOS


def test_reasons_audit_trail_excludes_abstains() -> None:
    d = aggregate([_v("ios", reason="res"), _v("ios", reason="p3"), _v("abstain", reason="x")])
    assert any("res" in r for r in d.reasons)
    assert all("x" not in r for r in d.reasons)


def test_bad_min_agree_raises() -> None:
    with pytest.raises(ValueError):
        aggregate([_v("ios")], min_agree=0)
