"""融合层（设备合并 + 欺诈评分）的测试。"""

from __future__ import annotations

from alipay_platform.fusion import (
    FraudSignals,
    device_prior_conflict,
    fraud_score,
    merge_device,
)


def test_merge_resolution_wins() -> None:
    r = merge_device("ios", cnn={"device": "android", "conf": 0.9})
    assert r["device"] == "ios" and r["source"] == "resolution"


def test_merge_abstain_uses_cnn() -> None:
    r = merge_device("abstain", cnn={"device": "android", "conf": 0.88})
    assert r["device"] == "android" and r["source"] == "cnn" and r["confidence"] == 0.88


def test_merge_abstain_without_cnn_is_unknown() -> None:
    assert merge_device("abstain", cnn=None)["device"] == "unknown"


def test_fraud_pass_when_no_signals() -> None:
    assert fraud_score(FraudSignals()).verdict == "pass"


def test_fraud_single_no_checkmark_goes_to_review() -> None:
    r = fraud_score(FraudSignals(no_checkmark=True))
    assert r.verdict == "review" and "no_checkmark" in r.reasons


def test_fraud_two_signals_reject() -> None:
    r = fraud_score(FraudSignals(no_checkmark=True, photo_of_screen=True))
    assert r.verdict == "reject"


def test_fraud_exif_mismatch_plus_conflict_reject() -> None:
    r = fraud_score(FraudSignals(exif_device_mismatch=True, device_prior_conflict=True))
    assert r.score >= 0.6 and r.verdict == "reject"


def test_device_prior_conflict() -> None:
    assert device_prior_conflict("ios", "android", 0.9) is True
    assert device_prior_conflict("ios", "ios", 0.9) is False
    assert device_prior_conflict("ios", "android", 0.5) is False   # 置信不够不算冲突
    assert device_prior_conflict("abstain", "android", 0.9) is False
