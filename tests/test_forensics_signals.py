"""取证信号提取 + 融合行为的测试。"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")

from alipay_platform.forensics_signals import extract_forensic_signals
from alipay_platform.fusion import FraudSignals, fraud_score


def _screen_grid(n: int = 480, freq: float = 0.12, amp: float = 10.0) -> np.ndarray:
    """可靠触发摩尔纹的合成屏栅色度图(同 test_moire)。"""
    yy, xx = np.mgrid[0:n, 0:n]
    patt = amp * np.cos(2 * np.pi * freq * (xx + yy))
    rng = np.random.default_rng(0)
    base = 128 + rng.normal(0, 2, (n, n))
    rgb = np.stack([base + patt, base, base - patt], axis=2)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _flat(n: int = 480) -> np.ndarray:
    return np.full((n, n, 3), 128, np.uint8)


def test_phone_screenshot_metadata_is_not_photo():
    sig = extract_forensic_signals(1080, 2400, has_capture_tags=False)
    assert sig["modality"] == "screenshot"
    assert not sig["photo_of_screen"]


def test_capture_tags_metadata_is_photo():
    sig = extract_forensic_signals(3000, 4000, has_capture_tags=True)
    assert sig["modality"] == "camera"
    assert sig["photo_of_screen"]
    # 只给元数据 -> 摩尔纹信号无从判断
    assert not sig["screen_recapture_moire"]


def test_moire_grid_confirms_recapture():
    sig = extract_forensic_signals(480, 480, has_capture_tags=True, rgb=_screen_grid())
    assert sig["photo_of_screen"]
    assert sig["screen_recapture_moire"]
    assert sig["moire_score"] is not None and sig["moire_score"] >= 50


def test_flat_camera_photo_no_moire():
    # 大尺寸平坦图 = 相机尺寸但无摩尔纹(类比随手拍的杂图)
    sig = extract_forensic_signals(480, 480, has_capture_tags=True, rgb=_flat())
    assert sig["photo_of_screen"]
    assert not sig["screen_recapture_moire"]


def test_fusion_grades_recapture():
    # 弱信号单条 -> pass
    assert fraud_score(FraudSignals(photo_of_screen=True)).verdict == "pass"
    # 摩尔纹确认单条 -> review
    assert fraud_score(FraudSignals(screen_recapture_moire=True)).verdict == "review"
    # 两条一起 -> reject
    both = fraud_score(FraudSignals(photo_of_screen=True, screen_recapture_moire=True))
    assert both.verdict == "reject"
    assert "screen_recapture_moire" in both.reasons
