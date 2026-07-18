"""零解码元数据标注函数的测试。"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from alipay_platform.metadata_seed import (
    IPHONE_RESOLUTIONS,
    exif_make_vote,
    icc_vote,
    metadata_votes,
    normalize_resolution,
    read_metadata_facts,
    resolution_vote,
    MetadataFacts,
)


def test_iphone_resolution_votes_ios_both_orientations() -> None:
    assert resolution_vote(1170, 2532).label == "ios"        # iPhone 13 竖屏
    assert resolution_vote(2532, 1170).label == "ios"        # 同一台，横屏
    assert (1170, 2532) in IPHONE_RESOLUTIONS


def test_common_android_resolution_abstains() -> None:
    # 1080x2340 / 1080x1920 / 1080x2400 是常见安卓分辨率，已刻意排除。
    for w, h in [(1080, 2340), (1080, 1920), (1080, 2400), (720, 1280)]:
        assert resolution_vote(w, h).label == "abstain"


def test_invalid_dimensions_abstain() -> None:
    assert resolution_vote(0, 100).label == "abstain"
    assert resolution_vote(-1, -1).label == "abstain"


def test_normalize_resolution_orientation_independent() -> None:
    assert normalize_resolution(1170, 2532) == normalize_resolution(2532, 1170) == (1170, 2532)


def test_icc_vote() -> None:
    assert icc_vote(b"blahblah Display P3 blah").label == "ios"
    assert icc_vote(b"sRGB IEC61966-2.1").label == "abstain"
    assert icc_vote(None).label == "abstain"
    assert icc_vote(b"").label == "abstain"


def test_exif_make_vote_born_digital_vs_photo() -> None:
    # 原生苹果截图（无相机字段）=> 弱苹果。
    assert exif_make_vote("Apple", has_capture_tags=False).label == "ios"
    # 照片（有相机字段）：Make 是相机，不代表截图系统。
    assert exif_make_vote("Apple", has_capture_tags=True).label == "abstain"
    assert exif_make_vote("samsung", has_capture_tags=False).label == "abstain"
    assert exif_make_vote(None, has_capture_tags=False).label == "abstain"


def test_read_metadata_facts_is_header_only_and_correct() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "shot.png"
        # 一张类似截图的 PNG：无 ICC，无 EXIF。
        Image.fromarray(np.zeros((2532, 1170, 3), dtype=np.uint8)).save(path)
        facts = read_metadata_facts(path)
        assert (facts.width, facts.height) == (1170, 2532)
        assert facts.icc_profile is None
        assert facts.make is None
        assert facts.has_capture_tags is False
        votes = metadata_votes(facts)
        # 分辨率标注函数投“苹果”，另外两个弃权。
        labels = sorted(v.label for v in votes)
        assert labels == ["abstain", "abstain", "ios"]


def test_metadata_votes_all_abstain_for_android_shape() -> None:
    facts = MetadataFacts(width=1080, height=2340, icc_profile=None, make=None, has_capture_tags=False)
    assert all(v.label == "abstain" for v in metadata_votes(facts))
