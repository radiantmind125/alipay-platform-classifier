"""engine B 篡改合成的自足测试:能定位大金额、每种改法都产出非空掩码且改动了像素。"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")
from PIL import Image, ImageDraw

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from engine_b_tamper import edit_copymove, edit_minus, edit_refont, locate_amount


def _fake_bill(neg: bool = True) -> Image.Image:
    """合成一张"白底账单详情":上方一个大金额 + 一些小字。"""
    im = Image.new("RGB", (480, 900), (250, 250, 250))
    d = ImageDraw.Draw(im)
    amt = ("-" if neg else "") + "100.00"
    try:
        from engine_b_tamper import _load_font
        d.text((110, 150), amt, font=_load_font(90), fill=(30, 30, 30))
        d.text((60, 400), "付款方式 余额宝", font=_load_font(26), fill=(90, 90, 90))
    except Exception:
        d.text((110, 150), amt, fill=(30, 30, 30))
    return im


def test_locate_amount_finds_big_number():
    loc = locate_amount(np.asarray(_fake_bill()))
    assert loc is not None
    x0, y0, x1, y1, glyphs = loc
    assert y0 < 400 and (x1 - x0) > 60 and len(glyphs) >= 2   # 定到上方的大金额


def _run(edit_fn, neg=True):
    im = _fake_bill(neg)
    before = np.asarray(im).copy()
    loc = locate_amount(np.asarray(im))
    assert loc is not None
    mask = edit_fn(im, loc[:4], loc[4], __import__("random").Random(0))
    after = np.asarray(im)
    return before, after, mask


def test_refont_changes_and_masks():
    before, after, mask = _run(edit_refont)
    assert mask.sum() > 0 and not np.array_equal(before, after)


def test_minus_insert_on_positive_masks():
    before, after, mask = _run(edit_minus, neg=False)     # 正数 -> 插负号
    assert mask.sum() > 0 and not np.array_equal(before, after)


def test_copymove_masks():
    before, after, mask = _run(edit_copymove)
    assert mask.sum() > 0


def test_mask_shape_matches_image():
    im = _fake_bill()
    loc = locate_amount(np.asarray(im))
    mask = edit_minus(im, loc[:4], loc[4], __import__("random").Random(1))
    assert mask.shape == np.asarray(im).shape[:2]
