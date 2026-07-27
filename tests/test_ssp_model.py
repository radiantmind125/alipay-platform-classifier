"""SSP-tiny 模型的自足测试:前向形状、多头、确定性、参数量小。"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from ssp_model import build_ssp, count_params


def test_forward_shape_single_head():
    m = build_ssp(head_names=("screen",), cin=4).eval()
    x = torch.randn(3, 8, 4, 32, 32)          # [B, K, C, 32, 32]
    with torch.no_grad():
        y = m(x)
    assert y.shape == (3, 1)


def test_forward_shape_multi_head():
    m = build_ssp(head_names=("screen", "gen", "sensor"), cin=4).eval()
    with torch.no_grad():
        y = m(torch.randn(2, 4, 4, 32, 32))
    assert y.shape == (2, 3)


def test_deterministic_eval():
    m = build_ssp().eval()
    x = torch.randn(2, 8, 4, 32, 32)
    with torch.no_grad():
        a, b = m(x), m(x)
    assert torch.equal(a, b)


def test_is_tiny():
    assert count_params(build_ssp(head_names=("screen", "gen"))) < 200_000
