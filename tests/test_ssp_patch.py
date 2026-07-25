"""SSP 确定性选块 + 残差前端的自足测试。"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")

from alipay_platform.ssp_patch import (
    npr_residual,
    residual_stack,
    select_patches,
    srm_residual,
)


def _const(v: int = 128, n: int = 128) -> np.ndarray:
    return np.full((n, n, 3), v, np.uint8)


def _flat_noisy(n: int = 128, std: float = 3.0) -> np.ndarray:
    rng = np.random.default_rng(0)
    g = np.clip(128 + rng.normal(0, std, (n, n)), 0, 255)
    return np.stack([g, g, g], axis=2).astype(np.uint8)


def _flat_plus_texture(n: int = 128) -> np.ndarray:
    img = _flat_noisy(n)
    # 右下角塞一块高纹理棋盘格(L_div 大,不该被优先选)
    chk = (np.indices((n // 2, n // 2)).sum(axis=0) % 2 * 255).astype(np.uint8)
    img[n // 2 :, n // 2 :] = chk[:, :, None]
    return img


def test_constant_is_residual_null():
    sel = select_patches(_const())
    assert sel.residual_null
    assert sel.coords == ()


def test_srm_residual_of_constant_is_zero():
    assert float(np.abs(srm_residual(_const()[:, :, 0].astype(np.float32))).max()) < 1e-3


def test_flat_noisy_selects_patches():
    sel = select_patches(_flat_noisy())
    assert not sel.residual_null
    assert len(sel.coords) > 0
    assert sel.n_candidates > 0


def test_prefers_flat_over_textured():
    n = 160
    sel = select_patches(_flat_plus_texture(n), topk=6)
    # 选中块不应落在右下棋盘格区(那里 L_div 最大)
    assert all(not (y >= n // 2 and x >= n // 2) for (y, x) in sel.coords)


def test_deterministic():
    img = _flat_noisy()
    assert select_patches(img).coords == select_patches(img).coords


def test_residual_stack_shape():
    img = _flat_noisy()
    st = residual_stack(img, patch=32)
    assert st.ndim == 4 and st.shape[1:] == (32, 32, 4)


def test_residual_stack_empty_on_constant():
    assert residual_stack(_const()).shape[0] == 0


def test_npr_zero_on_constant():
    assert float(np.abs(npr_residual(_const()[:, :, 0].astype(np.float32))).max()) < 1e-3


def test_rejects_non_rgb():
    with pytest.raises(ValueError):
        select_patches(np.zeros((10, 10), np.uint8))
