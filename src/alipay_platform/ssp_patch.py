"""SSP 的确定性选块 + 固定残差前端(纯 numpy/cv2,可在 C# 端逐位复现)。

修好开源 SSP 在我们图库上的两个 bug:
1. 它随机裁 64 块取纹理最平的 1 块 —— 干净截图上"最平块"是纯色常量,SRM 残差恒为 0、
   毫无信息;且随机 => 同图两次结果不同,风控闸门不能这样。
2. 上采样到 256 会把摩尔纹/噪声重采样掉。

改法(全确定性,无任何随机):
- 原分辨率、固定步长网格枚举 32x32 块(可选去掉顶部状态栏/底部导航条);
- 每块算纹理平坦度 L_div(SSP 原公式,四方向邻差和)+ SRM 残差能量;
- **跳过常量/近常量块**(残差能量 < eps),在剩下"平但有噪声底"的块里取 L_div 最小的 K 个;
- 若全是常量(纯干净截图)=> residual_null,SSP 弃权,交给字形/规则(这是正确行为,不是漏检)。

残差前端:SRM(3 个固定高通,抓传感器噪声/摩尔纹)+ NPR(2x2 邻差,抓生成器上采样指纹)。
Bayar(可学)那一路随骨干一起训,不在本模块。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

PATCH: int = 32
STRIDE: int = 16
TOPK: int = 8
CONST_EPS: float = 0.5     # SRM 残差标准差低于它算常量块(真实干净截图平坦区≈0)

# 3 个经典 SRM 固定高通核(与 SSP srm_conv 同族)
_SRM = [
    np.array([[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2],
              [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]], np.float32) / 12.0,
    np.array([[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0],
              [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]], np.float32) / 4.0,
    np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -2, 1, 0],
              [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], np.float32) / 2.0,
]


@dataclass(frozen=True)
class PatchSelection:
    coords: tuple[tuple[int, int], ...]   # 选中块左上角 (y, x)
    residual_null: bool                   # True=全是常量块,SSP 弃权
    n_candidates: int                     # 非常量候选块数


def srm_residual(gray: np.ndarray) -> np.ndarray:
    """灰度图 -> 3 通道 SRM 高通残差(HxWx3)。"""
    g = gray.astype(np.float32)
    return np.stack([cv2.filter2D(g, -1, k, borderType=cv2.BORDER_REFLECT) for k in _SRM], axis=2)


def npr_residual(gray: np.ndarray) -> np.ndarray:
    """NPR = x - up(down(x,1/2),2),最近邻,参数free,抓上采样栅格指纹。"""
    g = gray.astype(np.float32)
    h, w = g.shape
    down = cv2.resize(g, (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_NEAREST)
    up = cv2.resize(down, (w, h), interpolation=cv2.INTER_NEAREST)
    return g - up


def _l_div(patch: np.ndarray) -> float:
    """SSP 纹理平坦度:四方向邻差绝对值之和(越小越平)。patch=HxWxC。"""
    p = patch.astype(np.float32)
    d = (np.abs(p[:, :-1] - p[:, 1:]).sum() + np.abs(p[:-1, :] - p[1:, :]).sum()
         + np.abs(p[:-1, :-1] - p[1:, 1:]).sum() + np.abs(p[1:, :-1] - p[:-1, 1:]).sum())
    return float(d)


def select_patches(rgb: np.ndarray, *, patch: int = PATCH, stride: int = STRIDE,
                   topk: int = TOPK, const_eps: float = CONST_EPS,
                   top_margin: float = 0.0, bottom_margin: float = 0.0) -> PatchSelection:
    """确定性选出 topk 个"平但非常量"的块。top/bottom_margin 是要裁掉的上下比例(状态栏/导航)。"""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("需要 HxWx3 的 RGB 图")
    h, w = rgb.shape[:2]
    y0, y1 = int(h * top_margin), int(h * (1.0 - bottom_margin))
    gray = rgb[:, :, :3].mean(axis=2).astype(np.float32)
    cands: list[tuple[float, int, int]] = []   # (L_div, y, x),仅非常量块
    for y in range(y0, y1 - patch + 1, stride):
        for x in range(0, w - patch + 1, stride):
            blk = rgb[y:y + patch, x:x + patch]
            res = srm_residual(gray[y:y + patch, x:x + patch])
            if float(res.std()) < const_eps:      # 常量/近常量 -> 跳过(这是修的关键)
                continue
            cands.append((_l_div(blk), y, x))
    if not cands:
        return PatchSelection((), True, 0)
    cands.sort(key=lambda t: (t[0], t[1], t[2]))   # L_div 升序,再按坐标定序 => 完全确定
    chosen = tuple((y, x) for _, y, x in cands[:topk])
    return PatchSelection(chosen, False, len(cands))


def residual_stack(rgb: np.ndarray, sel: PatchSelection | None = None, *, patch: int = PATCH) -> np.ndarray:
    """把选中块的 [SRM3 | NPR1] 残差堆成 (K, patch, patch, 4),喂给后续骨干。residual_null 时返回空。"""
    if sel is None:
        sel = select_patches(rgb, patch=patch)
    if sel.residual_null:
        return np.empty((0, patch, patch, 4), np.float32)
    gray = rgb[:, :, :3].mean(axis=2).astype(np.float32)
    out = []
    for (y, x) in sel.coords:
        srm = srm_residual(gray[y:y + patch, x:x + patch])          # patch,patch,3
        npr = npr_residual(gray[y:y + patch, x:x + patch])[:, :, None]  # patch,patch,1
        out.append(np.concatenate([srm, npr], axis=2))
    return np.stack(out, axis=0).astype(np.float32)
