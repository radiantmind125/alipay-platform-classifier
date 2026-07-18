"""状态栏条的便宜特征（只用 numpy + PIL），用于安卓/苹果分类。

设计依据（真实数据肉眼确认）：安卓状态栏“吵”——网速文字、双卡信号、运营商名、电量百分比、
一排应用图标，内容多且铺得广；iOS 状态栏“干净”——左侧时钟+定位箭头、中间灵动岛/刘海、
右侧稀疏的信号+电池胶囊。

关键：所有条都先缩放到固定尺寸，因此特征与分辨率无关——绝不泄漏“分辨率”这个自举标签。
内容检测用“与背景色的差异”，因此蓝底白字、白底黑字都适用。
"""

from __future__ import annotations

from typing import Final

import numpy as np
from PIL import Image

_W: Final[int] = 480
_H: Final[int] = 48

FEATURE_NAMES: Final[tuple[str, ...]] = (
    "content_frac",
    *(f"bin_{i:02d}" for i in range(12)),
    "dark_center",     # 中央近黑区域（灵动岛/刘海）——iOS 信号
    "left_third",
    "mid_third",
    "right_third",
    "run_count",       # 内容列的分段数（图标个数近似）——安卓更多
    "spread",          # 内容水平分布的标准差
    "far_left",        # 最左（时钟区）
)


def extract_features(rgb: np.ndarray) -> np.ndarray:
    """从一条状态栏 RGB 图（HxWx3）提取固定长度特征向量。"""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("需要 HxWx3 的 RGB 图")
    a = np.asarray(Image.fromarray(rgb).resize((_W, _H)), dtype=np.float32)
    bg = np.median(a.reshape(-1, 3), axis=0)          # 背景色（状态栏底色）
    dist = np.abs(a - bg).sum(axis=2)                  # 与背景的 L1 距离
    content = dist > 110                                # 明显偏离背景的即为图标/文字
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    dark = (r < 55) & (g < 55) & (b < 55)

    feats: list[float] = [float(content.mean())]
    for i in range(12):
        feats.append(float(content[:, i * 40 : (i + 1) * 40].mean()))
    feats.append(float(dark[:, 192:288].mean()))       # 中央近黑
    feats.append(float(content[:, :160].mean()))        # 左三分之一
    feats.append(float(content[:, 160:320].mean()))     # 中三分之一
    feats.append(float(content[:, 320:].mean()))        # 右三分之一
    colprof = content.mean(axis=0) > 0.06
    runs = int(np.sum(colprof[1:] & ~colprof[:-1])) + int(colprof[0])
    feats.append(runs / 20.0)
    xs = np.where(content.any(axis=0))[0]
    feats.append(float(xs.std() / _W) if xs.size else 0.0)
    feats.append(float(content[:, :48].mean()))
    return np.asarray(feats, dtype=np.float32)


def load_features(path: str) -> np.ndarray:
    """从磁盘上的状态栏 PNG 提取特征。"""
    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB"))
    return extract_features(rgb)
