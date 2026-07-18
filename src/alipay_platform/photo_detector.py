"""多信号“翻拍图”（photo-of-screen）检测——大规模自举前的 P0 阻塞门。

为什么重要：分辨率自举“iPhone 分辨率=iOS”只对“直接截图”成立。用 iPhone 翻拍安卓屏的
照片若恰好被裁到 iPhone 分辨率，会被误标成 iOS，污染免费标签。

关键洞察：分辨率自举本来就只在“精确手机分辨率”上触发；原始相机照片是奇怪的大尺寸
（如 3024x4032），本就落到 abstain，不会拿到错误标签。真正的毒样本 = “带相机信号
却又恰好是手机分辨率”的图。所以检测优先用最可靠的元数据信号：

信号（可靠度从高到低）：
- EXIF 拍摄字段（DateTimeOriginal/FNumber/ISO/LensModel/相机 Make）——存在即翻拍（会被抹掉）。
- 短边尺寸：手机截图短边 ≤1440；相机照片短边通常 ≥2000 -> 短边>1500 基本不是截图。
- 长宽比：手机竖屏截图 h/w≈1.8~2.4。
- 平坦区高频噪声/摩尔纹：EXIF 被抹且分辨率又恰好命中时的兜底像素信号（弱，需在真实翻拍上标定）。

局限：本机只有已知截图、无真实翻拍样本，只能验证“特异度”（不误报截图）；噪声阈值需在
服务器真实翻拍子集上标定。真实截图实测：短边≤1440、无 EXIF、平坦区噪声≤0.13。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SHORT_SIDE_MAX: int = 1500     # 手机截图短边上限（iPhone≤1320, 安卓≤1440），超过基本是相机照片
NOISE_THRESHOLD: float = 1.5   # 平坦区噪声（真实截图≤0.13）；保守取值保特异度，真实翻拍需再标定
ASPECT_MIN: float = 1.60
ASPECT_MAX: float = 2.60


@dataclass(frozen=True)
class PhotoVerdict:
    is_photo: bool
    confidence: float
    flat_noise: float
    aspect: float
    short_side: int
    reasons: tuple[str, ...]


def flat_region_noise(rgb: np.ndarray, grid_h: int = 16, grid_w: int = 10) -> float:
    """“最平坦区域”的逐像素高频噪声（原分辨率上算，绝不先缩放——缩放会低通滤掉噪声）。"""
    g = rgb.mean(axis=2)
    h, w = g.shape
    ph, pw = max(4, h // grid_h), max(4, w // grid_w)
    noises: list[float] = []
    for i in range(0, h - ph + 1, ph):
        for j in range(0, w - pw + 1, pw):
            patch = g[i : i + ph, j : j + pw]
            noises.append(0.5 * float(np.abs(np.diff(patch, axis=0)).mean() + np.abs(np.diff(patch, axis=1)).mean()))
    return float(np.percentile(noises, 5)) if noises else 0.0


def photo_verdict_from_meta(width: int, height: int, *, has_capture_tags: bool = False) -> PhotoVerdict:
    """只用元数据的快速判定（零解码，适合先在全池跑一遍）。"""
    short = min(width, height)
    aspect = (max(width, height) / short) if short else 0.0
    if has_capture_tags:
        return PhotoVerdict(True, 0.95, -1.0, round(aspect, 3), short, ("EXIF 含相机拍摄字段",))
    if short > SHORT_SIDE_MAX:
        return PhotoVerdict(True, 0.9, -1.0, round(aspect, 3), short, (f"短边{short}>{SHORT_SIDE_MAX}，相机尺寸",))
    reasons = []
    score = 0.0
    if not (ASPECT_MIN <= aspect <= ASPECT_MAX):
        score += 0.35
        reasons.append(f"长宽比({aspect:.2f})非手机竖屏")
    return PhotoVerdict(score >= 0.5, round(0.5 + score, 3), -1.0, round(aspect, 3), short, tuple(reasons) or ("元数据像截图",))


def photo_verdict(rgb: np.ndarray, *, has_capture_tags: bool = False) -> PhotoVerdict:
    """元数据 + 像素综合判定（对分辨率命中的样本兜底查噪声/摩尔纹）。"""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("需要 HxWx3 的 RGB 图")
    h, w = rgb.shape[:2]
    meta = photo_verdict_from_meta(w, h, has_capture_tags=has_capture_tags)
    if meta.is_photo:
        return meta
    noise = flat_region_noise(rgb)
    reasons = list(meta.reasons)
    score = 0.0 if (ASPECT_MIN <= meta.aspect <= ASPECT_MAX) else 0.35
    if noise > NOISE_THRESHOLD:
        score += min(1.0, (noise - NOISE_THRESHOLD) / NOISE_THRESHOLD)
        reasons = [f"平坦区噪声偏高({noise:.1f}>{NOISE_THRESHOLD})"]
    is_photo = score >= 0.5
    return PhotoVerdict(is_photo, round(min(0.95, 0.5 + score / 2), 3), round(noise, 2), meta.aspect, meta.short_side, tuple(reasons))
