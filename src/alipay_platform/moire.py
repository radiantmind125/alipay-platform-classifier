"""翻拍(photo-of-screen)摩尔纹频谱检测——训练无关,纯 numpy/cv2,约 0.3 秒/张。

原理:翻拍 = 相机拍屏幕,相机传感器采样栅格与屏幕像素/子像素栅格拍频,在 2D 频谱的
**中频带**留下离轴的周期峰(色度尤强)。干净截图是渲染图、频谱几乎空;真实场景照片
(如风景)在低频有平滑梯度峰但**中频没有周期峰**。所以"中频带最大峰显著度"能把
翻拍 从 截图 和 真拍风景 里分出来。

关键工程点(踩过的坑):
- **绝不先缩放**。翻拍的屏栅摩尔纹是高频,缩到 1024 会被低通滤没(实测全归零)。在原
  分辨率上取中心方块做 FFT。
- **只看中频带 [0.10,0.90]**。低频(<0.10)是风景的天空/渐变梯度,真拍风景在那里也有
  强峰,会误报;中频才是屏栅拍频。
- **掩掉坐标轴十字 + JPEG 8x8 格点**(1/8 整数倍),否则任何 JPEG 都在轴上出峰。
- 实数图的 FFT 幅度本就点对称,不用再手动对称化(早期版本手动对称 min 因偶数尺寸
  差 1 像素把峰全杀了)。

定位:photo_detector 判 相机 vs 截图;本模块在**相机子集**里进一步判 翻拍 vs 真拍,
所以流水线里手机截图根本不会走到这里。峰在=高精度翻拍证据(应触发高风险/复核),
峰不在 != 没翻拍(分辨率对齐拍摄/去摩尔纹 App 会没峰),故只作复核触发,不做硬拒。

标定(本机小样本:10 张翻拍 chroma 峰 52~78;日落真拍 35;2 张平板截图 22/49):
默认阈值 50。样本很小,**上线前必须在服务器真实翻拍子集上复标**。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

SCORE_THRESHOLD: float = 50.0    # 中频带色度最大峰显著度(sigma);50 时本机 10/10 翻拍命中、日落/平板不中
CROP_MAX: int = 1600             # 原分辨率上取的中心方块上限(限 FFT 开销,不缩放)
_BAND_LO, _BAND_HI = 0.10, 0.90  # 归一化半径带(0=DC,1=短边 Nyquist)
_PEAK_K: float = 6.0             # 局部极大值阈值 = 带内均值 + k*标准差


@dataclass(frozen=True)
class MoireVerdict:
    is_recapture: bool
    score: float          # 中频带色度最大峰显著度
    n_peaks: int          # 中频带色度显著局部峰个数
    luma_score: float
    reasons: tuple[str, ...]


def _center_square(rgb: np.ndarray, cap: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    s = min(cap, h, w)
    return rgb[(h - s) // 2 : (h - s) // 2 + s, (w - s) // 2 : (w - s) // 2 + s]


def _band_peaks(chan: np.ndarray) -> tuple[float, int]:
    """一个通道:中频带、离轴、非 JPEG 格点的白化谱峰。返回 (最大峰显著度, 显著局部峰数)。"""
    c = chan.astype(np.float32)
    resid = c - cv2.GaussianBlur(c, (0, 0), 2.5)                 # 高通残差
    n = resid.shape[0]
    win = np.outer(np.hanning(n), np.hanning(n)).astype(np.float32)
    mag = np.abs(np.fft.fftshift(np.fft.fft2(resid * win)))       # 线性幅度(峰比 log 更尖)
    cy = cx = n // 2
    yy, xx = np.ogrid[:n, :n]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rint = r.astype(np.int32)
    prof = np.bincount(rint.ravel(), mag.ravel()) / np.maximum(np.bincount(rint.ravel()), 1)
    white = mag - prof[rint]                                      # 径向白化,去 1/f 斜坡
    rn = r / (n / 2.0)
    fx, fy = (xx - cx) / (n / 2.0), (yy - cy) / (n / 2.0)
    axis = (np.abs(yy - cy) < 2) | (np.abs(xx - cx) < 2)
    near8 = lambda f: np.abs(np.abs(f) * 4 - np.round(np.abs(f) * 4)) < 0.03   # 归一化频率 f≈k/4(=JPEG 8像素块周期的谐波),掩掉
    band = (rn > _BAND_LO) & (rn < _BAND_HI) & (~axis) & (~(near8(fx) & near8(fy)))
    vals = white[band]
    if vals.size == 0:
        return 0.0, 0
    mu, sd = float(vals.mean()), float(vals.std() + 1e-6)
    score = (float(vals.max()) - mu) / sd
    local_max = white >= cv2.dilate(white, np.ones((7, 7), np.float32))
    n_peaks = int((local_max & band & (white > mu + _PEAK_K * sd)).sum())
    return score, n_peaks


def moire_verdict(rgb: np.ndarray, *, threshold: float = SCORE_THRESHOLD, crop: int = CROP_MAX) -> MoireVerdict:
    """RGB(HxWx3,uint8)-> 翻拍判定。在**相机子集**上用(手机截图应先被 photo_detector 路由走)。"""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("需要 HxWx3 的 RGB 图")
    sq = _center_square(rgb, crop)
    ycc = cv2.cvtColor(sq.astype(np.uint8), cv2.COLOR_RGB2YCrCb).astype(np.float32)
    y_s, _ = _band_peaks(ycc[:, :, 0])
    cr_s, cr_n = _band_peaks(ycc[:, :, 1])
    cb_s, cb_n = _band_peaks(ycc[:, :, 2])
    score = max(cr_s, cb_s)
    n_peaks = cr_n if cr_s >= cb_s else cb_n
    is_rec = score >= threshold
    reasons = ((f"中频色度周期峰 显著度{score:.0f}sigma 峰{n_peaks}",) if is_rec
               else (f"无明显中频摩尔纹峰(显著度{score:.0f}<{threshold:.0f})",))
    return MoireVerdict(is_rec, round(score, 1), int(n_peaks), round(y_s, 1), reasons)
