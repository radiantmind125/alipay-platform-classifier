"""翻拍(photo-of-screen)模拟器:把干净截图合成为"相机拍屏"的样子,用来:
1) 给 SSP 的 head_screen(翻拍检测)造无限带标注正样本,含"翻拍假页"洗白类(输入换成假页即可);
2) 压测摩尔纹检测器的召回(分辨率对齐/去摩尔纹时会没峰)。

核心物理:屏幕每个像素由 R|G|B 三条子像素条纹发光(横向 x3),相机传感器采样栅格与这个
子像素条纹**拍频**产生**色度摩尔纹**——正是 moire 模块要抓的东西。所以关键步骤是:
子像素渲染 -> 轻微旋转 -> 按相机分辨率重采样(混叠出摩尔纹),再叠 镜头模糊/暗角/眩光/
传感器噪声/JPEG。给定 seed 完全确定。

局限:合成摩尔纹未必完全等于真实相机-屏幕的物理拍频;用途是"够像到能训 + 能压测",
真实阈值仍以服务器真实翻拍子集为准。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class RecaptureParams:
    theta_deg: float          # 屏幕相对相机的小旋转(制造拍频方向)
    cam_ratio: float          # 相机每屏幕像素采样率(!=整数才出摩尔纹)
    keystone: float           # 透视/梯形形变强度:屏幕微倾 -> 屏栅间距逐处变 -> 摩尔纹频率扫动 -> 峰变宽变真实
    grid_gap: float           # 子像素间黑缝强度 0~1(屏栅)
    psf_sub: float            # 镜头 PSF(子像素空间,重采样前),控制摩尔纹强弱/真实度
    blur_sigma: float         # 采样后轻微模糊(传感器抗锯齿)
    glare: float              # 屏幕反光高光强度 0~1
    vignette: float           # 暗角强度 0~1
    noise_sigma: float        # 传感器噪声
    jpeg_q: int               # 相机 JPEG 质量


def _rand_params(rng: np.random.Generator) -> RecaptureParams:
    # psf_sub 大 -> 子像素条纹被镜头糊到一起 -> 摩尔纹弱(更像真实、含分辨率对齐的难样本);
    # 小 -> 摩尔纹强。取一个跨度,让合成翻拍的摩尔纹分覆盖真实的 ~40~100 区间。
    return RecaptureParams(
        theta_deg=float(rng.uniform(0.6, 3.2) * rng.choice([-1, 1])),
        cam_ratio=float(rng.uniform(0.80, 1.30)),
        keystone=float(rng.uniform(0.02, 0.07)),
        grid_gap=float(rng.uniform(0.10, 0.35)),
        psf_sub=float(rng.uniform(0.6, 1.3)),
        blur_sigma=float(rng.uniform(0.3, 0.7)),
        glare=float(rng.uniform(0.0, 0.35)),
        vignette=float(rng.uniform(0.1, 0.35)),
        noise_sigma=float(rng.uniform(1.5, 5.0)),
        jpeg_q=int(rng.integers(72, 95)),
    )


def _subpixel(S: np.ndarray, grid_gap: float) -> np.ndarray:
    """RGB 条纹子像素渲染:H,W,3 -> H,3W,3(每列拆成 R|G|B 三条,亮度x3 抵消降采样)。"""
    h, w, _ = S.shape
    sub = np.zeros((h, w * 3, 3), np.float32)
    sub[:, 0::3, 0] = S[:, :, 0] * 3.0
    sub[:, 1::3, 1] = S[:, :, 1] * 3.0
    sub[:, 2::3, 2] = S[:, :, 2] * 3.0
    if grid_gap > 0:                              # 每个像素边界压暗一条,强化屏栅频率
        sub[:, 0::3, :] *= (1.0 - grid_gap)
    return sub


def _perspective(sub: np.ndarray, strength: float, rng: np.random.Generator) -> np.ndarray:
    """随机透视(梯形)形变:屏幕相对相机微倾,屏栅间距逐处不同 -> 摩尔纹频率空间扫动 -> 谱峰变宽。"""
    if strength <= 0:
        return sub
    h, w3 = sub.shape[:2]
    src = np.float32([[0, 0], [w3, 0], [w3, h], [0, h]])
    jit = np.stack([rng.uniform(-strength, strength, 4) * w3,
                    rng.uniform(-strength, strength, 4) * h], axis=1).astype(np.float32)
    m = cv2.getPerspectiveTransform(src, src + jit)
    return cv2.warpPerspective(sub, m, (w3, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def _vignette(img: np.ndarray, strength: float) -> np.ndarray:
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    mask = 1.0 - strength * np.clip(r, 0, 1) ** 2
    return img * mask[:, :, None]


def _glare(img: np.ndarray, strength: float, rng: np.random.Generator) -> np.ndarray:
    if strength <= 0:
        return img
    h, w = img.shape[:2]
    cyx = (rng.uniform(0.15, 0.85, 2) * [h, w])
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    g = np.exp(-(((yy - cyx[0]) ** 2 + (xx - cyx[1]) ** 2) / (2 * (0.22 * min(h, w)) ** 2)))
    return img + strength * 255.0 * g[:, :, None]


def _jpeg(img: np.ndarray, q: int) -> np.ndarray:
    bgr = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(q)])
    if not ok:
        return img
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32)


def simulate_recapture(rgb: np.ndarray, *, seed: int = 0,
                       params: RecaptureParams | None = None) -> np.ndarray:
    """干净截图 RGB(HxWx3 uint8)-> 合成翻拍 RGB(uint8)。给定 seed 确定。"""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("需要 HxWx3 的 RGB 图")
    rng = np.random.default_rng(seed)
    p = params or _rand_params(rng)
    S = rgb.astype(np.float32)
    h, w, _ = S.shape

    sub = _subpixel(S, p.grid_gap)                                  # H, 3W, 3
    M = cv2.getRotationMatrix2D((w * 1.5, h / 2.0), p.theta_deg, 1.0)
    sub = cv2.warpAffine(sub, M, (w * 3, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    sub = _perspective(sub, p.keystone, rng)                        # 微倾 -> 摩尔纹频率扫动 -> 峰变宽变真实
    if p.psf_sub > 0:
        sub = cv2.GaussianBlur(sub, (0, 0), p.psf_sub)             # 镜头 PSF 在传感器采样"之前"作用于屏光
    hc, wc = max(8, int(h * p.cam_ratio)), max(8, int(w * p.cam_ratio))
    cam = cv2.resize(sub, (wc, hc), interpolation=cv2.INTER_AREA)   # 混叠 -> 摩尔纹(现在强度真实)

    if p.blur_sigma > 0:
        cam = cv2.GaussianBlur(cam, (0, 0), p.blur_sigma)          # 传感器抗锯齿
    cam = _glare(cam, p.glare, rng)
    cam = _vignette(cam, p.vignette)
    cam += rng.normal(0, p.noise_sigma, cam.shape)                 # 传感器噪声
    cam = _jpeg(cam, p.jpeg_q)                                     # 相机 JPEG
    return np.clip(cam, 0, 255).astype(np.uint8)
