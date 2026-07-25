"""把一张图变成欺诈融合要的取证信号(连接 photo_detector 路由 + moire 翻拍确认)。

两级 翻拍 信号,精度递增:
- photo_of_screen(弱,元数据):路由判为相机照片。支付凭证本该发截图,发相机照片本身就可疑
  (可能是翻拍,也可能是随手拍的杂图),但只是弱信号。
- screen_recapture_moire(硬,像素):中频色度摩尔纹周期峰确认"确系拍屏"。把翻拍从随手拍的
  真实照片里分出来,是高精度证据。

给欺诈融合:photo_of_screen=0.25(单条 pass),screen_recapture_moire=0.5(单条 review),
两条一起=0.75(reject)。翻拍是否算有效凭证是经理业务政策,故默认不硬拒。
"""

from __future__ import annotations

import numpy as np

from .moire import SCORE_THRESHOLD, moire_verdict
from .photo_detector import photo_verdict, photo_verdict_from_meta

# 只认光学拍摄参数,别用 DateTimeOrig(安卓截图也盖时间戳,会误判)。供调用方读 EXIF 用。
OPTICAL_EXIF_TAGS = (33434, 33437, 34855, 37386)  # Exposure/FNumber/ISO/FocalLength


def extract_forensic_signals(width: int, height: int, *, has_capture_tags: bool = False,
                             rgb: np.ndarray | None = None,
                             moire_threshold: float = SCORE_THRESHOLD) -> dict:
    """返回 {modality, photo_of_screen, screen_recapture_moire, moire_score, reasons}。

    只有元数据时(rgb=None)只能出 photo_of_screen;给了 rgb 才能跑摩尔纹确认 翻拍。
    """
    if rgb is not None:
        pv = photo_verdict(rgb, has_capture_tags=has_capture_tags)
    else:
        pv = photo_verdict_from_meta(width, height, has_capture_tags=has_capture_tags)
    is_camera = pv.is_photo

    screen_recapture_moire = False
    moire_score: float | None = None
    if is_camera and rgb is not None:
        mv = moire_verdict(rgb, threshold=moire_threshold)
        screen_recapture_moire = mv.is_recapture
        moire_score = mv.score

    return {
        "modality": "camera" if is_camera else "screenshot",
        "photo_of_screen": bool(is_camera),
        "screen_recapture_moire": bool(screen_recapture_moire),
        "moire_score": moire_score,
        "reasons": list(pv.reasons),
    }
