"""支付宝截图安卓 / 苹果平台分类器。

一个省 CPU 的分类器，只用两个小区域——状态栏和成功对勾——把支付宝转账成功截图判为
``android`` / ``ios`` / ``inconsistent``。它复用同项目检测器（``transfer_receipt_ai``）
的区域定位和批量推理脚手架，但把 PaddleOCR 和 1536px 全图检测器排除在推理主链路之外。

不依赖 torch / OpenCV 的模块（标签定义、元数据种子标注、几何裁剪、投票聚合、哈希）可以
单独导入和测试。依赖检测器的区域缓存放在 ``cache_regions`` 里，并做了惰性导入保护。
"""

from __future__ import annotations

__version__ = "0.1.0"

from .platform_labels import (  # noqa: F401
    ID_TO_LABEL,
    LABEL_TO_ID,
    NUM_CLASSES,
    PLATFORM_CLASSES,
    validate_checkpoint_classes,
    validate_label,
)
