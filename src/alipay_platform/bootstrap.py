"""基于分辨率的高精度自举标注（仅用于弱标注/自举，绝不进模型特征）。

在本图片池（`raw_images`，全部为直接截图、无翻拍）上，已用肉眼核对验证：
- iPhone 精确分辨率 -> iOS（联系表样本 6/6 正确）
- 短边 ∈ {720,1080,1440}（iPhone 从不使用的安卓面板宽度）-> 安卓（样本 6/6 正确）
- 其余（1200~1290 等非常规宽度，多为裁剪/缩放）-> 弃权，交给像素模型
  （样本里为 5 安卓 / 1 iOS，其中那 1 张 iOS 靠灵动岛露馅）

因为本池没有翻拍图，精确 iPhone 分辨率几乎等同于 iOS；这与“翻拍图里 Make 是相机”的坑无关。
覆盖率：iOS 58.8% + 安卓 20.1% = 78.9% 可免费高精度标注，其余 21.1% 交像素模型。

再次强调：分辨率只用于产出自举标签，绝不能作为分类器的输入特征——造假者随手改尺寸即可绕过。
"""

from __future__ import annotations

from typing import Final

from .metadata_seed import IPHONE_RESOLUTIONS

# iPhone 从不使用这些面板宽度；它们是安卓 HD/FHD/QHD 的标准短边。
ANDROID_PANEL_WIDTHS: Final[frozenset[int]] = frozenset({720, 1080, 1440})

# 自举标注结果取值。
BOOTSTRAP_IOS: Final[str] = "ios"
BOOTSTRAP_ANDROID: Final[str] = "android"
BOOTSTRAP_ABSTAIN: Final[str] = "abstain"


def resolution_platform(width: int, height: int) -> str:
    """仅凭分辨率给出高精度自举标签：'ios' | 'android' | 'abstain'。"""
    if width <= 0 or height <= 0:
        return BOOTSTRAP_ABSTAIN
    if (width, height) in IPHONE_RESOLUTIONS:
        return BOOTSTRAP_IOS
    if min(width, height) in ANDROID_PANEL_WIDTHS:
        return BOOTSTRAP_ANDROID
    return BOOTSTRAP_ABSTAIN
