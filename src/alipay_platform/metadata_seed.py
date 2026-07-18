"""零解码的元数据标注函数（高精度银标）。

只读文件头 / EXIF / ICC，不解码整张图，因此在几万张图上几乎零成本，第一天就能得到一批
高精度的“苹果”正样本种子。

关键：这里的一切只用于弱标注，绝不能进模型当特征。分辨率 / ICC / EXIF 这些造假者随手
一改就没了，拿来当特征会让模型学到“非 iPhone 分辨率 => 安卓”这种捷径。最终模型必须
依赖状态栏和对勾的像素，正如需求所强调的。

设计要点：
- 安卓没有干净的分辨率白名单，所以分辨率命中只投“苹果”，不命中就弃权。绝不用元数据判安卓。
- 对翻拍图，EXIF Make 是拍摄相机（iPhone 也能拍安卓屏），所以只在没有相机拍摄字段的
  原生截图上才相信 Make。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from PIL import Image, ExifTags

from .platform_labels import VOTE_ABSTAIN, validate_vote

# 冻结的 iPhone 原生截图分辨率表（竖屏，宽 x 高，渲染尺寸=截图尺寸）。
# 版本：2026-07（覆盖到 iPhone 16 系列）。新机型上市时需按 Apple 规格重新核对本表。
# 只收 iPhone 独有分辨率；与常见安卓撞车的一律排除以保证 ~99% 精度（低召回没关系，
# 漏掉的会落到 abstain 交像素模型）：
#   排除 1080x1920（iPhone 6+/7+/8+ 显示降采样）——安卓 FHD 常用
#   排除 1080x2340（iPhone 12/13 mini）——安卓 FHD+ 极常用；mini 罕见，故舍之保精度
#   iPad 不入表（比例~4:3，会被翻拍门按长宽比另行处理；支付宝 iPad 场景罕见）
IPHONE_TABLE_VERSION: Final[str] = "2026-07 (up to iPhone 16 series)"
_IPHONE_PORTRAIT: Final[frozenset[tuple[int, int]]] = frozenset(
    {
        (640, 960),     # 4 / 4S（老机型，罕见）
        (640, 1136),    # 5 / 5s / 5c / SE(1)
        (750, 1334),    # 6 / 6s / 7 / 8 / SE2 / SE3
        (1242, 2208),   # 6+ / 6s+ / 7+ / 8+（渲染尺寸）
        (828, 1792),    # XR / 11
        (1125, 2436),   # X / XS / 11 Pro
        (1242, 2688),   # XS Max / 11 Pro Max
        (1170, 2532),   # 12 / 12 Pro / 13 / 13 Pro / 14 / 16e
        (1284, 2778),   # 12 Pro Max / 13 Pro Max / 14 Plus
        (1179, 2556),   # 14 Pro / 15 / 15 Pro / 16 / 16 Plus 之外的 6.1"
        (1290, 2796),   # 14 Pro Max / 15 Plus / 15 Pro Max / 16 Plus
        (1206, 2622),   # 16 Pro
        (1320, 2868),   # 16 Pro Max
    }
)

# 两个方向都算命中（截图旋转后像素对不变）。
IPHONE_RESOLUTIONS: Final[frozenset[tuple[int, int]]] = frozenset(
    _IPHONE_PORTRAIT | {(h, w) for (w, h) in _IPHONE_PORTRAIT}
)

# 需要用到的 EXIF tag id（避免在代码里直接写魔数）。
_TAG_TO_ID: Final[dict[str, int]] = {name: tag for tag, name in ExifTags.TAGS.items()}
_MAKE_ID: Final[int] = _TAG_TO_ID.get("Make", 271)
# 只要出现其中任一字段，基本可判定是真实相机拍摄（翻拍图）。
_CAPTURE_TAG_IDS: Final[frozenset[int]] = frozenset(
    _TAG_TO_ID[name]
    for name in ("ExposureTime", "FNumber", "ISOSpeedRatings", "DateTimeOriginal", "FocalLength", "LensModel")
    if name in _TAG_TO_ID
)


@dataclass(frozen=True)
class Vote:
    """一次标注函数投票。``label`` 取 'ios' | 'android' | 'abstain'。"""

    label: str
    confidence: float
    reason: str

    def __post_init__(self) -> None:
        validate_vote(self.label)


def _abstain(reason: str) -> Vote:
    return Vote(VOTE_ABSTAIN, 0.0, reason)


def normalize_resolution(width: int, height: int) -> tuple[int, int]:
    """返回 (短边, 长边)，让匹配与方向无关。"""
    return (min(width, height), max(width, height))


def resolution_vote(width: int, height: int) -> Vote:
    """iPhone 分辨率白名单：命中 => 苹果（高精度）；否则弃权。"""
    if width <= 0 or height <= 0:
        return _abstain("尺寸非法")
    if (width, height) in IPHONE_RESOLUTIONS:
        return Vote("ios", 0.97, f"精确命中 iPhone 分辨率 {width}x{height}")
    return _abstain(f"分辨率 {width}x{height} 不在 iPhone 白名单")


def icc_vote(icc_profile: bytes | None) -> Vote:
    """带 Display-P3 ICC 标 => 偏苹果；sRGB / 无标 => 弃权。

    iOS 截图通常带 Display P3 色彩标；安卓截图几乎都是 sRGB 或不带标。这是软投票（P3 在
    别处也可能出现），所以只能起佐证作用，不能单独定论。
    """
    if not icc_profile:
        return _abstain("无 ICC 色彩标")
    head = icc_profile[:512].lower()
    if b"display p3" in head or b"displayp3" in head or b"p3" in head[:200]:
        return Vote("ios", 0.6, "Display-P3 ICC 色彩标")
    return _abstain("非 P3 的 ICC 色彩标")


def exif_make_vote(make: str | None, has_capture_tags: bool) -> Vote:
    """原生截图（无相机字段）上 EXIF Make=Apple => 弱苹果。

    如果存在相机拍摄字段，说明是照片（可能是用 iPhone 翻拍安卓屏），此时 Make 是相机而不是
    截图的系统，所以弃权。
    """
    if has_capture_tags:
        return _abstain("含相机拍摄 EXIF（翻拍图），Make 是相机而非系统")
    if make and "apple" in make.strip().lower():
        return Vote("ios", 0.55, "原生截图 EXIF Make=Apple")
    return _abstain("无原生截图的 Apple Make")


@dataclass(frozen=True)
class MetadataFacts:
    """从单个文件里读到的、只看文件头的元数据。"""

    width: int
    height: int
    icc_profile: bytes | None
    make: str | None
    has_capture_tags: bool


def read_metadata_facts(path: str | Path) -> MetadataFacts:
    """读取文件头 / EXIF / ICC，不解码像素。

    ``Image.open`` 是惰性的：``.size`` 和 ``.info`` 来自文件头，``getexif()`` 只解析
    EXIF 块。全程不调用 ``.load()``，所以整张位图从不被解码。
    """
    with Image.open(path) as image:
        width, height = image.size
        icc_profile = image.info.get("icc_profile")
        make: str | None = None
        has_capture_tags = False
        try:
            exif = image.getexif()
        except Exception:
            exif = None
        if exif:
            raw_make = exif.get(_MAKE_ID)
            if isinstance(raw_make, bytes):
                raw_make = raw_make.decode("ascii", "ignore")
            make = raw_make if isinstance(raw_make, str) else None
            has_capture_tags = any(tag in exif for tag in _CAPTURE_TAG_IDS)
    return MetadataFacts(width, height, icc_profile, make, has_capture_tags)


def metadata_votes(facts: MetadataFacts) -> list[Vote]:
    """在已读好的元数据上运行全部元数据标注函数。"""
    return [
        resolution_vote(facts.width, facts.height),
        icc_vote(facts.icc_profile),
        exif_make_vote(facts.make, facts.has_capture_tags),
    ]


def metadata_votes_for_path(path: str | Path) -> list[Vote]:
    """便捷函数：读文件头元数据并返回全部元数据投票。"""
    return metadata_votes(read_metadata_facts(path))
