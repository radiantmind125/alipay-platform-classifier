"""融合层：把设备分类的两层结果合成最终设备标签；把多种信号融合成欺诈评分。

设计（依据评审）：
- 设备分类与欺诈检测**分开**。设备 = 分辨率(Tier-0) + CNN(Tier-1) 合并。
- “不一致(inconsistent)”与“假图(假)”不是 CNN 的类别，而是**融合决策**：多信号加权评分，
  中间带交人工，强多信号一致才自动拒。经理的“没有勾=假图”只是其中一个加权信号，不单独定论。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

DEVICE_FROM_RESOLUTION_CONF: Final[float] = 0.99


def merge_device(bootstrap_label: str, cnn: dict | None = None) -> dict:
    """合并 Tier-0（分辨率）与 Tier-1（CNN）得到最终设备标签。

    bootstrap_label ∈ {'ios','android','abstain'}；cnn 为 None 或 {'device','conf',...}。
    """
    if bootstrap_label in ("ios", "android"):
        return {"device": bootstrap_label, "source": "resolution", "confidence": DEVICE_FROM_RESOLUTION_CONF}
    if cnn is None:
        return {"device": "unknown", "source": "none", "confidence": 0.0}
    return {"device": cnn.get("device", "unknown"), "source": "cnn", "confidence": float(cnn.get("conf", 0.0))}


@dataclass(frozen=True)
class FraudSignals:
    """各信号为布尔；来源见字段说明。"""

    no_checkmark: bool = False          # mate 检测器：无 transfer_status（经理的“没有勾”）
    device_prior_conflict: bool = False # CNN 判定 与 分辨率/EXIF 先验 矛盾（疑似拼接篡改）
    photo_of_screen: bool = False       # 翻拍图
    exif_device_mismatch: bool = False  # 分辨率=iPhone 但 EXIF 是安卓厂商/含相机字段


# 权重可按经理对“误拒容忍度”调整（开放问题）。单个 no_checkmark -> 落人工带；两信号一致 -> 拒。
FRAUD_WEIGHTS: Final[dict[str, float]] = {
    "no_checkmark": 0.5,
    "device_prior_conflict": 0.35,
    "photo_of_screen": 0.25,
    "exif_device_mismatch": 0.35,
}
REJECT_THRESHOLD: Final[float] = 0.6
REVIEW_THRESHOLD: Final[float] = 0.3


@dataclass(frozen=True)
class FraudResult:
    score: float
    verdict: str                        # 'pass' | 'review' | 'reject'
    reasons: list[str] = field(default_factory=list)


def fraud_score(signals: FraudSignals) -> FraudResult:
    """多信号加权评分 -> pass/review/reject。"""
    score = sum(w for k, w in FRAUD_WEIGHTS.items() if getattr(signals, k))
    reasons = [k for k in FRAUD_WEIGHTS if getattr(signals, k)]
    if score >= REJECT_THRESHOLD:
        verdict = "reject"
    elif score >= REVIEW_THRESHOLD:
        verdict = "review"
    else:
        verdict = "pass"
    return FraudResult(round(score, 3), verdict, reasons)


def device_prior_conflict(resolution_platform: str, cnn_device: str, cnn_conf: float, *, min_conf: float = 0.8) -> bool:
    """分辨率给出明确平台、CNN 却高置信判成另一平台 -> 记为不一致（疑似篡改）。"""
    if resolution_platform not in ("ios", "android"):
        return False
    if cnn_device not in ("ios", "android"):
        return False
    return cnn_device != resolution_platform and cnn_conf >= min_conf
