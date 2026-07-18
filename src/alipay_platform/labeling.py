"""弃权式（Snorkel 风格）的标注函数投票聚合。

银标策略：
- 只有当至少 ``min_agree`` 个相互独立的标注函数意见一致、且没有冲突时，才自动采纳为银标。
- 任何冲突（既有 'ios' 票又有 'android' 票）=> 转人工。冲突本身也说明状态栏和对勾不一致，
  即疑似被篡改 / 伪造的截图。
- 其余情况（票数不够）=> 以“证据不足”转人工。

金标（人工标）在别处产生，这里绝不会凭空制造。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .metadata_seed import Vote
from .platform_labels import VOTE_ABSTAIN

# 聚合结果。
ACCEPT_IOS = "accept:ios"
ACCEPT_ANDROID = "accept:android"
REVIEW_CONFLICT = "review:conflict"          # 疑似篡改 / 伪造
REVIEW_INSUFFICIENT = "review:insufficient"  # 一致意见不够


@dataclass(frozen=True)
class Decision:
    """聚合后的银标决定，附带可追溯的依据。"""

    outcome: str
    label: str | None                      # 'ios' | 'android' | None
    ios_score: float
    android_score: float
    reasons: list[str] = field(default_factory=list)

    @property
    def accepted(self) -> bool:
        return self.outcome in (ACCEPT_IOS, ACCEPT_ANDROID)

    @property
    def needs_review(self) -> bool:
        return not self.accepted


def aggregate(votes: Iterable[Vote], *, min_agree: int = 2) -> Decision:
    """把相互独立的标注函数投票聚合成一个银标决定。

    ``min_agree`` 是自动采纳所需的、获胜一方的非弃权票数。置信度只用于审计打分；采纳与否
    只看一致的独立标注函数“个数”，这样再自信的单个标注函数也无法独自制造银标。
    """
    if min_agree < 1:
        raise ValueError("min_agree 必须 >= 1")
    votes = list(votes)
    ios = [v for v in votes if v.label == "ios"]
    android = [v for v in votes if v.label == "android"]
    reasons = [f"{v.label}:{v.reason}" for v in votes if v.label != VOTE_ABSTAIN]

    ios_score = sum(v.confidence for v in ios)
    android_score = sum(v.confidence for v in android)

    if ios and android:
        return Decision(REVIEW_CONFLICT, None, ios_score, android_score, reasons)
    if len(ios) >= min_agree:
        return Decision(ACCEPT_IOS, "ios", ios_score, android_score, reasons)
    if len(android) >= min_agree:
        return Decision(ACCEPT_ANDROID, "android", ios_score, android_score, reasons)
    return Decision(REVIEW_INSUFFICIENT, None, ios_score, android_score, reasons)
