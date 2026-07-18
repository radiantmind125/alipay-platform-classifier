"""平台分类的固定标签定义（安卓 / 苹果 / 不一致）。

沿用检测器工程 ``labels.py`` + ``model.validate_checkpoint_classes`` 的写法，
保证权重文件不会在类别含义上被悄悄改动。

类别顺序固定不变。``inconsistent`` 是给对抗/篡改截图用的第三类：例如把苹果状态栏
拼到安卓页面上、把状态栏裁掉、或者 AI 生成的截图里状态栏和成功对勾互相矛盾。推理时
两路（状态栏 + 对勾）会做一致性交叉校验，出现矛盾就归到 ``inconsistent``，作为疑似
篡改的信号交给真伪模型，而不是强行判一个平台。
"""

from __future__ import annotations

from typing import Final

# 顺序固定。这里 0 是合法类别（不同于检测器把 0 留给背景），因为这是普通图像分类。
PLATFORM_CLASSES: Final[tuple[str, ...]] = ("android", "ios", "inconsistent")

LABEL_TO_ID: Final[dict[str, int]] = {name: index for index, name in enumerate(PLATFORM_CLASSES)}
ID_TO_LABEL: Final[dict[int, str]] = {index: name for name, index in LABEL_TO_ID.items()}
NUM_CLASSES: Final[int] = len(PLATFORM_CLASSES)

# 标注函数允许的投票值。``abstain`` 表示“没有意见”，绝不能当成支持或反对某一类的证据。
VOTE_ABSTAIN: Final[str] = "abstain"
VALID_VOTES: Final[frozenset[str]] = frozenset({"android", "ios", VOTE_ABSTAIN})


def validate_label(name: str) -> str:
    """校验类别名，非法则抛出带提示的异常。"""
    if name not in LABEL_TO_ID:
        accepted = ", ".join(PLATFORM_CLASSES)
        raise ValueError(f"未知平台标签 {name!r}，应为其中之一：{accepted}")
    return name


def validate_vote(vote: str) -> str:
    """校验标注函数的投票值，非法则抛异常。"""
    if vote not in VALID_VOTES:
        accepted = ", ".join(sorted(VALID_VOTES))
        raise ValueError(f"未知投票 {vote!r}，应为其中之一：{accepted}")
    return vote


def validate_checkpoint_classes(payload: object) -> None:
    """拒绝加载类别顺序不一致的权重文件。

    比只校验张量形状更严格：两个权重都可能是三个 logit，却各自代表不同含义。这里刻意
    照搬检测器工程的守卫逻辑，让标签错配时直接报错而不是静默载入。
    """
    expected = list(PLATFORM_CLASSES)
    found = payload.get("classes") if isinstance(payload, dict) else None
    if not isinstance(found, (list, tuple)) or list(found) != expected:
        raise ValueError(
            "权重文件的类别与当前平台定义不一致。"
            f"期望 {expected}，实际 {found!r}。请用已标注数据重新训练。"
        )
