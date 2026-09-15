"""数字串长度下界的测试。

这个脚本要回答的只有一个问题: **订单号是不是超过识别流程的单行长度上限。**
所以宁可数少不能数多 —— 下界超过上限, 真值就一定超过上限。
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

TRAINING = Path(__file__).resolve().parents[1] / "training"
sys.path.insert(0, str(TRAINING))

from count_digits import longest_digit_run, text_mask  # noqa: E402


def _imwrite(path: Path, img: np.ndarray) -> None:
    """cv2.imwrite 碰到非 ASCII 路径会直接失败(pytest 的 tmp_path 带中文用例名就是).

    和 count_digits 里读图用 fromfile + imdecode 是同一个道理, 写也要绕开。
    """
    ok, buf = cv2.imencode(path.suffix, img)
    assert ok, "编码失败"
    buf.tofile(str(path))


def _page_with_digits(n: int, tmp_path: Path, *, digit_w: int = 18,
                      gap: int = 6, height: int = 26) -> Path:
    """造一张白底图, 右半边画一行 n 个等宽方块当数字。"""
    H, W = 1400, 1000
    img = np.full((H, W), 255, np.uint8)
    x0, y0 = int(W * 0.32), 700
    for i in range(n):
        x = x0 + i * (digit_w + gap)
        if x + digit_w >= W:
            break
        img[y0:y0 + height, x:x + digit_w] = 0
    p = tmp_path / f"page_{n}.png"
    _imwrite(p, img)
    return p


def test_能数出一行等宽方块的个数(tmp_path: Path) -> None:
    p = _page_with_digits(28, tmp_path)
    got = longest_digit_run(p)
    assert got is not None
    assert got[0] == 28


def test_下界不会数多(tmp_path: Path) -> None:
    """粘连只会让它数少, 绝不会数多 —— 这是结论成立的前提。"""
    for n in (26, 30, 32):
        p = _page_with_digits(n, tmp_path)
        got = longest_digit_run(p)
        assert got is not None
        assert got[0] <= n, f"{n} 位被数成了 {got[0]}, 数多了"


def test_间距大的地方会断开(tmp_path: Path) -> None:
    """一行里有两段(比如标签和值), 只算最长那一段, 不能连起来算。"""
    H, W = 1400, 1000
    img = np.full((H, W), 255, np.uint8)
    y0, h, dw, gap = 700, 26, 18, 6
    x = int(W * 0.32)
    for _ in range(6):                      # 前一段 6 个
        img[y0:y0 + h, x:x + dw] = 0
        x += dw + gap
    x += 160                                # 明显更大的空档
    for _ in range(12):                     # 后一段 12 个
        img[y0:y0 + h, x:x + dw] = 0
        x += dw + gap
    p = tmp_path / "two_runs.png"
    _imwrite(p, img)
    got = longest_digit_run(p)
    assert got is not None
    assert got[0] == 12                     # 取长的那段, 不是 18


def test_短行不算(tmp_path: Path) -> None:
    """只有几个块的行不可能是订单号, 不要报出来。"""
    p = _page_with_digits(4, tmp_path)
    got = longest_digit_run(p)
    assert got is None or got[0] < 8


def test_只在左边的东西不参与(tmp_path: Path) -> None:
    """标签在左边, 值在右边。**整个落在左边**的内容不能混进来。

    ★ 已知限制: 判的是每个块自己的 x, 所以一串**从左边开始但一直延伸到右边**的东西,
      右半边那部分仍然会被数进去。真回单上标签只有两三个字(订单号), 到不了那么右,
      所以不影响实际使用。这里验的是"整串都在左边"这种情形。
    """
    H, W = 1400, 1000
    img = np.full((H, W), 255, np.uint8)
    y0, h, dw, gap = 700, 26, 18, 6
    x = 40
    for _ in range(6):                      # 6 个块, 到 x=184, 全在 22% 分界线(220)左边
        img[y0:y0 + h, x:x + dw] = 0
        x += dw + gap
    assert x < W * 0.22, "用例本身要保证整串都在左边"
    p = tmp_path / "left_only.png"
    _imwrite(p, img)
    got = longest_digit_run(p)
    assert got is None or got[0] < 8


def test_文字掩膜不分正负(tmp_path: Path) -> None:
    """白底黑字和黑底白字都要能切出来 —— 蓝底白字页靠的就是这个。"""
    H, W = 400, 400
    dark_on_light = np.full((H, W), 240, np.uint8)
    dark_on_light[180:210, 100:300] = 20
    light_on_dark = np.full((H, W), 20, np.uint8)
    light_on_dark[180:210, 100:300] = 240
    for img in (dark_on_light, light_on_dark):
        m = text_mask(img)
        assert m[190, 200] > 0, "这一块应当被算成文字"
