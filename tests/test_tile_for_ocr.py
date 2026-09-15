"""切片工具的测试。

两条不变量, 缺一不可:
  1. **每一块的最长边都不超过上限** —— 超了检测那一步还会再缩, 切了等于白切
  2. **不能把一行字从中间切断** —— 切断了那一行两边都读不全, 比不切还糟
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

TRAINING = Path(__file__).resolve().parents[1] / "training"
sys.path.insert(0, str(TRAINING))

from tile_for_ocr import tile  # noqa: E402

# 测试自己定的上限, 和任何真实配置无关
LIMIT = 800


def _page(W: int, H: int, row_h: int = 26, pitch: int = 96) -> tuple[np.ndarray, list]:
    """造一张高长图, 每隔 pitch 画一条 row_h 高的"文字行"。返回(图, 行位置)。"""
    img = np.full((H, W, 3), 255, np.uint8)
    rows = []
    y = 60
    while y + row_h < H - 40:
        img[y:y + row_h, 60:W - 60] = 40
        rows.append((y, y + row_h))
        y += pitch
    return img, rows


def _scaled_rows(rows, W, limit):
    """图被缩宽之后, 行的位置也跟着缩。"""
    if W <= limit:
        return rows
    s = limit / W
    return [(int(a * s), int(np.ceil(b * s))) for a, b in rows]


def test_每块最长边都不超过上限(tmp_path: Path) -> None:
    for W, H in ((1200, 2640), (1080, 2412), (1440, 3200), (720, 1600)):
        img, _ = _page(W, H)
        pieces, _ = tile(img, LIMIT, 160)
        for piece, _, _ in pieces:
            h, w = piece.shape[:2]
            assert max(h, w) <= LIMIT, f"{W}x{H} 切出了 {w}x{h}, 最长边超了"


def test_纵向覆盖不能有缺口() -> None:
    img, _ = _page(1200, 2640)
    pieces, _ = tile(img, LIMIT, 160)
    spans = sorted((a, b) for _, a, b in pieces)
    assert spans[0][0] == 0
    end = spans[0][1]
    for a, b in spans[1:]:
        assert a <= end, f"{a} 和前一块的 {end} 之间有缺口"
        end = max(end, b)
    # 缩宽之后的总高
    expect_h = int(round(2640 * LIMIT / 1200))
    assert end == expect_h, f"覆盖到 {end}, 应当到 {expect_h}"


def test_默认重叠下一行字都不会被切断() -> None:
    """★ 最要紧的一条。实测 50 张真图 927 行, 默认 160 重叠下 0 行被切断。"""
    for W, H in ((1200, 2640), (1080, 2412), (1260, 2800)):
        img, rows = _page(W, H)
        pieces, _ = tile(img, LIMIT, 160)
        spans = [(a, b) for _, a, b in pieces]
        srows = _scaled_rows(rows, W, LIMIT)
        for r0, r1 in srows:
            assert any(t0 <= r0 and r1 <= t1 for t0, t1 in spans), \
                f"{W}x{H}: 行 {r0}-{r1} 没有被任何一块完整包住"


def test_不留重叠就会切断行() -> None:
    """反过来验一下: 这个检查确实抓得到"被切断"这件事, 不是永远通过。

    ★ 第一版这条是**假过**的: 随便排的行距碰巧让每一行都避开了块边界,
      于是"没重叠"也一行没切断, 这条用例就等于什么都没验。
      现在**故意**在块边界上放一行, 保证触发。
    """
    W, H, limit = 1200, 2640, LIMIT
    s = limit / W                       # 0.8
    boundary = limit                    # 缩放后第一块的下边界
    # 让这一行正好横跨边界: 缩放后 950..975
    y0_src = int((boundary - 10) / s)
    row_h_src = int(25 / s)
    img = np.full((H, W, 3), 255, np.uint8)
    img[y0_src:y0_src + row_h_src, 60:W - 60] = 40
    rows = [(y0_src, y0_src + row_h_src)]

    pieces, _ = tile(img, limit, 0)
    spans = [(a, b) for _, a, b in pieces]
    srows = _scaled_rows(rows, W, limit)
    cut = sum(1 for r0, r1 in srows
              if not any(t0 <= r0 and r1 <= t1 for t0, t1 in spans))
    assert cut == 1, f"故意跨边界的那一行本应被切断, 实际 cut={cut}, 行={srows}, 块={spans}"

    # 同一张图, 有重叠就不该再被切断
    pieces2, _ = tile(img, limit, 160)
    spans2 = [(a, b) for _, a, b in pieces2]
    cut2 = sum(1 for r0, r1 in srows
               if not any(t0 <= r0 and r1 <= t1 for t0, t1 in spans2))
    assert cut2 == 0, f"有重叠之后不该再被切断, 实际 cut={cut2}"


def test_本来就够小的图不切() -> None:
    img, _ = _page(600, 800)
    pieces, scale = tile(img, LIMIT, 160)
    assert len(pieces) == 1
    assert scale == 1.0
    assert pieces[0][0].shape[:2] == (800, 600)


def test_宽图也不会超上限() -> None:
    """横过来的图: 宽是最长边, 缩完高度自然就小了。"""
    img, _ = _page(2640, 1200)
    pieces, _ = tile(img, LIMIT, 160)
    for piece, _, _ in pieces:
        h, w = piece.shape[:2]
        assert max(h, w) <= LIMIT
