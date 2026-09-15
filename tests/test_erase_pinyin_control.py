"""`--no-erase` 对照组的测试。

为什么要有这个对照组
--------------------
★★★★★ 原图是 jpg, 擦完存的是 png。直接拿这两个去比, 比的是
   **擦拼音 + 换编码**两件事混在一起, 分不清哪件起的作用。

   这个坑真踩过: 拿 jpg 解码的图和 BGR 存成 png 再解码的图比, 报出来
   40 张全被擦坏了 —— 其实**一张都没坏**, 差的全是编码。

   `--no-erase` 走**完全一样**的解码和存盘, 但一个像素都不改。
   三列一比就分得清:
       原图jpg -> 原图png    这一步的差 = 换编码带来的
       原图png -> 擦完png    这一步的差 = 真正擦拼音带来的
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

TRAINING = Path(__file__).resolve().parents[1] / "training"
SCRIPT = TRAINING / "erase_pinyin.py"


def _fake_receipt(p: Path) -> np.ndarray:
    """造一张有"小块压在大块上面"的图, 让擦除逻辑有东西可擦。

    ★★★★★ 返回的是**从 jpg 读回来的**那份, 不是编码前那份。

    写这个测试时自己就踩了一次: 一开始返回编码前的数组, 拿它和对照组的 png
    比, 报"对照组改了像素" —— 其实对照组一个像素都没改, **差的是 jpg 的有损压缩**。
    和这个对照组当初要解决的问题**一模一样**: 隔着一次编码去比, 比出来的
    永远混着编码的差。
    """
    img = np.full((900, 500, 3), 246, np.uint8)
    for row in range(6):
        y = 80 + row * 120
        for col in range(10):          # 大块: 汉字
            x = 40 + col * 42
            img[y: y + 34, x: x + 34] = 40
        for col in range(10):          # 小块: 压在正上方的拼音
            x = 44 + col * 42
            img[y - 20: y - 8, x: x + 26] = 60
    cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tofile(str(p))
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)


def _run(src: Path, out: Path, *extra: str) -> str:
    p = subprocess.run([sys.executable, str(SCRIPT),
                        "--src", str(src), "--out", str(out), *extra],
                       capture_output=True, text=True, encoding="utf-8")
    return p.stdout + p.stderr


def test_对照组必须和原图像素完全一致(tmp_path: Path) -> None:
    """★★★★★ 这一条立不住, 整个对照就没意义了。"""
    src = tmp_path / "src"
    src.mkdir()
    orig = _fake_receipt(src / "a.jpg")

    out = tmp_path / "ne"
    _run(src, out, "--no-erase")

    got = cv2.imdecode(np.fromfile(str(out / "a_clean.png"), np.uint8),
                       cv2.IMREAD_COLOR)
    # 注意: 这里比的是**解码后的像素**, 不是文件字节。
    # jpg 解出来什么样, png 就该存什么样, 一个像素都不能差。
    assert got.shape == orig.shape
    assert not np.any(cv2.absdiff(got, orig)), "对照组改动了像素, 那就不是对照组了"


def test_不给这个开关时确实会擦(tmp_path: Path) -> None:
    """对照组要有意义, 正常那一路必须真的改了东西。"""
    src = tmp_path / "src"
    src.mkdir()
    orig = _fake_receipt(src / "a.jpg")

    out = tmp_path / "e"
    _run(src, out)

    f = out / "a_clean.png"
    assert f.exists(), "正常那一路没出文件"
    got = cv2.imdecode(np.fromfile(str(f), np.uint8), cv2.IMREAD_COLOR)
    changed = int(np.count_nonzero(cv2.absdiff(got, orig).max(axis=2)))
    assert changed > 0, "正常那一路一个像素都没改, 那和对照组就没差别了"


def test_对照组一张都不算擦(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    for i in range(3):
        _fake_receipt(src / f"{i}.jpg")
    txt = _run(src, tmp_path / "ne", "--no-erase")
    assert "擦了 0" in txt, f"对照组不该擦任何东西:\n{txt}"


def test_对照组也要给每张都出文件(tmp_path: Path) -> None:
    """两边张数必须一样, 否则没法比。"""
    src = tmp_path / "src"
    src.mkdir()
    for i in range(4):
        _fake_receipt(src / f"{i}.jpg")

    ne, e = tmp_path / "ne", tmp_path / "e"
    _run(src, ne, "--no-erase")
    _run(src, e)
    assert len(list(ne.glob("*.png"))) == 4
    assert len(list(ne.glob("*.png"))) == len(list(e.glob("*.png"))), \
        "两边张数对不上, 比出来的数会偏"


def test_两个反向的开关不能一起给(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    _fake_receipt(src / "a.jpg")
    txt = _run(src, tmp_path / "o", "--no-erase", "--force")
    assert "不能一起给" in txt
