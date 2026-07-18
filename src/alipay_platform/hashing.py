"""感知哈希（dHash），用于按交易分组和去重。

用途：(a) 把同一笔交易的所有近似副本（旋转、重压缩等变体）放进同一个 train/val/test 划分，
避免数据泄漏；(b) 主动学习时按分组限流，防止近似重复图刷屏标注队列或悄悄改变类别比例。

dHash（差值哈希）计算便宜，只需 PIL + numpy，对轻微的亮度/缩放/JPEG 变化鲁棒——正好是
检测器工程的增强和聊天软件重压缩会带来的那类变换。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def dhash(image: Image.Image, *, hash_size: int = 8) -> int:
    """返回 ``hash_size**2`` 位的差值哈希（整数）。

    缩放到 (hash_size+1, hash_size) 的灰度图，再按每个像素是否比右邻更亮置位。
    """
    if hash_size < 2:
        raise ValueError("hash_size 必须 >= 2")
    reduced = image.convert("L").resize((hash_size + 1, hash_size), Image.BILINEAR)
    pixels = np.asarray(reduced, dtype=np.int16)
    diff = pixels[:, 1:] > pixels[:, :-1]
    bits = diff.flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def dhash_path(path: str | Path, *, hash_size: int = 8) -> int:
    """按路径对文件做 dHash（只读文件头 + 小幅解码）。"""
    with Image.open(path) as image:
        return dhash(image, hash_size=hash_size)


def hamming_distance(left: int, right: int) -> int:
    """两个哈希之间不同的位数。"""
    return int(bin(left ^ right).count("1"))


def is_near_duplicate(left: int, right: int, *, threshold: int = 8) -> bool:
    """两个 dHash 相差不超过 ``threshold`` 位（默认 8/64）则判为近似重复。"""
    return hamming_distance(left, right) <= threshold
