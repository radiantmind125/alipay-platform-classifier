"""状态栏条数据集（读 prepare_dataset.py 产出的清单）。用共用 preprocess，避免 train/serve 偏差。

温和增强（仅训练）：轻微亮度/对比度抖动、小幅横向缩放。绝不做强 JPEG/模糊——那会抹掉
状态栏图标/文字的抗锯齿，正是模型要看的判别信号。
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset

from preprocess import normalize, strip_to_canvas  # 同目录，共用预处理


def _augment(canvas_uint8: np.ndarray) -> np.ndarray:
    pil = Image.fromarray(canvas_uint8)
    if random.random() < 0.6:
        pil = ImageEnhance.Brightness(pil).enhance(random.uniform(0.85, 1.15))
    if random.random() < 0.6:
        pil = ImageEnhance.Contrast(pil).enhance(random.uniform(0.9, 1.1))
    if random.random() < 0.4:  # 轻微横向缩放，模拟不同状态栏比例
        w, h = pil.size
        s = random.uniform(0.94, 1.06)
        pil = pil.resize((max(1, int(w * s)), h)).resize((w, h))
    return np.asarray(pil)


class StripDataset(Dataset):
    def __init__(self, manifest: str | Path, *, train: bool = False) -> None:
        self.rows = [json.loads(l) for l in Path(manifest).read_text(encoding="utf-8").splitlines() if l.strip()]
        self.train = train

    def __len__(self) -> int:
        return len(self.rows)

    def labels(self) -> list[int]:
        return [int(r["label"]) for r in self.rows]

    def __getitem__(self, i: int):
        row = self.rows[i]
        with Image.open(row["strip"]) as im:
            canvas = strip_to_canvas(np.asarray(im.convert("RGB")))
        if self.train:
            canvas = _augment(canvas)
        x = torch.from_numpy(normalize(canvas))
        return x, int(row.get("label", -1))
