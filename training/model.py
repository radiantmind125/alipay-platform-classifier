"""状态栏条的极小 CNN（深度可分离卷积），面向 CPU 量化部署。

输入：状态栏条缩放到固定 (H=64, W=384) 的 RGB。输出：2 类（android=0, ios=1）。
参数量约 10 万级，INT8 量化后在 CPU 上单张推理约毫秒级。
"""

from __future__ import annotations

import torch
from torch import nn

PLATFORM_CLASSES = ("android", "ios")


class DWSep(nn.Module):
    """深度可分离卷积块：depthwise 3x3 + pointwise 1x1 + BN + ReLU。"""

    def __init__(self, cin: int, cout: int, stride: int = 1) -> None:
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 3, stride=stride, padding=1, groups=cin, bias=False)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pw(self.dw(x))))


class StatusBarNet(nn.Module):
    def __init__(self, num_classes: int = 2, width: float = 1.0) -> None:
        super().__init__()

        def c(n: int) -> int:
            return max(8, int(round(n * width)))

        self.stem = nn.Sequential(
            nn.Conv2d(3, c(16), 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c(16)),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            DWSep(c(16), c(32), stride=2),
            DWSep(c(32), c(48), stride=2),
            DWSep(c(48), c(64), stride=2),
            DWSep(c(64), c(96), stride=1),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(c(96), num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.head(self.dropout(x))


def build_model(num_classes: int = 2, width: float = 1.0) -> StatusBarNet:
    return StatusBarNet(num_classes=num_classes, width=width)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
