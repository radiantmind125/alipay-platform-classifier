"""SSP-tiny:在残差块上的极小 CNN,用来"重训 SSP"——把开源 SSP 的 ResNet50 换成
设备模型同款 DWSep 小骨干,吃我们自己的残差前端(SRM+NPR),多标签输出,CPU 可训可跑。

输入:每张图选出的 K 个残差块 [B, K, C, 32, 32](C=4:SRM3+NPR1,来自 ssp_patch.residual_stack)。
逐块编码 -> 对 K 个块 mean 池化 -> 多标签头(head_screen 翻拍 / head_gen AI / ...)。
这就是经理说的"那几个评估模型重新训练"里,我们自己域上重训的那个小模型。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import DWSep  # 复用设备模型的深度可分离块


class _PatchEncoder(nn.Module):
    def __init__(self, cin: int = 4, width: float = 1.0) -> None:
        super().__init__()

        def c(n: int) -> int:
            return max(8, int(round(n * width)))

        self.stem = nn.Sequential(
            nn.Conv2d(cin, c(16), 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c(16)), nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            DWSep(c(16), c(32), stride=2),
            DWSep(c(32), c(64), stride=2),
            DWSep(c(64), c(96), stride=1),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = c(96)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [N, cin, 32, 32] -> [N, out_dim]
        return self.pool(self.blocks(self.stem(x))).flatten(1)


class SSPTinyNet(nn.Module):
    """残差块 -> 逐块编码 -> mean 池化 -> 多标签 sigmoid 头。head_names 决定输出几个头。"""

    def __init__(self, cin: int = 4, head_names: tuple[str, ...] = ("screen",), width: float = 1.0) -> None:
        super().__init__()
        self.head_names = tuple(head_names)
        self.enc = _PatchEncoder(cin, width)
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(self.enc.out_dim, len(self.head_names))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, K, cin, 32, 32] -> [B, n_heads] logits
        b, k = x.shape[:2]
        e = self.enc(x.reshape(b * k, *x.shape[2:]))       # [B*K, D]
        e = e.reshape(b, k, -1).mean(dim=1)                # 对 K 个块 mean 池化 -> [B, D]
        return self.head(self.dropout(e))


def build_ssp(head_names: tuple[str, ...] = ("screen",), cin: int = 4, width: float = 1.0) -> SSPTinyNet:
    return SSPTinyNet(cin=cin, head_names=head_names, width=width)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
