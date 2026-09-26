from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn

from core.registry import NETS

from .mlp import orthogonal_init


@NETS.register("nature_cnn")
class NatureCNN(nn.Module):
    def __init__(self, in_shape: Tuple[int, ...], out_dim: int = 512, scale: float = 255.0):
        super().__init__()
        channels = in_shape[0]
        self.scale = scale
        self.body = nn.Sequential(
            nn.Conv2d(channels, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            flat = self.body(torch.zeros(1, *in_shape)).shape[1]
        self.head = nn.Sequential(nn.Linear(flat, out_dim), nn.ReLU())
        self.out_dim = out_dim
        orthogonal_init(self, gain=2**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x.float() / self.scale))


@NETS.register("impala")
class ImpalaCNN(nn.Module):
    def __init__(
        self,
        in_shape: Tuple[int, ...],
        out_dim: int = 256,
        depths: Sequence[int] = (16, 32, 32),
        scale: float = 255.0,
    ):
        super().__init__()
        self.scale = scale
        blocks: list[nn.Module] = []
        channels = in_shape[0]
        for depth in depths:
            blocks.append(_ImpalaBlock(channels, depth))
            channels = depth
        self.body = nn.Sequential(*blocks, nn.ReLU(), nn.Flatten())
        with torch.no_grad():
            flat = self.body(torch.zeros(1, *in_shape)).shape[1]
        self.head = nn.Sequential(nn.Linear(flat, out_dim), nn.ReLU())
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x.float() / self.scale))


class _Residual(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class _ImpalaBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.MaxPool2d(3, stride=2, padding=1),
            _Residual(out_channels),
            _Residual(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
