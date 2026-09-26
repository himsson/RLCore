from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from core.registry import NETS

ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "elu": nn.ELU,
}


def orthogonal_init(module: nn.Module, gain: float = 1.0) -> nn.Module:
    for layer in module.modules():
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(layer.weight, gain)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
    return module


@NETS.register("mlp")
def mlp(
    in_dim: int,
    out_dim: int,
    hidden: Sequence[int] = (256, 256),
    activation: str = "relu",
    layer_norm: bool = False,
    out_gain: float = 0.01,
) -> nn.Sequential:
    act = ACTIVATIONS[activation]
    dims = [in_dim, *hidden]
    layers: list[nn.Module] = []
    for a, b in zip(dims[:-1], dims[1:]):
        layers.append(nn.Linear(a, b))
        if layer_norm:
            layers.append(nn.LayerNorm(b))
        layers.append(act())
    head = nn.Linear(dims[-1], out_dim)
    net = nn.Sequential(*layers, head)
    orthogonal_init(net, gain=2**0.5)
    nn.init.orthogonal_(head.weight, out_gain)
    nn.init.zeros_(head.bias)
    return net


@NETS.register("dueling")
class DuelingHead(nn.Module):
    def __init__(self, in_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.value = mlp(in_dim, 1, (hidden,), out_gain=1.0)
        self.advantage = mlp(in_dim, n_actions, (hidden,), out_gain=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = self.value(x)
        adv = self.advantage(x)
        return value + adv - adv.mean(dim=-1, keepdim=True)


@NETS.register("ensemble")
class Ensemble(nn.Module):
    def __init__(self, factory, n: int = 2):
        super().__init__()
        self.members = nn.ModuleList([factory() for _ in range(n)])

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return torch.stack([m(*args, **kwargs) for m in self.members], dim=0)

    def min(self, *args, **kwargs) -> torch.Tensor:
        return self.forward(*args, **kwargs).min(dim=0).values


class NoisyLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, sigma: float = 0.5):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.sigma = sigma
        self.weight_mu = nn.Parameter(torch.empty(out_dim, in_dim))
        self.weight_sigma = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias_mu = nn.Parameter(torch.empty(out_dim))
        self.bias_sigma = nn.Parameter(torch.empty(out_dim))
        self.register_buffer("weight_eps", torch.zeros(out_dim, in_dim))
        self.register_buffer("bias_eps", torch.zeros(out_dim))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        bound = 1.0 / self.in_dim**0.5
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.uniform_(self.bias_mu, -bound, bound)
        nn.init.constant_(self.weight_sigma, self.sigma / self.in_dim**0.5)
        nn.init.constant_(self.bias_sigma, self.sigma / self.out_dim**0.5)

    @staticmethod
    def _scaled_noise(size: int, device) -> torch.Tensor:
        x = torch.randn(size, device=device)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self) -> None:
        eps_in = self._scaled_noise(self.in_dim, self.weight_mu.device)
        eps_out = self._scaled_noise(self.out_dim, self.weight_mu.device)
        self.weight_eps.copy_(eps_out.outer(eps_in))
        self.bias_eps.copy_(eps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_eps
            bias = self.bias_mu + self.bias_sigma * self.bias_eps
        else:
            weight, bias = self.weight_mu, self.bias_mu
        return nn.functional.linear(x, weight, bias)
