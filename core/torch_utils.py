from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn as nn


def to_tensor(x, device: str = "cpu", dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    t = torch.as_tensor(np.asarray(x), device=device)
    if dtype is not None:
        return t.to(dtype)
    if t.dtype == torch.float64:
        return t.float()
    if t.dtype == torch.bool:
        return t.float()
    return t


def to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


@torch.no_grad()
def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.mul_(1.0 - tau).add_(sp, alpha=tau)
    for tb, sb in zip(target.buffers(), source.buffers()):
        tb.copy_(sb)


@torch.no_grad()
def hard_update(target: nn.Module, source: nn.Module) -> None:
    target.load_state_dict(source.state_dict())


def freeze(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        p.requires_grad_(False)
    return module


def clip_grads(params: Iterable[torch.nn.Parameter], max_norm: float) -> float:
    return float(torch.nn.utils.clip_grad_norm_(params, max_norm))


def grad_norm(params: Iterable[torch.nn.Parameter]) -> float:
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum())
    return total**0.5


def explained_variance(pred: torch.Tensor, target: torch.Tensor) -> float:
    var = target.var()
    if float(var) == 0.0:
        return float("nan")
    return float(1.0 - (target - pred).var() / var)


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def huber(pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    return nn.functional.smooth_l1_loss(pred, target, beta=delta, reduction="none")


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
