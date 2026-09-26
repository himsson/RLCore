from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0
HALF_LOG_2PI = 0.5 * math.log(2 * math.pi)
HALF_LOG_2PIE = 0.5 * math.log(2 * math.pi * math.e)
LOG_2 = math.log(2.0)


class Policy(ABC):
    @abstractmethod
    def sample(self) -> torch.Tensor:
        ...

    @abstractmethod
    def mode(self) -> torch.Tensor:
        ...

    @abstractmethod
    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def entropy(self) -> torch.Tensor:
        ...


class Categorical(Policy):
    def __init__(self, logits: torch.Tensor, mask: Optional[torch.Tensor] = None):
        if mask is not None:
            logits = logits.masked_fill(~mask.bool(), torch.finfo(logits.dtype).min)
        self.raw_logits = logits
        self._logits: Optional[torch.Tensor] = None

    @property
    def logits(self) -> torch.Tensor:
        if self._logits is None:
            self._logits = self.raw_logits - self.raw_logits.logsumexp(dim=-1, keepdim=True)
        return self._logits

    @property
    def probs(self) -> torch.Tensor:
        return self.logits.exp()

    def sample(self) -> torch.Tensor:
        noise = torch.empty_like(self.raw_logits).exponential_()
        return noise.log_().neg_().add_(self.raw_logits).argmax(dim=-1)

    def mode(self) -> torch.Tensor:
        return self.raw_logits.argmax(dim=-1)

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        return self.logits.gather(-1, action.long().unsqueeze(-1)).squeeze(-1)

    def entropy(self) -> torch.Tensor:
        return -(self.probs * self.logits).sum(dim=-1)


class DiagGaussian(Policy):
    def __init__(self, mean: torch.Tensor, log_std: torch.Tensor):
        self.mean = mean
        self.log_std = log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        self.std = self.log_std.exp()

    def sample(self) -> torch.Tensor:
        return self.mean + self.std * torch.randn_like(self.mean)

    def mode(self) -> torch.Tensor:
        return self.mean

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        z = (action - self.mean) / self.std
        lp = -0.5 * z.pow(2) - self.log_std - HALF_LOG_2PI
        return lp.sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        return (self.log_std + HALF_LOG_2PIE).sum(dim=-1)


class SquashedGaussian(DiagGaussian):
    def __init__(self, mean: torch.Tensor, log_std: torch.Tensor, eps: float = 1e-6):
        super().__init__(mean, log_std)
        self.eps = eps

    def sample_with_logprob(self) -> Tuple[torch.Tensor, torch.Tensor]:
        pre = self.mean + self.std * torch.randn_like(self.mean)
        action = torch.tanh(pre)
        log_prob = super().log_prob(pre)
        log_prob = log_prob - (2 * (LOG_2 - pre - F.softplus(-2 * pre))).sum(dim=-1)
        return action, log_prob

    def sample(self) -> torch.Tensor:
        return self.sample_with_logprob()[0]

    def mode(self) -> torch.Tensor:
        return torch.tanh(self.mean)

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        clipped = action.clamp(-1 + self.eps, 1 - self.eps)
        pre = torch.atanh(clipped)
        correction = torch.log(1 - clipped.pow(2) + self.eps).sum(dim=-1)
        return super().log_prob(pre) - correction


class StateIndependentStd(nn.Module):
    def __init__(self, action_dim: int, init: float = 0.0):
        super().__init__()
        self.log_std = nn.Parameter(torch.full((action_dim,), float(init)))

    def forward(self, mean: torch.Tensor) -> DiagGaussian:
        return DiagGaussian(mean, self.log_std.expand_as(mean))
