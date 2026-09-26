from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

Obs = np.ndarray
Action = Any


@dataclass(slots=True)
class Space:
    shape: Tuple[int, ...] = ()
    discrete: bool = False
    n: Optional[int] = None
    low: Optional[np.ndarray] = None
    high: Optional[np.ndarray] = None
    dtype: Any = np.float32

    @staticmethod
    def box(shape, low=-np.inf, high=np.inf, dtype=np.float32) -> "Space":
        shape = tuple(shape)
        return Space(
            shape=shape,
            discrete=False,
            low=np.broadcast_to(np.asarray(low, dtype=dtype), shape).copy(),
            high=np.broadcast_to(np.asarray(high, dtype=dtype), shape).copy(),
            dtype=dtype,
        )

    @staticmethod
    def categorical(n: int) -> "Space":
        return Space(shape=(), discrete=True, n=int(n), dtype=np.int64)

    @property
    def size(self) -> int:
        return int(np.prod(self.shape)) if self.shape else 1

    @property
    def action_dim(self) -> int:
        return int(self.n) if self.discrete else self.size

    def sample(self, rng: np.random.Generator, batch: int = 0):
        shape = (batch, *self.shape) if batch else self.shape
        if self.discrete:
            return rng.integers(0, self.n, size=shape or None)
        low = self.low if self.low is not None else np.full(self.shape, -1.0, np.float32)
        high = self.high if self.high is not None else np.full(self.shape, 1.0, np.float32)
        low = np.clip(low, -1e6, 1e6)
        high = np.clip(high, -1e6, 1e6)
        return rng.uniform(low, high, size=shape).astype(self.dtype)

    def contains(self, x) -> bool:
        x = np.asarray(x)
        if self.discrete:
            return bool(np.all((x >= 0) & (x < self.n)))
        if x.shape[-len(self.shape) :] != self.shape:
            return False
        if self.low is None or self.high is None:
            return True
        return bool(np.all(x >= self.low - 1e-6) and np.all(x <= self.high + 1e-6))


@dataclass(slots=True)
class Step:
    obs: Obs
    action: Action
    reward: float
    next_obs: Obs
    terminated: bool
    truncated: bool
    info: Dict[str, Any] = field(default_factory=dict)

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


@dataclass(slots=True)
class Batch:
    obs: Any
    actions: Any
    rewards: Any
    next_obs: Any
    terminated: Any
    truncated: Any
    extras: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.rewards)

    def to(self, device: str, dtype=None) -> "Batch":
        import torch

        def conv(x, floats=True):
            t = torch.as_tensor(np.asarray(x), device=device)
            if floats and t.is_floating_point() and dtype is not None:
                t = t.to(dtype)
            if t.dtype == torch.bool:
                t = t.float()
            return t

        return Batch(
            obs=conv(self.obs),
            actions=conv(self.actions, floats=False),
            rewards=conv(self.rewards),
            next_obs=conv(self.next_obs),
            terminated=conv(self.terminated),
            truncated=conv(self.truncated),
            extras={k: conv(v) for k, v in self.extras.items()},
        )
