from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


class RunningMeanStd:
    def __init__(self, shape: Tuple[int, ...] = (), epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
        self._cache: tuple = ()

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        count = x.shape[0]
        if self.mean.ndim == 0:
            mean = x.mean()
            self._merge_scalar(float(mean), float(np.square(x - mean).mean()), count)
            return
        mean = x.mean(axis=0)
        self.update_from_moments(mean, np.square(x - mean).mean(axis=0), count)

    def _merge_scalar(self, mean: float, var: float, count: int) -> None:
        old_count = self.count
        total = old_count + count
        delta = mean - float(self.mean)
        share = count / total
        self.mean = np.float64(float(self.mean) + delta * share)
        self.var = np.float64(
            (float(self.var) * old_count + var * count + delta * delta * old_count * share) / total
        )
        self.count = total
        self._cache = ()

    def update_from_moments(self, mean: np.ndarray, var: np.ndarray, count: int) -> None:
        if self.mean.ndim == 0:
            self._merge_scalar(float(mean), float(var), count)
            return
        old_count = self.count
        total = old_count + count
        share = count / total
        delta = mean - self.mean
        self.mean = self.mean + delta * share
        m2 = self.var * old_count + var * count + np.square(delta) * (old_count * share)
        self.var = m2 / total
        self.count = total
        self._cache = ()

    def scalers(self, epsilon: float = 1e-8):
        if not self._cache or self._cache[2] != epsilon:
            mean = self.mean.astype(np.float32)
            inv_std = (1.0 / np.sqrt(self.var + epsilon)).astype(np.float32)
            self._cache = (mean, inv_std, epsilon)
        return self._cache[0], self._cache[1]

    def normalize(self, x: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
        mean, inv_std = self.scalers(epsilon)
        return (np.asarray(x, dtype=np.float32) - mean) * inv_std

    def denormalize(self, x: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
        return (x * np.sqrt(self.var + epsilon) + self.mean).astype(np.float32)

    def state_dict(self) -> Dict[str, Any]:
        return {"mean": self.mean.copy(), "var": self.var.copy(), "count": self.count}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.mean = np.asarray(state["mean"], dtype=np.float64)
        self.var = np.asarray(state["var"], dtype=np.float64)
        self.count = state["count"]
        self._cache = ()


class RunningReturn:
    def __init__(self, num_envs: int, gamma: float):
        self.gamma = gamma
        self.ret = np.zeros(num_envs, dtype=np.float64)
        self.rms = RunningMeanStd(shape=())

    def update(self, rewards: np.ndarray, dones: np.ndarray) -> np.ndarray:
        self.ret = self.ret * self.gamma + np.asarray(rewards, dtype=np.float64)
        self.rms.update(self.ret)
        self.ret[np.asarray(dones, dtype=bool)] = 0.0
        return self.rms.normalize(np.asarray(rewards, dtype=np.float64))

    def state_dict(self) -> Dict[str, Any]:
        return {"ret": self.ret.copy(), "rms": self.rms.state_dict(), "gamma": self.gamma}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.ret = np.asarray(state["ret"], dtype=np.float64)
        self.rms.load_state_dict(state["rms"])
        self.gamma = state["gamma"]
