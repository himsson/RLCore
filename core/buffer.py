from __future__ import annotations

from collections import deque
from typing import Any, Dict, Optional

import numpy as np

from .types import Batch, Space


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        obs_space: Space,
        action_space: Space,
        num_envs: int = 1,
        n_step: int = 1,
        gamma: float = 0.99,
        obs_dtype=None,
        rng: Optional[np.random.Generator] = None,
    ):
        self.num_envs = num_envs
        self.rows = max(capacity // num_envs, 1)
        self.capacity = self.rows * num_envs
        self.n_step = n_step
        self.gamma = gamma
        self.rng = rng or np.random.default_rng()
        obs_dtype = obs_dtype or obs_space.dtype
        adtype = np.int64 if action_space.discrete else np.float32
        self.obs = np.zeros((self.capacity, *obs_space.shape), dtype=obs_dtype)
        self.next_obs = np.zeros((self.capacity, *obs_space.shape), dtype=obs_dtype)
        self.actions = np.zeros((self.capacity, *action_space.shape), dtype=adtype)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.terminated = np.zeros(self.capacity, dtype=bool)
        self.truncated = np.zeros(self.capacity, dtype=bool)
        self.row = 0
        self.size = 0
        self._pending: deque = deque(maxlen=n_step)

    def __len__(self) -> int:
        return self.size * self.num_envs

    @property
    def full(self) -> bool:
        return self.size == self.rows

    def _collapse(self, obs, actions, rewards, next_obs, terminated, truncated):
        self._pending.append(
            (
                np.asarray(obs),
                np.asarray(actions),
                np.asarray(rewards, dtype=np.float32),
                np.asarray(next_obs),
                np.asarray(terminated, dtype=bool),
                np.asarray(truncated, dtype=bool),
            )
        )
        if len(self._pending) < self.n_step:
            return None
        obs, actions = self._pending[0][0], self._pending[0][1]
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        next_obs = self._pending[-1][3]
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)
        cut = np.zeros(self.num_envs, dtype=bool)
        for k, (_, _, r, no, term, trunc) in enumerate(self._pending):
            rewards = np.where(cut, rewards, rewards + (self.gamma**k) * r)
            done = term | trunc
            next_obs = np.where((~cut & done)[(...,) + (None,) * (next_obs.ndim - 1)], no, next_obs)
            terminated = np.where(cut, terminated, terminated | term)
            truncated = np.where(cut, truncated, truncated | trunc)
            cut = cut | done
        return obs, actions, rewards, next_obs, terminated, truncated

    def add(self, obs, actions, rewards, next_obs, terminated, truncated) -> None:
        if self.n_step > 1:
            collapsed = self._collapse(obs, actions, rewards, next_obs, terminated, truncated)
            if collapsed is None:
                return
            obs, actions, rewards, next_obs, terminated, truncated = collapsed

        lo = self.row * self.num_envs
        hi = lo + self.num_envs
        self.obs[lo:hi] = obs
        self.actions[lo:hi] = actions
        self.rewards[lo:hi] = rewards
        self.next_obs[lo:hi] = next_obs
        self.terminated[lo:hi] = terminated
        self.truncated[lo:hi] = truncated
        self.row = (self.row + 1) % self.rows
        self.size = min(self.size + 1, self.rows)

    def sample(self, batch_size: int) -> Batch:
        return self._gather(self.rng.integers(0, self.size * self.num_envs, size=batch_size))

    def _gather(self, idx: np.ndarray, extras: Optional[Dict[str, Any]] = None) -> Batch:
        return Batch(
            obs=self.obs[idx],
            actions=self.actions[idx],
            rewards=self.rewards[idx],
            next_obs=self.next_obs[idx],
            terminated=self.terminated[idx],
            truncated=self.truncated[idx],
            extras=extras or {},
        )

    def state_dict(self) -> Dict[str, Any]:
        n = self.size * self.num_envs
        return {
            "obs": self.obs[:n],
            "next_obs": self.next_obs[:n],
            "actions": self.actions[:n],
            "rewards": self.rewards[:n],
            "terminated": self.terminated[:n],
            "truncated": self.truncated[:n],
            "row": self.row,
            "size": self.size,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.size = int(state["size"])
        self.row = int(state["row"])
        n = self.size * self.num_envs
        self.obs[:n] = state["obs"]
        self.next_obs[:n] = state["next_obs"]
        self.actions[:n] = state["actions"]
        self.rewards[:n] = state["rewards"]
        self.terminated[:n] = state["terminated"]
        self.truncated[:n] = state["truncated"]


class SumTree:
    def __init__(self, capacity: int):
        self.capacity = 1 << (int(capacity) - 1).bit_length()
        self.tree = np.zeros(2 * self.capacity, dtype=np.float64)

    def update(self, idx, values) -> None:
        pos = np.asarray(idx, dtype=np.int64) + self.capacity
        self.tree[pos] = values
        parents = np.unique(pos >> 1)
        tree = self.tree
        while parents.size > 1:
            left = parents << 1
            tree[parents] = tree[left] + tree[left + 1]
            parents >>= 1
            keep = np.empty(parents.size, dtype=bool)
            keep[0] = True
            np.not_equal(parents[1:], parents[:-1], out=keep[1:])
            parents = parents[keep]
        node = int(parents[0])
        while node >= 1:
            tree[node] = tree[node << 1] + tree[(node << 1) + 1]
            node >>= 1

    @property
    def total(self) -> float:
        return float(self.tree[1])

    def sample(self, values: np.ndarray) -> np.ndarray:
        idx = np.ones(len(values), dtype=np.int64)
        values = values.copy()
        tree = self.tree
        while idx[0] < self.capacity:
            left = idx << 1
            left_sum = tree[left]
            go_right = values > left_sum
            values -= np.where(go_right, left_sum, 0.0)
            idx = left + go_right
        return idx - self.capacity


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, *args, alpha: float = 0.6, eps: float = 1e-6, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.eps = eps
        self.tree = SumTree(self.capacity)
        self.max_priority = 1.0
        self._new_slots = np.arange(self.num_envs, dtype=np.int64)

    def add(self, *args) -> None:
        row = self.row
        size = self.size
        super().add(*args)
        if self.row == row and self.size == size:
            return
        slots = self._new_slots + row * self.num_envs
        self.tree.update(slots, np.full(self.num_envs, self.max_priority**self.alpha))

    def sample(self, batch_size: int, beta: float = 0.4) -> Batch:
        total = self.tree.total
        segment = total / batch_size
        offsets = (np.arange(batch_size) + self.rng.random(batch_size)) * segment
        slots = self.tree.sample(offsets)
        probs = self.tree.tree[slots + self.tree.capacity] / max(total, 1e-12)
        weights = (self.size * self.num_envs * np.maximum(probs, 1e-12)) ** (-beta)
        weights /= weights.max()
        return self._gather(
            slots,
            extras={"weights": weights.astype(np.float32), "indices": slots},
        )

    def update_priorities(self, indices, td_errors) -> None:
        p = np.abs(np.asarray(td_errors, dtype=np.float64)) + self.eps
        self.tree.update(indices, p**self.alpha)
        self.max_priority = max(self.max_priority, float(p.max()))


class RolloutBuffer:
    def __init__(
        self,
        horizon: int,
        num_envs: int,
        obs_space: Space,
        action_space: Space,
        gamma: float = 0.99,
        lam: float = 0.95,
    ):
        self.horizon = horizon
        self.num_envs = num_envs
        self.gamma = gamma
        self.lam = lam
        shape = (horizon, num_envs)
        adtype = np.int64 if action_space.discrete else np.float32
        self.obs = np.zeros((*shape, *obs_space.shape), dtype=obs_space.dtype)
        self.actions = np.zeros((*shape, *action_space.shape), dtype=adtype)
        self.rewards = np.zeros(shape, dtype=np.float32)
        self.values = np.zeros(shape, dtype=np.float32)
        self.logprobs = np.zeros(shape, dtype=np.float32)
        self.terminated = np.zeros(shape, dtype=bool)
        self.truncated = np.zeros(shape, dtype=bool)
        self.bootstrap_values = np.zeros(shape, dtype=np.float32)
        self.advantages = np.zeros(shape, dtype=np.float32)
        self.returns = np.zeros(shape, dtype=np.float32)
        self.t = 0

    @property
    def full(self) -> bool:
        return self.t >= self.horizon

    def reset(self) -> None:
        self.t = 0

    def add(
        self,
        obs,
        actions,
        rewards,
        values,
        logprobs,
        terminated,
        truncated,
        bootstrap_values=None,
    ) -> None:
        t = self.t
        self.obs[t] = obs
        self.actions[t] = actions
        self.rewards[t] = rewards
        self.values[t] = values
        self.logprobs[t] = logprobs
        self.terminated[t] = terminated
        self.truncated[t] = truncated
        if bootstrap_values is not None:
            self.bootstrap_values[t] = bootstrap_values
        self.t += 1

    def compute_gae(self, last_values: np.ndarray) -> None:
        nonterminal = (~self.terminated).astype(np.float32)
        cont = nonterminal * (~self.truncated).astype(np.float32)
        next_values = np.empty_like(self.values)
        next_values[:-1] = self.values[1:]
        next_values[-1] = last_values
        np.copyto(next_values, self.bootstrap_values, where=self.truncated)
        deltas = self.rewards + self.gamma * next_values * nonterminal - self.values
        decay = cont * (self.gamma * self.lam)
        adv = np.zeros(self.num_envs, dtype=np.float32)
        advantages = self.advantages
        for t in range(self.horizon - 1, -1, -1):
            adv = deltas[t] + decay[t] * adv
            advantages[t] = adv
        np.add(self.advantages, self.values, out=self.returns)

    def flat(self) -> Dict[str, np.ndarray]:
        n = self.horizon * self.num_envs
        return {
            "obs": self.obs.reshape(n, *self.obs.shape[2:]),
            "actions": self.actions.reshape(n, *self.actions.shape[2:]),
            "logprobs": self.logprobs.reshape(n),
            "values": self.values.reshape(n),
            "advantages": self.advantages.reshape(n),
            "returns": self.returns.reshape(n),
        }

    def minibatches(self, batch_size: int, rng: np.random.Generator, epochs: int = 1):
        data = self.flat()
        n = self.horizon * self.num_envs
        for _ in range(epochs):
            order = rng.permutation(n)
            for start in range(0, n, batch_size):
                idx = order[start : start + batch_size]
                yield {k: v[idx] for k, v in data.items()}
