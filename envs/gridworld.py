from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from core.env import Env
from core.registry import ENVS
from core.types import Space

MOVES = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int64)


@ENVS.register("gridworld")
class GridWorld(Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, size: int = 8, n_traps: int = 5, step_penalty: float = 0.01):
        self.size = size
        self.n_traps = n_traps
        self.step_penalty = step_penalty
        self.observation_space = Space.box((4,), 0.0, 1.0)
        self.action_space = Space.categorical(4)
        self.rng = np.random.default_rng()
        self.agent = np.zeros(2, dtype=np.int64)
        self.goal = np.zeros(2, dtype=np.int64)
        self.traps: set[tuple[int, int]] = set()

    def _obs(self) -> np.ndarray:
        scale = max(self.size - 1, 1)
        return np.concatenate([self.agent, self.goal]).astype(np.float32) / scale

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        cells = self.rng.permutation(self.size * self.size)[: self.n_traps + 2]
        coords = [(int(c // self.size), int(c % self.size)) for c in cells]
        self.agent = np.array(coords[0], dtype=np.int64)
        self.goal = np.array(coords[1], dtype=np.int64)
        self.traps = set(coords[2:])
        return self._obs(), {}

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.agent = np.clip(self.agent + MOVES[int(action)], 0, self.size - 1)
        cell = (int(self.agent[0]), int(self.agent[1]))
        reward = -self.step_penalty
        terminated = False
        if np.array_equal(self.agent, self.goal):
            reward, terminated = 1.0, True
        elif cell in self.traps:
            reward, terminated = -1.0, True
        return self._obs(), reward, terminated, False, {}

    def render(self) -> str:
        grid = [["." for _ in range(self.size)] for _ in range(self.size)]
        for r, c in self.traps:
            grid[r][c] = "x"
        grid[self.goal[0]][self.goal[1]] = "G"
        grid[self.agent[0]][self.agent[1]] = "A"
        return "\n".join("".join(row) for row in grid)
