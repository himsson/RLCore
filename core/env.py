from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .types import Action, Obs, Space


class Env(ABC):
    observation_space: Space
    action_space: Space
    metadata: Dict[str, Any] = {}

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> Tuple[Obs, Dict[str, Any]]:
        ...

    @abstractmethod
    def step(self, action: Action) -> Tuple[Obs, float, bool, bool, Dict[str, Any]]:
        ...

    def render(self) -> Any:
        raise NotImplementedError

    def close(self) -> None:
        pass

    @property
    def unwrapped(self) -> "Env":
        return self


class Wrapper(Env):
    def __init__(self, env: Env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.metadata = env.metadata

    def reset(self, seed: Optional[int] = None):
        return self.env.reset(seed)

    def step(self, action: Action):
        return self.env.step(action)

    def render(self):
        return self.env.render()

    def close(self) -> None:
        self.env.close()

    @property
    def unwrapped(self) -> Env:
        return self.env.unwrapped

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.env, name)


class ObservationWrapper(Wrapper):
    def observation(self, obs: Obs) -> Obs:
        raise NotImplementedError

    def reset(self, seed: Optional[int] = None):
        obs, info = self.env.reset(seed)
        return self.observation(obs), info

    def step(self, action: Action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if "final_obs" in info:
            info = dict(info)
            info["final_obs"] = self.observation(info["final_obs"])
        return self.observation(obs), reward, terminated, truncated, info


class ActionWrapper(Wrapper):
    def action(self, action: Action) -> Action:
        raise NotImplementedError

    def step(self, action: Action):
        return self.env.step(self.action(action))


class RewardWrapper(Wrapper):
    def reward(self, reward: float) -> float:
        raise NotImplementedError

    def step(self, action: Action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, self.reward(reward), terminated, truncated, info


class TimeLimit(Wrapper):
    def __init__(self, env: Env, max_steps: int):
        super().__init__(env)
        self.max_steps = max_steps
        self._t = 0

    def reset(self, seed: Optional[int] = None):
        self._t = 0
        return self.env.reset(seed)

    def step(self, action: Action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._t += 1
        if self._t >= self.max_steps and not terminated:
            truncated = True
        return obs, reward, terminated, truncated, info


class EpisodeStats(Wrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        self._ret = 0.0
        self._len = 0

    def reset(self, seed: Optional[int] = None):
        self._ret = 0.0
        self._len = 0
        return self.env.reset(seed)

    def step(self, action: Action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._ret += float(reward)
        self._len += 1
        if terminated or truncated:
            info = dict(info)
            info["episode"] = {"return": self._ret, "length": self._len}
        return obs, reward, terminated, truncated, info


class FrameStack(ObservationWrapper):
    def __init__(self, env: Env, k: int):
        super().__init__(env)
        self.k = k
        space = env.observation_space
        self.observation_space = Space.box(
            (k, *space.shape),
            low=np.broadcast_to(space.low, (k, *space.shape)) if space.low is not None else -np.inf,
            high=np.broadcast_to(space.high, (k, *space.shape)) if space.high is not None else np.inf,
            dtype=space.dtype,
        )
        self.frames: deque = deque(maxlen=k)

    def observation(self, obs: Obs) -> Obs:
        if not self.frames:
            for _ in range(self.k):
                self.frames.append(obs)
        else:
            self.frames.append(obs)
        return np.stack(self.frames)

    def reset(self, seed: Optional[int] = None):
        self.frames.clear()
        return super().reset(seed)


class RescaleAction(ActionWrapper):
    def __init__(self, env: Env, low: float = -1.0, high: float = 1.0):
        super().__init__(env)
        space = env.action_space
        if space.discrete:
            raise TypeError("RescaleAction requires a continuous action space")
        self.src_low = np.asarray(space.low, dtype=np.float32)
        self.src_high = np.asarray(space.high, dtype=np.float32)
        self.action_space = Space.box(space.shape, low, high, np.float32)
        self.low = low
        self.high = high

    def action(self, action: Action) -> Action:
        action = np.clip(action, self.low, self.high)
        frac = (action - self.low) / (self.high - self.low)
        return self.src_low + frac * (self.src_high - self.src_low)


class ClipAction(ActionWrapper):
    def action(self, action: Action) -> Action:
        space = self.env.action_space
        if space.low is None or space.high is None:
            return action
        return np.clip(action, space.low, space.high)
