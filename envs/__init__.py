from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from core.env import Env, EpisodeStats, FrameStack, TimeLimit
from core.registry import ENVS, autoload
from core.vec import AsyncVectorEnv, SyncVectorEnv, VecEnv, VecNormalize

autoload("envs")


@dataclass(frozen=True)
class EnvFactory:
    name: str
    kwargs: Dict[str, Any] = field(default_factory=dict)
    max_episode_steps: int = 0
    frame_stack: int = 0

    def __call__(self) -> Env:
        autoload("envs")
        env = ENVS.get(self.name)(**self.kwargs)
        if self.max_episode_steps:
            env = TimeLimit(env, self.max_episode_steps)
        if self.frame_stack:
            env = FrameStack(env, self.frame_stack)
        return EpisodeStats(env)


def make_env(name: str, **kwargs: Any) -> EnvFactory:
    return EnvFactory(name=name, kwargs=kwargs.pop("env_kwargs", {}), **kwargs)


def make_vec_env(
    name: str,
    num_envs: int = 1,
    env_kwargs: Dict[str, Any] | None = None,
    max_episode_steps: int = 0,
    frame_stack: int = 0,
    asynchronous: bool = False,
    autoreset_mode: str = "same-step",
    norm_obs: bool = False,
    norm_reward: bool = False,
    clip_obs: float = 10.0,
    clip_reward: float = 10.0,
    gamma: float = 0.99,
) -> VecEnv:
    factories: List[EnvFactory] = [
        EnvFactory(
            name=name,
            kwargs=dict(env_kwargs or {}),
            max_episode_steps=max_episode_steps,
            frame_stack=frame_stack,
        )
        for _ in range(num_envs)
    ]
    cls = AsyncVectorEnv if asynchronous and num_envs > 1 else SyncVectorEnv
    venv: VecEnv = cls(factories, autoreset_mode=autoreset_mode)
    if norm_obs or norm_reward:
        venv = VecNormalize(
            venv,
            norm_obs=norm_obs,
            norm_reward=norm_reward,
            clip_obs=clip_obs,
            clip_reward=clip_reward,
            gamma=gamma,
        )
    return venv
