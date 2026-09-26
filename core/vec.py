from __future__ import annotations

import multiprocessing as mp
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .env import Env
from .running import RunningMeanStd, RunningReturn
from .types import Space

SAME_STEP = "same-step"
NEXT_STEP = "next-step"
AUTORESET_MODES = (SAME_STEP, NEXT_STEP)


def check_autoreset_mode(mode: str) -> str:
    if mode not in AUTORESET_MODES:
        raise ValueError(f"unknown autoreset_mode {mode!r}; expected one of {AUTORESET_MODES}")
    return mode

VecStep = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]


class VecEnv(ABC):
    num_envs: int
    observation_space: Space
    action_space: Space
    autoreset_mode: str = SAME_STEP

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        ...

    @abstractmethod
    def step(self, actions) -> VecStep:
        ...

    def close(self) -> None:
        pass

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        pass

    @property
    def unwrapped(self) -> "VecEnv":
        return self


class SyncVectorEnv(VecEnv):
    def __init__(self, factories: Sequence[Callable[[], Env]], autoreset_mode: str = SAME_STEP):
        self.envs = [f() for f in factories]
        self.num_envs = len(self.envs)
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        self.autoreset_mode = check_autoreset_mode(autoreset_mode)
        self._needs_reset = np.zeros(self.num_envs, dtype=bool)

    def reset(self, seed: Optional[int] = None):
        obs, infos = [], []
        for i, env in enumerate(self.envs):
            o, info = env.reset(None if seed is None else seed + i)
            obs.append(o)
            infos.append(info)
        self._needs_reset[:] = False
        return np.stack(obs), infos

    def step(self, actions) -> VecStep:
        obs, rewards, term, trunc, infos = [], [], [], [], []
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            if self.autoreset_mode == NEXT_STEP and self._needs_reset[i]:
                o, info = env.reset()
                self._needs_reset[i] = False
                obs.append(o)
                rewards.append(0.0)
                term.append(False)
                trunc.append(False)
                infos.append({**info, 'reset': True})
                continue
            o, r, t, tr, info = env.step(action)
            if t or tr:
                if self.autoreset_mode == SAME_STEP:
                    info = dict(info)
                    info["final_obs"] = o
                    o, reset_info = env.reset()
                    info["reset_info"] = reset_info
                else:
                    self._needs_reset[i] = True
            obs.append(o)
            rewards.append(r)
            term.append(t)
            trunc.append(tr)
            infos.append(info)
        return (
            np.stack(obs),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(term, dtype=bool),
            np.asarray(trunc, dtype=bool),
            infos,
        )

    def close(self) -> None:
        for env in self.envs:
            env.close()


def _worker(remote, parent_remote, factory, autoreset_mode):
    parent_remote.close()
    env = factory()
    needs_reset = False
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                if autoreset_mode == NEXT_STEP and needs_reset:
                    obs, info = env.reset()
                    needs_reset = False
                    remote.send((obs, 0.0, False, False, {**info, 'reset': True}))
                    continue
                obs, reward, terminated, truncated, info = env.step(data)
                if terminated or truncated:
                    if autoreset_mode == SAME_STEP:
                        info = dict(info)
                        info["final_obs"] = obs
                        obs, reset_info = env.reset()
                        info["reset_info"] = reset_info
                    else:
                        needs_reset = True
                remote.send((obs, reward, terminated, truncated, info))
            elif cmd == "reset":
                needs_reset = False
                remote.send(env.reset(data))
            elif cmd == "spaces":
                remote.send((env.observation_space, env.action_space))
            elif cmd == "render":
                remote.send(env.render())
            elif cmd == "close":
                env.close()
                remote.close()
                break
    except (KeyboardInterrupt, EOFError):
        env.close()


class AsyncVectorEnv(VecEnv):
    def __init__(
        self,
        factories: Sequence[Callable[[], Env]],
        autoreset_mode: str = SAME_STEP,
        start_method: str = "spawn",
    ):
        ctx = mp.get_context(start_method)
        self.num_envs = len(factories)
        self.autoreset_mode = check_autoreset_mode(autoreset_mode)
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.num_envs)])
        self.processes = []
        for remote, work_remote, factory in zip(self.remotes, self.work_remotes, factories):
            proc = ctx.Process(
                target=_worker,
                args=(work_remote, remote, factory, autoreset_mode),
                daemon=True,
            )
            proc.start()
            self.processes.append(proc)
            work_remote.close()
        self.remotes[0].send(("spaces", None))
        self.observation_space, self.action_space = self.remotes[0].recv()
        self.closed = False

    def reset(self, seed: Optional[int] = None):
        for i, remote in enumerate(self.remotes):
            remote.send(("reset", None if seed is None else seed + i))
        results = [remote.recv() for remote in self.remotes]
        obs, infos = zip(*results)
        return np.stack(obs), list(infos)

    def step(self, actions) -> VecStep:
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        results = [remote.recv() for remote in self.remotes]
        obs, rewards, term, trunc, infos = zip(*results)
        return (
            np.stack(obs),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(term, dtype=bool),
            np.asarray(trunc, dtype=bool),
            list(infos),
        )

    def close(self) -> None:
        if self.closed:
            return
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for proc in self.processes:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
        self.closed = True


class VecWrapper(VecEnv):
    def __init__(self, venv: VecEnv):
        self.venv = venv
        self.num_envs = venv.num_envs
        self.observation_space = venv.observation_space
        self.action_space = venv.action_space
        self.autoreset_mode = venv.autoreset_mode

    def reset(self, seed: Optional[int] = None):
        return self.venv.reset(seed)

    def step(self, actions) -> VecStep:
        return self.venv.step(actions)

    def close(self) -> None:
        self.venv.close()

    @property
    def unwrapped(self) -> VecEnv:
        return self.venv.unwrapped

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.venv, name)


class VecNormalize(VecWrapper):
    def __init__(
        self,
        venv: VecEnv,
        norm_obs: bool = True,
        norm_reward: bool = True,
        clip_obs: float = 10.0,
        clip_reward: float = 10.0,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
    ):
        super().__init__(venv)
        self.norm_obs = norm_obs
        self.norm_reward = norm_reward
        self.clip_obs = clip_obs
        self.clip_reward = clip_reward
        self.epsilon = epsilon
        self.training = True
        self.obs_rms = RunningMeanStd(shape=self.observation_space.shape)
        self.ret_rms = RunningReturn(self.num_envs, gamma)

    def train(self, mode: bool = True) -> "VecNormalize":
        self.training = mode
        return self

    def eval(self) -> "VecNormalize":
        return self.train(False)

    def _obs(self, obs: np.ndarray, update: bool) -> np.ndarray:
        if not self.norm_obs:
            return obs
        if update and self.training:
            self.obs_rms.update(obs)
        return np.clip(self.obs_rms.normalize(obs, self.epsilon), -self.clip_obs, self.clip_obs)

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        return self._obs(np.asarray(obs, dtype=np.float32), update=False)

    def reset(self, seed: Optional[int] = None):
        obs, infos = self.venv.reset(seed)
        self.ret_rms.ret[:] = 0.0
        return self._obs(obs, update=True), infos

    def step(self, actions) -> VecStep:
        obs, rewards, term, trunc, infos = self.venv.step(actions)
        for info in infos:
            if "final_obs" in info:
                info["final_obs"] = self._obs(np.asarray(info["final_obs"])[None], False)[0]
        obs = self._obs(obs, update=True)
        if self.norm_reward:
            dones = term | trunc
            if self.training:
                rewards = self.ret_rms.update(rewards, dones)
            else:
                rewards = self.ret_rms.rms.normalize(rewards, self.epsilon)
            rewards = np.clip(rewards, -self.clip_reward, self.clip_reward).astype(np.float32)
        return obs, rewards, term, trunc, infos

    def state_dict(self) -> Dict[str, Any]:
        return {"obs_rms": self.obs_rms.state_dict(), "ret_rms": self.ret_rms.state_dict()}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.obs_rms.load_state_dict(state["obs_rms"])
        self.ret_rms.load_state_dict(state["ret_rms"])
