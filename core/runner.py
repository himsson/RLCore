from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .algo import Algorithm
from .callbacks import Callback, CallbackList
from .checkpoint import global_rng_state, load_checkpoint, load_global_rng_state, save_checkpoint
from .config import RunConfig
from .evaluation import evaluate
from .logger import Logger
from .seeding import resolve_device
from .vec import NEXT_STEP, VecEnv, VecNormalize


class Runner:
    def __init__(
        self,
        env: VecEnv,
        algo: Algorithm,
        cfg: RunConfig,
        logger: Logger,
        eval_env: Optional[VecEnv] = None,
        callbacks: Optional[List[Callback]] = None,
    ):
        self.env = env
        self.algo = algo
        self.cfg = cfg
        self.logger = logger
        self.eval_env = eval_env
        self.callbacks = CallbackList(callbacks)
        self.global_step = 0
        self.episodes = 0
        self.stop = False
        self._obs: Optional[np.ndarray] = None

    def run(self) -> Dict[str, Any]:
        self.callbacks.on_train_start(self)
        if self._obs is None:
            self._obs, _ = self.env.reset(self.cfg.seed)

        next_log = self.global_step + self.cfg.log_interval
        next_eval = self.global_step + self.cfg.eval_interval
        next_save = self.global_step + self.cfg.save_interval

        while self.global_step < self.cfg.total_steps and not self.stop:
            self.collect_step()

            metrics = self.algo.update()
            if metrics:
                self.logger.add_many(metrics)
                self.callbacks.on_update(self, metrics)

            if self.cfg.log_interval and self.global_step >= next_log:
                self.logger.dump(self.global_step, {"episodes": self.episodes})
                next_log += self.cfg.log_interval

            if self.cfg.eval_interval and self.global_step >= next_eval:
                self.evaluate()
                next_eval += self.cfg.eval_interval

            if self.cfg.save_interval and self.global_step >= next_save:
                self.save()
                next_save += self.cfg.save_interval

        self.callbacks.on_train_end(self)
        final = self.logger.dump(self.global_step, {"episodes": self.episodes})
        return final

    def collect_step(self) -> None:
        obs = self._obs
        actions = self.algo.act(obs)
        next_obs, rewards, terminated, truncated, infos = self.env.step(actions)

        real_next = next_obs
        if any("final_obs" in info for info in infos):
            real_next = next_obs.copy()
            for i, info in enumerate(infos):
                if "final_obs" in info:
                    real_next[i] = info["final_obs"]

        mask = np.ones(self.env.num_envs, dtype=bool)
        if self.env.autoreset_mode == NEXT_STEP:
            for i, info in enumerate(infos):
                mask[i] = not info.get("reset", False)

        if mask.any():
            self.algo.observe(
                {
                    "obs": obs,
                    "actions": actions,
                    "rewards": rewards,
                    "next_obs": real_next,
                    "terminated": terminated,
                    "truncated": truncated,
                    "mask": mask,
                    "infos": infos,
                }
            )

        self._obs = next_obs
        self.global_step += self.env.num_envs
        self.algo.set_progress(self.global_step, self.cfg.total_steps)

        for info in infos:
            ep = info.get("episode")
            if ep:
                self.episodes += 1
                self.logger.add("ep/return", ep["return"])
                self.logger.add("ep/length", ep["length"])
                self.algo.on_episode_end(ep)
                self.callbacks.on_episode_end(self, ep)

        self.callbacks.on_rollout_step(self, infos)

    def evaluate(self) -> Dict[str, float]:
        if self.eval_env is None:
            return {}
        self._sync_normalizer()
        if hasattr(self.algo, "eval"):
            self.algo.eval()
        stats = evaluate(
            self.eval_env,
            lambda obs: self.algo.act(obs, deterministic=True),
            self.cfg.eval_episodes,
            seed=self.cfg.seed + 10_000,
        )
        if hasattr(self.algo, "train"):
            self.algo.train()
        if stats:
            self.logger.add_many(stats)
            self.callbacks.on_evaluate(self, stats)
        return stats

    def _sync_normalizer(self) -> None:
        source = _find(self.env, VecNormalize)
        target = _find(self.eval_env, VecNormalize)
        if source is not None and target is not None:
            target.load_state_dict(source.state_dict())
            target.eval()

    def save(self, path: Optional[str | Path] = None, extra: Optional[Dict[str, Any]] = None) -> Path:
        path = Path(path or Path(self.logger.run_dir) / f"ckpt_{self.global_step}.pt")
        payload = {
            "global_step": self.global_step,
            "episodes": self.episodes,
            "algo": self.algo.state_dict(),
            "env": self.env.state_dict(),
            "rng": global_rng_state(),
            "config": vars(self.cfg),
        }
        if extra:
            payload["extra"] = extra
        return save_checkpoint(path, payload)

    def load(self, path: str | Path, resume: bool = True) -> None:
        payload = load_checkpoint(path, map_location=resolve_device(self.cfg.device))
        self.algo.load_state_dict(payload["algo"])
        self.env.load_state_dict(payload.get("env", {}))
        if resume:
            self.global_step = payload.get("global_step", 0)
            self.episodes = payload.get("episodes", 0)
            if "rng" in payload:
                load_global_rng_state(payload["rng"])
            self.algo.set_progress(self.global_step, self.cfg.total_steps)


def _find(venv: Any, cls):
    while venv is not None:
        if isinstance(venv, cls):
            return venv
        venv = getattr(venv, "venv", None)
    return None
