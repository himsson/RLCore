from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .vec import VecEnv, VecNormalize


def evaluate(
    venv: VecEnv,
    act: Callable[[np.ndarray], np.ndarray],
    episodes: int,
    seed: Optional[int] = None,
    max_steps: int = 1_000_000,
) -> Dict[str, float]:
    norm = _find_normalizer(venv)
    was_training = norm.training if norm is not None else None
    if norm is not None:
        norm.eval()

    n = venv.num_envs
    quota = np.full(n, episodes // n, dtype=np.int64)
    quota[: episodes % n] += 1
    done_count = np.zeros(n, dtype=np.int64)
    returns: List[float] = []
    lengths: List[float] = []

    obs, _ = venv.reset(seed)
    steps = 0
    while np.any(done_count < quota) and steps < max_steps:
        obs, _, _, _, infos = venv.step(act(obs))
        steps += 1
        for i, info in enumerate(infos):
            ep = info.get("episode")
            if ep and done_count[i] < quota[i]:
                done_count[i] += 1
                returns.append(ep["return"])
                lengths.append(ep["length"])

    if norm is not None and was_training:
        norm.train()

    if not returns:
        return {}
    return {
        "eval/return": float(np.mean(returns)),
        "eval/return_std": float(np.std(returns)),
        "eval/return_min": float(np.min(returns)),
        "eval/return_max": float(np.max(returns)),
        "eval/length": float(np.mean(lengths)),
        "eval/episodes": float(len(returns)),
    }


def _find_normalizer(venv: Any) -> Optional[VecNormalize]:
    while venv is not None:
        if isinstance(venv, VecNormalize):
            return venv
        venv = getattr(venv, "venv", None)
    return None
