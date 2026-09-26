from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from algos import make_algo
from core.checkpoint import load_checkpoint
from core.evaluation import evaluate
from core.seeding import resolve_device
from envs import make_vec_env


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--fps", type=float, default=10.0)
    args = parser.parse_args()

    path = Path(args.checkpoint)
    payload = load_checkpoint(path, map_location="cpu")
    cfg = payload["config"]
    device = resolve_device(cfg.get("device"))

    venv = make_vec_env(
        cfg["env"],
        num_envs=1,
        env_kwargs=cfg.get("env_kwargs", {}),
        max_episode_steps=cfg.get("max_episode_steps", 0),
        frame_stack=cfg.get("frame_stack", 0),
        norm_obs=cfg.get("norm_obs", False),
        gamma=cfg.get("gamma", 0.99),
    )
    venv.load_state_dict(payload.get("env", {}))

    algo_cfg = dict(cfg.get("algo_kwargs", {}))
    algo_cfg.update({"seed": cfg.get("seed", 0), "device": device, "num_envs": 1})
    algo = make_algo(cfg["algo"], venv.observation_space, venv.action_space, algo_cfg)
    algo.load_state_dict(payload["algo"])
    if hasattr(algo, "eval"):
        algo.eval()

    if args.render:
        env = venv.unwrapped.envs[0]
        obs, _ = venv.reset(cfg.get("seed", 0))
        for _ in range(args.episodes):
            done = False
            while not done:
                print(env.render())
                obs, _, term, trunc, _ = venv.step(algo.act(obs, deterministic=True))
                done = bool(term[0] or trunc[0])
                time.sleep(1.0 / max(args.fps, 1e-3))
    else:
        stats = evaluate(
            venv,
            lambda obs: algo.act(obs, deterministic=True),
            args.episodes,
            seed=cfg.get("seed", 0),
        )
        print(json.dumps(stats, indent=2))

    venv.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
