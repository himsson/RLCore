from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

from algos import make_algo
from core import (
    BestCheckpoint,
    Logger,
    RunConfig,
    Runner,
    Throughput,
    dump_config,
    latest_checkpoint,
    load_config,
    parse_overrides,
    seed_everything,
)
from core import __author__, __version__
from core.seeding import resolve_device, set_threads
from envs import make_vec_env


def build_run_dir(cfg: RunConfig) -> Path:
    name = cfg.run_name or f"{cfg.algo}-{cfg.env}-s{cfg.seed}-{time.strftime('%m%d-%H%M%S')}"
    path = Path(cfg.run_dir) / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def build(cfg: RunConfig, run_dir: Path) -> Runner:
    env = make_vec_env(
        cfg.env,
        num_envs=cfg.num_envs,
        env_kwargs=cfg.env_kwargs,
        max_episode_steps=cfg.max_episode_steps,
        frame_stack=cfg.frame_stack,
        asynchronous=cfg.async_envs,
        autoreset_mode=cfg.autoreset_mode,
        norm_obs=cfg.norm_obs,
        norm_reward=cfg.norm_reward,
        clip_obs=cfg.clip_obs,
        clip_reward=cfg.clip_reward,
        gamma=cfg.gamma,
    )
    eval_env = None
    if cfg.eval_interval and cfg.eval_episodes:
        eval_env = make_vec_env(
            cfg.env,
            num_envs=cfg.eval_envs,
            env_kwargs=cfg.env_kwargs,
            max_episode_steps=cfg.max_episode_steps,
            frame_stack=cfg.frame_stack,
            autoreset_mode=cfg.autoreset_mode,
            norm_obs=cfg.norm_obs,
            norm_reward=False,
            clip_obs=cfg.clip_obs,
            gamma=cfg.gamma,
        )

    algo_cfg = dict(cfg.algo_kwargs)
    algo_cfg.setdefault("seed", cfg.seed)
    algo_cfg.setdefault("device", cfg.device)
    algo_cfg.setdefault("num_envs", cfg.num_envs)
    algo_cfg.setdefault("gamma", cfg.gamma)
    algo_cfg.setdefault("total_steps", cfg.total_steps)
    algo = make_algo(cfg.algo, env.observation_space, env.action_space, algo_cfg)

    logger = Logger(
        run_dir,
        window=cfg.log_window,
        csv_out=cfg.csv,
        tensorboard=cfg.tensorboard,
    )
    callbacks = [Throughput()]
    if cfg.keep_best:
        callbacks.append(BestCheckpoint())
    return Runner(env, algo, cfg, logger, eval_env, callbacks)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rlcore",
        epilog="RLCore " + __version__ + " · " + __author__ + " · MIT",
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--set", nargs="*", default=[])
    parser.add_argument("--resume", default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config, parse_overrides(args.set))
    if args.resume:
        cfg.resume = args.resume
    cfg.device = resolve_device(cfg.device)
    set_threads(cfg.torch_threads)
    seed_everything(cfg.seed, deterministic=cfg.deterministic)

    run_dir = build_run_dir(cfg)
    dump_config(cfg, run_dir / "config.json")
    runner = build(cfg, run_dir)

    if cfg.resume:
        path = cfg.resume
        if Path(path).is_dir():
            path = latest_checkpoint(path)
        if path:
            runner.load(path)

    try:
        runner.run()
    finally:
        runner.save(run_dir / "final.pt")
        runner.logger.close()
        runner.env.close()
        if runner.eval_env is not None:
            runner.eval_env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
