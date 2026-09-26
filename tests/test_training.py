from __future__ import annotations

from pathlib import Path

import numpy as np

from algos import make_algo
from core import Logger, RunConfig, Runner
from core.checkpoint import latest_checkpoint
from envs import make_vec_env


def build(tmp_path: Path, **overrides) -> Runner:
    cfg = RunConfig(
        env="gridworld",
        algo="random",
        num_envs=2,
        total_steps=400,
        max_episode_steps=20,
        log_interval=200,
        eval_interval=200,
        eval_episodes=3,
        save_interval=200,
        run_dir=str(tmp_path),
        **overrides,
    )
    env = make_vec_env("gridworld", num_envs=cfg.num_envs, max_episode_steps=cfg.max_episode_steps)
    eval_env = make_vec_env("gridworld", num_envs=1, max_episode_steps=cfg.max_episode_steps)
    algo = make_algo("random", env.observation_space, env.action_space, {"seed": 0, "num_envs": 2})
    logger = Logger(tmp_path / "run", stdout=False)
    return Runner(env, algo, cfg, logger, eval_env)


def test_runner_trains_and_logs(tmp_path):
    runner = build(tmp_path)
    row = runner.run()
    runner.logger.close()
    assert runner.global_step >= 400
    assert row["episodes"] > 0
    assert (tmp_path / "run" / "metrics.jsonl").exists()


def test_checkpoint_resume_restores_step(tmp_path):
    runner = build(tmp_path)
    runner.run()
    runner.logger.close()
    path = latest_checkpoint(tmp_path / "run")
    assert path is not None

    fresh = build(tmp_path / "second")
    fresh.load(path)
    assert fresh.global_step == runner.global_step
    fresh.logger.close()


def test_evaluate_returns_stats(tmp_path):
    runner = build(tmp_path)
    runner._obs, _ = runner.env.reset(0)
    stats = runner.evaluate()
    runner.logger.close()
    assert "eval/return" in stats
    assert np.isfinite(stats["eval/return"])
