from __future__ import annotations

import numpy as np
import pytest

from core.buffer import PrioritizedReplayBuffer, ReplayBuffer, RolloutBuffer
from core.running import RunningMeanStd
from core.schedules import make_schedule
from core.types import Space
from core.vec import NEXT_STEP, SAME_STEP, VecNormalize
from envs import make_vec_env

OBS = Space.box((4,), 0.0, 1.0)
ACT = Space.categorical(4)


def test_space_sample_in_bounds():
    rng = np.random.default_rng(0)
    box = Space.box((3,), -2.0, 2.0)
    assert box.contains(box.sample(rng, batch=16))
    assert ACT.contains(ACT.sample(rng, batch=16))


def test_running_mean_std_matches_numpy():
    rng = np.random.default_rng(0)
    data = rng.normal(3.0, 2.0, size=(500, 4))
    rms = RunningMeanStd(shape=(4,))
    for chunk in np.array_split(data, 10):
        rms.update(chunk)
    assert np.allclose(rms.mean, data.mean(axis=0), atol=1e-6)
    assert np.allclose(rms.var, data.var(axis=0), atol=1e-6)


def test_replay_roundtrip():
    buf = ReplayBuffer(1000, OBS, ACT, num_envs=2)
    for _ in range(50):
        buf.add(
            np.ones((2, 4), np.float32),
            np.zeros(2, np.int64),
            np.ones(2, np.float32),
            np.zeros((2, 4), np.float32),
            np.zeros(2, bool),
            np.zeros(2, bool),
        )
    batch = buf.sample(32)
    assert len(batch) == 32
    assert batch.obs.shape == (32, 4)


def test_nstep_discounts_and_cuts_on_done():
    buf = ReplayBuffer(100, OBS, ACT, num_envs=1, n_step=3, gamma=0.5)
    rewards = [1.0, 1.0, 1.0]
    for i, r in enumerate(rewards):
        buf.add(
            np.full((1, 4), i, np.float32),
            np.zeros(1, np.int64),
            np.full(1, r, np.float32),
            np.zeros((1, 4), np.float32),
            np.array([i == 1]),
            np.zeros(1, bool),
        )
    assert buf.size == 1
    assert buf.rewards[0] == pytest.approx(1.0 + 0.5)
    assert bool(buf.terminated[0])


def test_prioritized_weights_and_updates():
    buf = PrioritizedReplayBuffer(256, OBS, ACT, num_envs=1, rng=np.random.default_rng(0))
    for _ in range(64):
        buf.add(
            np.zeros((1, 4), np.float32),
            np.zeros(1, np.int64),
            np.zeros(1, np.float32),
            np.zeros((1, 4), np.float32),
            np.zeros(1, bool),
            np.zeros(1, bool),
        )
    batch = buf.sample(16, beta=0.4)
    assert batch.extras["weights"].max() == pytest.approx(1.0)
    buf.update_priorities(batch.extras["indices"], np.arange(16, dtype=np.float32))
    assert buf.tree.total > 0


def test_gae_matches_manual_case():
    buf = RolloutBuffer(3, 1, OBS, ACT, gamma=0.9, lam=0.8)
    for t in range(3):
        buf.add(
            np.zeros((1, 4), np.float32),
            np.zeros(1, np.int64),
            np.ones(1, np.float32),
            np.zeros(1, np.float32),
            np.zeros(1, np.float32),
            np.zeros(1, bool),
            np.zeros(1, bool),
        )
    buf.compute_gae(np.zeros(1, np.float32))
    expected = 1.0 + 0.72 * (1.0 + 0.72 * 1.0)
    assert buf.advantages[0, 0] == pytest.approx(expected, rel=1e-5)


def test_gae_stops_at_termination():
    buf = RolloutBuffer(2, 1, OBS, ACT, gamma=0.9, lam=1.0)
    buf.add(
        np.zeros((1, 4), np.float32),
        np.zeros(1, np.int64),
        np.ones(1, np.float32),
        np.zeros(1, np.float32),
        np.zeros(1, np.float32),
        np.ones(1, bool),
        np.zeros(1, bool),
    )
    buf.add(
        np.zeros((1, 4), np.float32),
        np.zeros(1, np.int64),
        np.ones(1, np.float32),
        np.zeros(1, np.float32),
        np.zeros(1, np.float32),
        np.zeros(1, bool),
        np.zeros(1, bool),
    )
    buf.compute_gae(np.zeros(1, np.float32))
    assert buf.advantages[0, 0] == pytest.approx(1.0)


def test_gae_bootstraps_on_truncation():
    buf = RolloutBuffer(1, 1, OBS, ACT, gamma=0.9, lam=1.0)
    buf.add(
        np.zeros((1, 4), np.float32),
        np.zeros(1, np.int64),
        np.zeros(1, np.float32),
        np.zeros(1, np.float32),
        np.zeros(1, np.float32),
        np.zeros(1, bool),
        np.ones(1, bool),
        bootstrap_values=np.full(1, 5.0, np.float32),
    )
    buf.compute_gae(np.zeros(1, np.float32))
    assert buf.advantages[0, 0] == pytest.approx(0.9 * 5.0)


def test_schedules():
    assert make_schedule(0.5)(0.9) == 0.5
    assert make_schedule("linear:1,0")(0.5) == pytest.approx(0.5)
    assert make_schedule({"type": "linear", "start": 1.0, "end": 0.0, "end_fraction": 0.5})(
        0.5
    ) == pytest.approx(0.0)


def test_vec_same_step_exposes_final_obs():
    venv = make_vec_env("gridworld", num_envs=2, max_episode_steps=3)
    venv.reset(0)
    seen = False
    for _ in range(20):
        _, _, term, trunc, infos = venv.step([0, 1])
        for info in infos:
            if "final_obs" in info:
                seen = True
                assert "episode" in info
    venv.close()
    assert seen


def test_vec_next_step_mode_emits_reset_step():
    venv = make_vec_env(
        "gridworld", num_envs=1, max_episode_steps=2, autoreset_mode=NEXT_STEP
    )
    venv.reset(0)
    dones = []
    for _ in range(6):
        _, reward, term, trunc, _ = venv.step([0])
        dones.append(bool(term[0] or trunc[0]))
    venv.close()
    assert any(dones)


def test_vec_normalize_state_roundtrip():
    venv = make_vec_env("gridworld", num_envs=2, norm_obs=True, norm_reward=True)
    obs, _ = venv.reset(0)
    for _ in range(20):
        obs, *_ = venv.step([0, 1])
    assert np.isfinite(obs).all()
    state = venv.state_dict()
    other = make_vec_env("gridworld", num_envs=2, norm_obs=True, norm_reward=True)
    other.load_state_dict(state)
    assert np.allclose(other.obs_rms.mean, venv.obs_rms.mean)
    venv.close()
    other.close()


def test_vec_normalize_eval_freezes_stats():
    venv = make_vec_env("gridworld", num_envs=1, norm_obs=True)
    venv.reset(0)
    for _ in range(10):
        venv.step([0])
    venv.eval()
    before = venv.obs_rms.mean.copy()
    for _ in range(10):
        venv.step([1])
    assert np.allclose(before, venv.obs_rms.mean)
    venv.close()


def test_next_step_mode_flags_reset_steps():
    venv = make_vec_env("gridworld", num_envs=1, max_episode_steps=2, autoreset_mode=NEXT_STEP)
    venv.reset(0)
    flagged = 0
    after_done = False
    for _ in range(8):
        _, reward, term, trunc, infos = venv.step([0])
        if after_done:
            assert infos[0].get("reset") is True
            assert reward[0] == 0.0
            flagged += 1
        after_done = bool(term[0] or trunc[0])
    venv.close()
    assert flagged > 0


def test_evaluate_quota_is_balanced_across_envs():
    from core.evaluation import evaluate

    venv = make_vec_env("gridworld", num_envs=4, max_episode_steps=20)
    rng = np.random.default_rng(0)
    stats = evaluate(venv, lambda obs: rng.integers(0, 4, size=len(obs)), episodes=8, seed=0)
    venv.close()
    assert stats["eval/episodes"] == 8


def test_unknown_autoreset_mode_raises():
    with pytest.raises(ValueError):
        make_vec_env("gridworld", num_envs=1, autoreset_mode='"next-step"')


def test_running_mean_std_scalar_matches_numpy():
    rng = np.random.default_rng(1)
    data = rng.normal(-2.0, 3.0, size=(400,))
    rms = RunningMeanStd(shape=())
    for chunk in np.array_split(data, 8):
        rms.update(chunk)
    assert float(rms.mean) == pytest.approx(float(data.mean()), abs=1e-6)
    assert float(rms.var) == pytest.approx(float(data.var()), abs=1e-6)


def test_checkpoint_carries_metadata(tmp_path):
    from core.checkpoint import AUTHOR, load_checkpoint, save_checkpoint

    path = save_checkpoint(tmp_path / "x.pt", {"global_step": 7})
    payload = load_checkpoint(path)
    assert payload["author"] == AUTHOR
    assert payload["format"] == 1
    assert payload["global_step"] == 7
