from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


class Callback:
    def on_train_start(self, runner) -> None:
        pass

    def on_rollout_step(self, runner, infos: List[Dict[str, Any]]) -> None:
        pass

    def on_episode_end(self, runner, stats: Dict[str, float]) -> None:
        pass

    def on_update(self, runner, metrics: Dict[str, float]) -> None:
        pass

    def on_evaluate(self, runner, stats: Dict[str, float]) -> None:
        pass

    def on_train_end(self, runner) -> None:
        pass


class CallbackList(Callback):
    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.callbacks = list(callbacks or [])

    def append(self, callback: Callback) -> None:
        self.callbacks.append(callback)

    def on_train_start(self, runner) -> None:
        for cb in self.callbacks:
            cb.on_train_start(runner)

    def on_rollout_step(self, runner, infos) -> None:
        for cb in self.callbacks:
            cb.on_rollout_step(runner, infos)

    def on_episode_end(self, runner, stats) -> None:
        for cb in self.callbacks:
            cb.on_episode_end(runner, stats)

    def on_update(self, runner, metrics) -> None:
        for cb in self.callbacks:
            cb.on_update(runner, metrics)

    def on_evaluate(self, runner, stats) -> None:
        for cb in self.callbacks:
            cb.on_evaluate(runner, stats)

    def on_train_end(self, runner) -> None:
        for cb in self.callbacks:
            cb.on_train_end(runner)


class Throughput(Callback):
    def __init__(self, window: float = 5.0):
        self.window = window
        self._last_t = 0.0
        self._last_step = 0

    def on_train_start(self, runner) -> None:
        self._last_t = time.perf_counter()
        self._last_step = runner.global_step

    def on_rollout_step(self, runner, infos) -> None:
        now = time.perf_counter()
        dt = now - self._last_t
        if dt >= self.window:
            runner.logger.add("perf/sps", (runner.global_step - self._last_step) / dt)
            self._last_t = now
            self._last_step = runner.global_step


class BestCheckpoint(Callback):
    def __init__(self, metric: str = "eval/return", mode: str = "max"):
        self.metric = metric
        self.mode = mode
        self.best = -np.inf if mode == "max" else np.inf

    def on_evaluate(self, runner, stats) -> None:
        if self.metric not in stats:
            return
        value = stats[self.metric]
        better = value > self.best if self.mode == "max" else value < self.best
        if better:
            self.best = value
            runner.save(Path(runner.logger.run_dir) / "best.pt", extra={self.metric: value})


class EarlyStop(Callback):
    def __init__(self, metric: str = "eval/return", threshold: float = np.inf, patience: int = 0):
        self.metric = metric
        self.threshold = threshold
        self.patience = patience
        self.best = -np.inf
        self.waits = 0

    def on_evaluate(self, runner, stats) -> None:
        if self.metric not in stats:
            return
        value = stats[self.metric]
        if value >= self.threshold:
            runner.stop = True
            return
        if value > self.best:
            self.best = value
            self.waits = 0
        elif self.patience:
            self.waits += 1
            if self.waits >= self.patience:
                runner.stop = True
