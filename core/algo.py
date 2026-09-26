from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np

from .types import Space


class Algorithm(ABC):
    name: str = "base"

    def __init__(self, obs_space: Space, action_space: Space, cfg: Dict[str, Any]):
        self.obs_space = obs_space
        self.action_space = action_space
        self.cfg = dict(cfg)
        self.device = cfg.get("device", "cpu")
        self.num_envs = int(cfg.get("num_envs", 1))
        self.seed = int(cfg.get("seed", 0))
        self.rng = np.random.default_rng(self.seed)
        self.step_count = 0
        self.progress = 0.0

    @abstractmethod
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        ...

    @abstractmethod
    def observe(self, transition: Dict[str, Any]) -> None:
        ...

    @abstractmethod
    def update(self) -> Dict[str, float]:
        ...

    def on_episode_end(self, stats: Dict[str, float]) -> None:
        pass

    def set_progress(self, step: int, total: int) -> None:
        self.step_count = step
        self.progress = min(step / max(total, 1), 1.0)

    def state_dict(self) -> Dict[str, Any]:
        return {"step_count": self.step_count}

    def load_state_dict(self, state: Dict[str, Any], strict: bool = True) -> None:
        self.step_count = state.get("step_count", 0)

    def modules(self) -> Dict[str, Any]:
        return {}

    def optimizers(self) -> Dict[str, Any]:
        return {}


class TorchAlgorithm(Algorithm):
    def state_dict(self) -> Dict[str, Any]:
        state = super().state_dict()
        state["modules"] = {k: m.state_dict() for k, m in self.modules().items()}
        state["optimizers"] = {k: o.state_dict() for k, o in self.optimizers().items()}
        return state

    def load_state_dict(self, state: Dict[str, Any], strict: bool = True) -> None:
        super().load_state_dict(state, strict)
        for key, module in self.modules().items():
            if key in state.get("modules", {}):
                module.load_state_dict(state["modules"][key])
        for key, opt in self.optimizers().items():
            if key in state.get("optimizers", {}):
                opt.load_state_dict(state["optimizers"][key])

    def train(self, mode: bool = True) -> None:
        for module in self.modules().values():
            module.train(mode)

    def eval(self) -> None:
        self.train(False)

    def to(self, device: str) -> "TorchAlgorithm":
        self.device = device
        for module in self.modules().values():
            module.to(device)
        return self

    def parameters(self, key: Optional[str] = None):
        modules = self.modules()
        targets = [modules[key]] if key else list(modules.values())
        for module in targets:
            yield from module.parameters()
