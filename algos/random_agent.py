from __future__ import annotations

from typing import Any, Dict

import numpy as np

from core.algo import Algorithm
from core.registry import ALGOS


@ALGOS.register("random")
class RandomAgent(Algorithm):
    name = "random"

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        return self.action_space.sample(self.rng, batch=len(obs))

    def observe(self, transition: Dict[str, Any]) -> None:
        pass

    def update(self) -> Dict[str, float]:
        return {}
