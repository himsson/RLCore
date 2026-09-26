from typing import Any, Dict

from core.algo import Algorithm
from core.registry import ALGOS, autoload
from core.types import Space

autoload("algos")


def make_algo(name: str, obs_space: Space, action_space: Space, cfg: Dict[str, Any]) -> Algorithm:
    return ALGOS.get(name)(obs_space, action_space, cfg)
