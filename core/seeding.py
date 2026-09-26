from __future__ import annotations

import os
import random
from typing import List, Optional

import numpy as np


def seed_everything(seed: int, deterministic: bool = False) -> np.random.Generator:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)
        else:
            torch.backends.cudnn.benchmark = True
    except ImportError:
        pass
    return np.random.default_rng(seed)


def spawn(rng: np.random.Generator, n: int) -> List[np.random.Generator]:
    if hasattr(rng, "spawn"):
        return list(rng.spawn(n))
    return [np.random.default_rng(int(rng.integers(0, 2**31 - 1))) for _ in range(n)]


def resolve_device(device: Optional[str]) -> str:
    if device and device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def set_threads(n: int) -> None:
    if n <= 0:
        return
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    try:
        import torch

        torch.set_num_threads(n)
    except ImportError:
        pass
