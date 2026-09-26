from __future__ import annotations

import pickle
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


AUTHOR = "himsson"
FORMAT_VERSION = 1


def _torch():
    try:
        import torch

        return torch
    except ImportError:
        return None


def save_checkpoint(path: str | Path, payload: Dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"author": AUTHOR, "format": FORMAT_VERSION, **payload}
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch = _torch()
    if torch is not None:
        torch.save(payload, tmp)
    else:
        tmp.write_bytes(pickle.dumps(payload))
    tmp.replace(path)
    return path


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> Dict[str, Any]:
    path = Path(path)
    torch = _torch()
    if torch is not None:
        return torch.load(path, map_location=map_location, weights_only=False)
    return pickle.loads(path.read_bytes())


def latest_checkpoint(run_dir: str | Path, pattern: str = "ckpt_*.pt") -> Optional[Path]:
    files = sorted(Path(run_dir).glob(pattern), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def global_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {"python": random.getstate(), "numpy": np.random.get_state()}
    torch = _torch()
    if torch is not None:
        state["torch"] = torch.get_rng_state()
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def load_global_rng_state(state: Dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch = _torch()
    if torch is not None and "torch" in state:
        torch.set_rng_state(state["torch"].cpu().to(torch.uint8))
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu().to(torch.uint8) for s in state["cuda"]])
