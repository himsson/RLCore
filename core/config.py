from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class RunConfig:
    env: str = "gridworld"
    env_kwargs: Dict[str, Any] = field(default_factory=dict)
    algo: str = "random"
    algo_kwargs: Dict[str, Any] = field(default_factory=dict)

    num_envs: int = 1
    async_envs: bool = False
    autoreset_mode: str = "same-step"
    max_episode_steps: int = 0
    frame_stack: int = 0

    norm_obs: bool = False
    norm_reward: bool = False
    clip_obs: float = 10.0
    clip_reward: float = 10.0
    gamma: float = 0.99

    total_steps: int = 100_000
    seed: int = 0
    device: str = "auto"
    torch_threads: int = 0
    deterministic: bool = False

    log_interval: int = 1000
    log_window: int = 100
    csv: bool = False
    tensorboard: bool = False

    eval_interval: int = 10_000
    eval_episodes: int = 10
    eval_envs: int = 1

    save_interval: int = 0
    keep_best: bool = True
    resume: Optional[str] = None

    run_dir: str = "runs"
    run_name: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    def replace(self, **kwargs: Any) -> "RunConfig":
        data = asdict(self)
        data.update(kwargs)
        return _coerce(RunConfig, data)


def _coerce(cls, data: Dict[str, Any]):
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise KeyError(f"unknown config keys: {sorted(unknown)}")
    return cls(**data)


def _read(path: Path) -> Dict[str, Any]:
    if path.suffix in (".yaml", ".yml"):
        import yaml

        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return json.loads(path.read_text(encoding="utf-8"))


def _deep_update(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path, overrides: Optional[Dict[str, Any]] = None) -> RunConfig:
    path = Path(path)
    raw = _read(path)
    parent = raw.pop("defaults", None)
    if parent:
        base = _read(path.parent / parent)
        raw = _deep_update(base, raw)
    if overrides:
        _deep_update(raw, overrides)
    return _coerce(RunConfig, raw)


def dump_config(cfg: RunConfig, path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(asdict(cfg), indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )


def parse_overrides(pairs: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        try:
            parsed: Any = json.loads(value)
        except json.JSONDecodeError:
            parsed = value
        node = out
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = parsed
    return out
