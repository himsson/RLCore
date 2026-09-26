from __future__ import annotations

import argparse
import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import train


def expand(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def run_one(args) -> Dict[str, Any]:
    config, overrides = args
    flat = [f"{k}={json.dumps(v)}" for k, v in overrides.items()]
    train.main(["--config", config, "--set", *flat])
    return overrides


def summarize(run_dir: Path, key: str = "eval/return") -> Dict[str, float]:
    finals = []
    for metrics in run_dir.glob("*/metrics.jsonl"):
        rows = [json.loads(line) for line in metrics.read_text(encoding="utf-8").splitlines()]
        values = [r[key] for r in rows if key in r]
        if values:
            finals.append(values[-1])
    if not finals:
        return {}
    return {
        "runs": len(finals),
        f"{key}/mean": float(np.mean(finals)),
        f"{key}/std": float(np.std(finals)),
        f"{key}/median": float(np.median(finals)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--grid", default='{"seed": [0, 1, 2]}')
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--run-dir", default="runs")
    args = parser.parse_args()

    jobs = [(args.config, o) for o in expand(json.loads(args.grid))]
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as pool:
            list(pool.map(run_one, jobs))
    else:
        for job in jobs:
            run_one(job)

    print(json.dumps(summarize(Path(args.run_dir)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
