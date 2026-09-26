from __future__ import annotations

import csv
import json
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import numpy as np


class Meter:
    def __init__(self, window: int = 100):
        self.values: Deque[float] = deque(maxlen=window)

    def add(self, value: float) -> None:
        self.values.append(float(value))

    @property
    def mean(self) -> float:
        return float(np.mean(self.values)) if self.values else 0.0

    @property
    def std(self) -> float:
        return float(np.std(self.values)) if self.values else 0.0


class Backend:
    def log(self, row: Dict[str, Any], step: int) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class JsonlBackend(Backend):
    def __init__(self, path: Path):
        self.file = path.open("a", encoding="utf-8")

    def log(self, row: Dict[str, Any], step: int) -> None:
        self.file.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.file.flush()

    def close(self) -> None:
        self.file.close()


class CsvBackend(Backend):
    def __init__(self, path: Path):
        self.path = path
        self.fields: List[str] = []
        self.rows: List[Dict[str, Any]] = []

    def log(self, row: Dict[str, Any], step: int) -> None:
        new = [k for k in row if k not in self.fields]
        self.rows.append(row)
        if new:
            self.fields += new
            self._rewrite()
        else:
            with self.path.open("a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, self.fields).writerow(row)

    def _rewrite(self) -> None:
        with self.path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, self.fields)
            writer.writeheader()
            writer.writerows(self.rows)


class TensorBoardBackend(Backend):
    def __init__(self, path: Path):
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(str(path))

    def log(self, row: Dict[str, Any], step: int) -> None:
        for key, value in row.items():
            if key != "step" and isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, step)

    def close(self) -> None:
        self.writer.close()


class StdoutBackend(Backend):
    def __init__(self, keys: Optional[List[str]] = None):
        self.keys = keys

    def log(self, row: Dict[str, Any], step: int) -> None:
        items = row.items() if self.keys is None else [(k, row[k]) for k in self.keys if k in row]
        print(" | ".join(f"{k}={v}" for k, v in items), flush=True)


class Logger:
    def __init__(
        self,
        run_dir: str | Path,
        window: int = 100,
        stdout: bool = True,
        csv_out: bool = False,
        tensorboard: bool = False,
    ):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.meters: Dict[str, Meter] = defaultdict(lambda: Meter(window))
        self.start = time.time()
        self.backends: List[Backend] = [JsonlBackend(self.run_dir / "metrics.jsonl")]
        if stdout:
            self.backends.append(StdoutBackend())
        if csv_out:
            self.backends.append(CsvBackend(self.run_dir / "metrics.csv"))
        if tensorboard:
            self.backends.append(TensorBoardBackend(self.run_dir / "tb"))

    def attach(self, backend: Backend) -> None:
        self.backends.append(backend)

    def add(self, key: str, value: float) -> None:
        self.meters[key].add(value)

    def add_many(self, values: Dict[str, float]) -> None:
        for key, value in values.items():
            self.add(key, value)

    def snapshot(self) -> Dict[str, float]:
        return {k: round(m.mean, 6) for k, m in self.meters.items() if m.values}

    def dump(self, step: int, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        row: Dict[str, Any] = {"step": step, "time": round(time.time() - self.start, 2)}
        row.update(self.snapshot())
        if extra:
            row.update(extra)
        for backend in self.backends:
            backend.log(row, step)
        return row

    def close(self) -> None:
        for backend in self.backends:
            backend.close()
