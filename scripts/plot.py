from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(run: Path, key: str):
    xs, ys = [], []
    for line in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if key in row:
            xs.append(row["step"])
            ys.append(row[key])
    return xs, ys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--key", default="ep/return")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    for run in args.runs:
        path = Path(run)
        xs, ys = load(path, args.key)
        plt.plot(xs, ys, label=path.name)
    plt.xlabel("step")
    plt.ylabel(args.key)
    plt.legend()
    plt.grid(alpha=0.3)
    if args.out:
        plt.savefig(args.out, dpi=130, bbox_inches="tight")
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
