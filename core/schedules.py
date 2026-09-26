from __future__ import annotations

import math
from typing import Any, Callable, Dict, Sequence, Union

Schedule = Callable[[float], float]


def constant(value: float) -> Schedule:
    return lambda progress: value


def linear(start: float, end: float, end_fraction: float = 1.0) -> Schedule:
    def fn(progress: float) -> float:
        p = min(max(progress, 0.0) / max(end_fraction, 1e-8), 1.0)
        return start + (end - start) * p

    return fn


def exponential(start: float, decay: float, end: float = 0.0) -> Schedule:
    return lambda progress: max(end, start * decay ** max(progress, 0.0))


def cosine(start: float, end: float = 0.0, warmup: float = 0.0) -> Schedule:
    def fn(progress: float) -> float:
        if warmup and progress < warmup:
            return start * progress / warmup
        p = (min(max(progress, 0.0), 1.0) - warmup) / max(1.0 - warmup, 1e-8)
        return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * p))

    return fn


def piecewise(points: Sequence[Sequence[float]]) -> Schedule:
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]

    def fn(progress: float) -> float:
        if progress <= xs[0]:
            return ys[0]
        for i in range(1, len(xs)):
            if progress <= xs[i]:
                span = max(xs[i] - xs[i - 1], 1e-8)
                w = (progress - xs[i - 1]) / span
                return ys[i - 1] + w * (ys[i] - ys[i - 1])
        return ys[-1]

    return fn


BUILDERS: Dict[str, Callable[..., Schedule]] = {
    "constant": constant,
    "linear": linear,
    "exponential": exponential,
    "cosine": cosine,
    "piecewise": piecewise,
}


def make_schedule(spec: Union[float, int, str, Dict[str, Any], Schedule]) -> Schedule:
    if callable(spec):
        return spec
    if isinstance(spec, (int, float)):
        return constant(float(spec))
    if isinstance(spec, str):
        kind, _, rest = spec.partition(":")
        args = [float(x) for x in rest.split(",")] if rest else []
        return BUILDERS[kind](*args)
    spec = dict(spec)
    kind = spec.pop("type")
    return BUILDERS[kind](**spec)
