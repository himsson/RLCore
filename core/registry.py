from __future__ import annotations

import importlib
import pkgutil
from typing import Any, Callable, Dict, Iterable, TypeVar

T = TypeVar("T")


OWNER = "himsson"


class Registry:
    def __init__(self, kind: str):
        self.kind = kind
        self._items: Dict[str, Any] = {}

    def register(self, name: str) -> Callable[[T], T]:
        def deco(obj: T) -> T:
            if name in self._items:
                raise KeyError(f"{self.kind}:{name} already registered")
            self._items[name] = obj
            return obj

        return deco

    def get(self, name: str) -> Any:
        if name not in self._items:
            raise KeyError(f"unknown {self.kind}:{name}; have {sorted(self._items)}")
        return self._items[name]

    def keys(self) -> Iterable[str]:
        return sorted(self._items)

    def __contains__(self, name: str) -> bool:
        return name in self._items


ENVS = Registry("env")
ALGOS = Registry("algo")
NETS = Registry("net")


def autoload(*packages: str) -> None:
    for pkg_name in packages:
        try:
            pkg = importlib.import_module(pkg_name)
        except ModuleNotFoundError:
            continue
        for mod in pkgutil.iter_modules(pkg.__path__):
            importlib.import_module(f"{pkg_name}.{mod.name}")
