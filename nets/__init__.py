from core.registry import NETS, autoload

from .mlp import NoisyLinear, orthogonal_init

autoload("nets")


def make_net(name: str, *args, **kwargs):
    return NETS.get(name)(*args, **kwargs)


__all__ = ["NETS", "NoisyLinear", "make_net", "orthogonal_init"]
