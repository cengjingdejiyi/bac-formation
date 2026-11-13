"""算法包。

延迟导入：``RolloutBuffer`` / ``BetaDual`` 是纯 numpy，可脱离 torch 运行；
torch 侧算法只在取用时加载。
"""
from __future__ import annotations

from importlib import import_module

_LAZY = {
    "RolloutBuffer": ".buffer",
    "BetaDual": ".dual",
    "ppo_update": ".mappo",
    "gae": ".mappo",
}
__all__ = sorted(_LAZY)


def __getattr__(name):
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(_LAZY[name], __name__), name)


def __dir__():
    return sorted(__all__)
