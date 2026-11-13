"""通信模块包。

延迟导入：``import bacmarl.comm.np_channel`` 这类纯 numpy 用法不会被 torch 拖累；
只有在真正取用 torch 侧符号时才加载 ``base`` / ``bac`` 等子模块。
"""
from __future__ import annotations

from importlib import import_module

_LAZY = {
    "CommModule": ".base", "CommOut": ".base", "build_comm": ".base",
    "COMM_REGISTRY": ".base", "register": ".base",
    "ChannelNoise": ".channel", "sample_noise": ".channel", "apply_noise": ".channel",
    "sample_noise_np": ".np_channel", "apply_noise_np": ".np_channel",
    "BACComm": ".bac", "IC3NetComm": ".ic3net", "TarMACComm": ".tarmac", "NoneComm": ".none",
}
__all__ = sorted(_LAZY)


def __getattr__(name):
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod = import_module(_LAZY[name], __name__)
    return getattr(mod, name)


def __dir__():
    return sorted(__all__)
