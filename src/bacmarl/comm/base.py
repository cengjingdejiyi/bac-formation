"""通信模块基类与注册表（负责人 A）。

四种通信方式（none / ic3net / tarmac / bac）都实现这一接口，Actor 只认它。
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class CommOut:
    c: torch.Tensor          # [B, N, C] 聚合后的通信上下文
    bits: torch.Tensor       # [B, N]    本步实际发送比特数
    attn: torch.Tensor       # [B, N, N] 注意力/可见权重，attn[b,i,j] 为接收者 i 对发送者 j 的权重
    noise: object = None     # ChannelNoise，供 PPO 更新时复用


class CommModule(nn.Module):
    """所有通信方式的共同接口。"""

    has_gate: bool = False
    bits_per_msg: int = 0

    def __init__(self, hidden: int, msg_dim: int = 8, msg_bits: int = 4, recv_k: int = 3):
        super().__init__()
        self.hidden = hidden
        self.msg_dim = msg_dim
        self.msg_bits = msg_bits
        self.recv_k = recv_k

    def forward(self, h, comm_mask, gate, channel, noise=None) -> CommOut:
        """h [B, N, H]；comm_mask [B, N, N]；gate [B, N]；noise=None 表示新采样。"""
        raise NotImplementedError

    # 供子类复用的工具
    @staticmethod
    def _masked_softmax(score, mask, dim=-1):
        score = score.masked_fill(~mask, -1e9)
        return torch.softmax(score, dim=dim)


COMM_REGISTRY = {}


def register(name: str):
    def _wrap(cls):
        COMM_REGISTRY[name] = cls
        return cls

    return _wrap


def build_comm(name: str, hidden: int, cfg: dict | None = None) -> CommModule:
    """按配置构造通信模块；模块自注册，新增方式只需 register。"""
    cfg = cfg or {}
    if name not in COMM_REGISTRY:
        # 延迟导入，避免循环依赖
        from . import bac, ic3net, none, tarmac  # noqa: F401

    if name not in COMM_REGISTRY:
        raise KeyError(f"未知通信方式 {name}，可选：{sorted(COMM_REGISTRY)}")
    cls = COMM_REGISTRY[name]
    return cls(
        hidden=hidden,
        msg_dim=int(cfg.get("msg_dim", 8)),
        msg_bits=int(cfg.get("msg_bits", 4)),
        recv_k=int(cfg.get("recv_k", 3)),
        hidden_dim=hidden,
    )
