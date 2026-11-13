"""信道模型：整条丢弃 + 比特翻转（负责人 A，丢包部分与 TMC 对齐）。

sample 与 apply 分离，是为了让 PPO 更新时能复用采样阶段的噪声样本。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class ChannelNoise:
    drop: torch.Tensor   # [B, N, N] bool，True 表示 i→j 整条消息被丢弃
    flip: torch.Tensor   # [B, N, N, L] bool，True 表示该比特翻转（仅二值消息）


def sample_noise(comm_mask: torch.Tensor, msg_bits: int, drop_p: float, flip_p: float,
                 generator: torch.Generator | None = None) -> ChannelNoise:
    """按接收侧可连通关系采样噪声。comm_mask [B, N, N]。"""
    b, n, _ = comm_mask.shape
    drop = (torch.rand((b, n, n), generator=generator) < drop_p) & comm_mask
    if msg_bits > 0:
        flip = (torch.rand((b, n, n, msg_bits), generator=generator) < flip_p) & comm_mask.unsqueeze(-1)
    else:
        flip = torch.zeros((b, n, n, 0), dtype=torch.bool)
    return ChannelNoise(drop=drop, flip=flip)


def apply_noise(msgs: torch.Tensor, recv_mask: torch.Tensor, noise: ChannelNoise,
                binary: bool = True) -> torch.Tensor:
    """把 [B, N, L] 消息按边展开成 [B, N, N, L] 并施加噪声。

    key 约定：msgs[:, i] 是 i 发出的消息，落在第 2 维的 i 位置；recv_mask[i, j] 表示 j 能否收到 i。
    """
    b, n, l = msgs.shape
    edge = msgs.unsqueeze(2).expand(b, n, n, l)                    # [B, N, N, L]
    keep_edge = recv_mask & ~noise.drop                             # [B, N, N]
    edge = edge * keep_edge.unsqueeze(-1).to(edge.dtype)
    if binary and l > 0:
        edge = torch.where(noise.flip, -edge, edge)
    return edge

# 无 torch 场景请直接用 np_channel（同名函数在此再导出，便于调用方统一入口）
from .np_channel import apply_noise_np, sample_noise_np  # noqa: F401
