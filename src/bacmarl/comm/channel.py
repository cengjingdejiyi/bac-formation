"""信道模型：整条丢弃 + 比特翻转（负责人 A，丢包部分与 TMC 对齐）。

sample 与 apply 分离，是为了让 PPO 更新时能复用采样阶段的噪声样本。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ChannelNoise:
    drop: torch.Tensor  # [B, N, N] bool，[b,i,j] 表示 j→i 整条消息被丢弃
    flip: torch.Tensor  # [B, N, N, L] bool，True 表示该比特翻转（仅二值消息）


def sample_noise(
    comm_mask: torch.Tensor,
    msg_bits: int,
    drop_p: float | torch.Tensor,
    flip_p: float | torch.Tensor,
    generator: torch.Generator | None = None,
) -> ChannelNoise:
    """comm_mask [B, receiver, sender]；概率为标量或逐环境 [B]。"""
    b, n, _ = comm_mask.shape

    def probability(value: float | torch.Tensor) -> torch.Tensor:
        p = torch.as_tensor(value, dtype=torch.float32, device=comm_mask.device)
        if p.ndim == 0:
            return p
        if p.shape == (b,):
            return p.reshape(b, 1, 1)
        raise ValueError("channel probability must be a scalar or [B]")

    drop_prob, flip_prob = probability(drop_p), probability(flip_p)
    drop = (
        torch.rand((b, n, n), device=comm_mask.device, generator=generator) < drop_prob
    ) & comm_mask
    if msg_bits > 0:
        flip = (
            torch.rand(
                (b, n, n, msg_bits), device=comm_mask.device, generator=generator
            )
            < flip_prob.unsqueeze(-1)
        ) & comm_mask.unsqueeze(-1)
    else:
        flip = torch.zeros((b, n, n, 0), dtype=torch.bool, device=comm_mask.device)
    return ChannelNoise(drop=drop, flip=flip)


def apply_noise(
    msgs: torch.Tensor,
    recv_mask: torch.Tensor,
    noise: ChannelNoise,
    binary: bool = True,
) -> torch.Tensor:
    """把 [B, N, L] 消息按边展开成 [B, N, N, L] 并施加噪声。

    msgs[b,j] 是 j 发出的消息；recv_mask[b,i,j] 表示 i 能否收到 j。
    返回 edge[b,i,j,:] = msgs[b,j,:]（再施加该 j→i 边上的丢包和翻转）。
    """
    b, n, l = msgs.shape
    edge = msgs.unsqueeze(1).expand(b, n, n, l)  # [B, receiver, sender, L]
    keep_edge = recv_mask & ~noise.drop  # [B, N, N]
    edge = edge * keep_edge.unsqueeze(-1).to(edge.dtype)
    if binary and l > 0:
        edge = torch.where(noise.flip, -edge, edge)
    return edge


# 无 torch 场景请直接用 np_channel（同名函数在此再导出，便于调用方统一入口）
from .np_channel import apply_noise_np, sample_noise_np  # noqa: F401
