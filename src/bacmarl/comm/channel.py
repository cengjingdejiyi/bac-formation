"""信道模型：整条丢弃 + 比特翻转（负责人 A，丢包部分与 TMC 对齐）。

边约定（与环境、注意力一致）：``mask[b, i, j]`` 中 i 是接收者、j 是发送者，
``True`` 表示接收者 i 能收到发送者 j 的消息；``edge[b, i, j] = msgs[b, j]``。

sample 与 apply 分离，是为了让 PPO 更新时能复用采样阶段的噪声样本。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class ChannelNoise:
    drop: torch.Tensor   # [B, N, N] bool，边 i←j 整条消息被丢弃
    flip: torch.Tensor   # [B, N, N, L] bool，边 i←j 上该比特翻转（仅二值消息）


def _as_batch_prob(prob, batch: int, ndim: int):
    """把标量或逐环境概率 ``[B]`` 转成可与 ``[B, ...]`` 广播的形状。

    标量原样返回 float；长度 B 的向量（list/tuple/ndarray/Tensor）返回
    ``[B, 1, ..., 1]``（共 ndim 维）的 float 张量，便于按 batch 轴广播。
    """
    if isinstance(prob, (bool, int, float)):
        return float(prob)
    t = torch.as_tensor(prob, dtype=torch.float32).reshape(-1)
    if t.numel() == 1:
        return float(t.item())
    if t.numel() != batch:
        raise ValueError(f"信道概率长度 {t.numel()} 与 batch={batch} 不匹配")
    return t.reshape(batch, *([1] * (ndim - 1)))


def sample_noise(comm_mask: torch.Tensor, msg_bits: int,
                 drop_p: float | "torch.Tensor" | np.ndarray,
                 flip_p: float | "torch.Tensor" | np.ndarray,
                 generator: torch.Generator | None = None) -> ChannelNoise:
    """按接收侧可连通关系采样噪声。comm_mask [B, N, N]。

    ``drop_p`` / ``flip_p`` 支持标量或逐环境概率 ``[B]``：并行环境每回合各自
    采样信道参数时用后者，噪声会落在各自的 batch 片上。
    """
    b, n, _ = comm_mask.shape
    drop = (torch.rand((b, n, n), generator=generator) < _as_batch_prob(drop_p, b, 3)) & comm_mask
    if msg_bits > 0:
        flip = ((torch.rand((b, n, n, msg_bits), generator=generator) < _as_batch_prob(flip_p, b, 4))
                & comm_mask.unsqueeze(-1))
    else:
        flip = torch.zeros((b, n, n, 0), dtype=torch.bool)
    return ChannelNoise(drop=drop, flip=flip)


def apply_noise(msgs: torch.Tensor, recv_mask: torch.Tensor, noise: ChannelNoise,
                binary: bool = True) -> torch.Tensor:
    """把 [B, N, L] 消息按边展开成 [B, N, N, L] 并施加噪声。

    key 约定：``msgs[b, j]`` 是发送者 j 的消息，落在第 3 维（发送者轴）的 j 位置；
    ``recv_mask[b, i, j]`` 表示接收者 i 能否收到发送者 j；因此
    ``edge[b, i, j] = msgs[b, j]``。
    """
    b, n, l = msgs.shape
    edge = msgs.unsqueeze(1).expand(b, n, n, l)                    # [B, N, N, L]：j 轴放发送者消息
    keep_edge = recv_mask & ~noise.drop                             # [B, N, N]
    edge = edge * keep_edge.unsqueeze(-1).to(edge.dtype)
    if binary and l > 0:
        edge = torch.where(noise.flip, -edge, edge)
    return edge

# 无 torch 场景请直接用 np_channel（同名函数在此再导出，便于调用方统一入口）
from .np_channel import apply_noise_np, sample_noise_np  # noqa: F401
