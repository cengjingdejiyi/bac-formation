"""信道噪声的 numpy 实现：不依赖 torch，供冒烟、环境侧与测试复用。

与 ``channel.py`` 的 torch 版语义逐项对齐（同一条 j→i 边、同一 drop / flip 掩码定义）：
``recv_mask[b, i, j]`` 中 i 是接收者、j 是发送者，``edge[b, i, j] = msgs[b, j]``。
"""
from __future__ import annotations

import numpy as np


def _np_prob(prob, batch: int, ndim: int):
    """标量原样返回 float；逐环境概率 [B] 返回 [B, 1, ..., 1]，便于按 batch 广播。"""
    if isinstance(prob, (bool, int, float)):
        return float(prob)
    a = np.asarray(prob, dtype=np.float64).reshape(-1)
    if a.size == 1:
        return float(a[0])
    if a.size != batch:
        raise ValueError(f"信道概率长度 {a.size} 与 batch={batch} 不匹配")
    return a.reshape((batch,) + (1,) * (ndim - 1))


def sample_noise_np(rng, comm_mask, msg_bits: int, drop_p, flip_p):
    """rng 为 np.random.Generator，comm_mask [B, N, N]。返回 dict(drop, flip)。

    ``drop_p`` / ``flip_p`` 支持标量或逐环境概率 ``[B]``。
    """
    b = comm_mask.shape[0]
    drop = (rng.random(comm_mask.shape) < _np_prob(drop_p, b, comm_mask.ndim)) & comm_mask
    if msg_bits > 0:
        flip = ((rng.random(comm_mask.shape + (msg_bits,)) < _np_prob(flip_p, b, comm_mask.ndim + 1))
                & comm_mask[..., None])
    else:
        flip = np.zeros(comm_mask.shape + (0,), dtype=bool)
    return {"drop": drop, "flip": flip}


def apply_noise_np(msgs, recv_mask, noise, binary: bool = True):
    """msgs [B, N, L]、recv_mask [B, N, N]；返回按边展开的 [B, N, N, L]。

    ``edge[b, i, j] = msgs[b, j]``：第 3 维（发送者轴）放的是发送者 j 的消息。
    """
    b, n, l = msgs.shape
    edge = np.broadcast_to(msgs[:, None, :, :], (b, n, n, l))
    keep = recv_mask & ~noise["drop"]
    edge = edge * keep[..., None]
    if binary and l > 0:
        edge = np.where(noise["flip"], -edge, edge)
    return edge
