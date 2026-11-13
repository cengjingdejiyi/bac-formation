"""信道噪声的 numpy 实现：不依赖 torch，供冒烟、环境侧与测试复用。

与 ``channel.py`` 的 torch 版语义逐项对齐（同一 drop / flip 掩码定义）。
"""

from __future__ import annotations

import numpy as np


def sample_noise_np(rng, comm_mask, msg_bits: int, drop_p: float, flip_p: float):
    """rng 为 np.random.Generator，comm_mask [B, N, N]。返回 dict(drop, flip)。"""
    b = comm_mask.shape[0]

    def probability(value):
        p = np.asarray(value, dtype=np.float32)
        if p.ndim == 0:
            return p
        if p.shape == (b,):
            return p.reshape(b, 1, 1)
        raise ValueError("channel probability must be a scalar or [B]")

    drop_prob, flip_prob = probability(drop_p), probability(flip_p)
    drop = (rng.random(comm_mask.shape) < drop_prob) & comm_mask
    if msg_bits > 0:
        flip = (
            rng.random(comm_mask.shape + (msg_bits,)) < flip_prob[..., None]
        ) & comm_mask[..., None]
    else:
        flip = np.zeros(comm_mask.shape + (0,), dtype=bool)
    return {"drop": drop, "flip": flip}


def apply_noise_np(msgs, recv_mask, noise, binary: bool = True):
    """msgs [B,sender,L]；返回 edge [B,receiver,sender,L]，掩码/噪声均按 j→i。"""
    edge = np.broadcast_to(
        msgs[:, None, :, :], msgs.shape[:2] + (msgs.shape[1], msgs.shape[2])
    )
    keep = recv_mask & ~noise["drop"]
    edge = edge * keep[..., None]
    if binary and msgs.shape[-1] > 0:
        edge = np.where(noise["flip"], -edge, edge)
    return edge
