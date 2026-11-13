"""TarMAC 式：全精度连续广播 + 注意力（通信性能上界，负责人 D）。

边约定与 BAC 一致：``comm_mask[b, i, j]`` 中 i 是接收者、j 是发送者；
注意力只在物理可达且未被丢弃的 j→i 边上归一化，整行无有效边时权重与上下文为 0。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import CommModule, CommOut, register
from .channel import apply_noise, sample_noise


@register("tarmac")
class CommTarMAC(CommModule):
    has_gate = False

    def __init__(self, hidden: int, msg_dim: int = 32, msg_bits: int = 4, recv_k: int = 3,
                 hidden_dim: int | None = None, **kw):
        super().__init__(hidden, msg_dim, msg_bits, recv_k)
        self.bits_per_msg = 32 * msg_dim
        self.w_msg = nn.Linear(hidden, msg_dim)
        self.w_q = nn.Linear(hidden, hidden)
        self.w_k = nn.Linear(msg_dim, hidden)
        self.w_v = nn.Linear(msg_dim, hidden)

    def forward(self, h, comm_mask, gate, channel, noise=None) -> CommOut:
        b, n, _ = h.shape
        channel = channel if isinstance(channel, dict) else {}
        drop_p = channel.get("drop_p", 0.0)
        if noise is None:
            noise = sample_noise(comm_mask, 0, drop_p, 0.0)
        msgs = self.w_msg(h).contiguous()                 # [B, N, M]
        msgs = msgs * gate.unsqueeze(-1).to(msgs.dtype)   # 无门控时 gate 全 1
        edge = apply_noise(msgs, comm_mask, noise, binary=False)
        q = self.w_q(h)
        k = self.w_k(edge)
        score = torch.einsum("bnh,bnmh->bnm", q, k) / (self.hidden ** 0.5)
        vis = comm_mask & ~noise.drop                     # [B, N, N]：丢弃边不参与归一化
        attn = self._masked_softmax(score, vis)
        attn = attn * vis.any(dim=-1, keepdim=True).to(attn.dtype)
        v = self.w_v(edge)
        c = torch.einsum("bnm,bnmh->bnh", attn, v)
        bits = gate.float() * float(self.bits_per_msg)
        return CommOut(c=c, bits=bits, attn=attn, noise=noise)
