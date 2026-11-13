"""IC3Net 式：连续消息 + 均值聚合（对比基线，负责人 D）。"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import CommModule, CommOut, register
from .channel import apply_noise, sample_noise


@register("ic3net")
class CommIC3Net(CommModule):
    has_gate = True

    def __init__(
        self,
        hidden: int,
        msg_dim: int = 32,
        msg_bits: int = 4,
        recv_k: int = 3,
        hidden_dim: int | None = None,
        **kw,
    ):
        super().__init__(hidden, msg_dim, msg_bits, recv_k)
        self.bits_per_msg = 32 * msg_dim  # 连续消息按 32 bit/维计
        self.w_msg = nn.Linear(hidden, msg_dim)
        self.w_out = nn.Linear(msg_dim, hidden)

    def forward(self, h, comm_mask, gate, channel, noise=None) -> CommOut:
        b, n, _ = h.shape
        if noise is None:
            drop_p = channel.get("drop_p", 0.0) if isinstance(channel, dict) else 0.0
            noise = sample_noise(comm_mask, 0, drop_p, 0.0)
        msgs = self.w_msg(h) * gate.unsqueeze(-1).to(h.dtype)  # [B, N, M]
        edge = apply_noise(msgs, comm_mask, noise, binary=False)  # [B, N, N, M]
        vis = comm_mask & ~noise.drop & gate.to(torch.bool).unsqueeze(1)
        w = vis.to(edge.dtype)
        denom = w.sum(dim=-1, keepdim=True).clamp_min(1.0)  # [B, N, 1]
        pooled = torch.einsum("bnm,bnmh->bnh", w, edge) / denom
        c = self.w_out(pooled) * vis.any(dim=-1, keepdim=True).to(edge.dtype)
        bits = gate.float() * float(self.bits_per_msg)
        attn = w / denom
        return CommOut(c=c, bits=bits, attn=attn, noise=noise)
