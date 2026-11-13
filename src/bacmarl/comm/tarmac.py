"""TarMAC 式：全精度连续广播 + 注意力（通信性能上界，负责人 D）。"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import CommModule, CommOut, register
from .channel import apply_noise, sample_noise


@register("tarmac")
class CommTarMAC(CommModule):
    has_gate = False

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
        self.bits_per_msg = 32 * msg_dim
        self.w_msg = nn.Linear(hidden, msg_dim)
        self.w_q = nn.Linear(hidden, hidden)
        self.w_k = nn.Linear(msg_dim, hidden)
        self.w_v = nn.Linear(msg_dim, hidden)

    def forward(self, h, comm_mask, gate, channel, noise=None) -> CommOut:
        b, n, _ = h.shape
        if noise is None:
            drop_p = channel.get("drop_p", 0.0) if isinstance(channel, dict) else 0.0
            noise = sample_noise(comm_mask, 0, drop_p, 0.0)
        msgs = self.w_msg(h).contiguous()  # [B, N, M]
        msgs = msgs * gate.unsqueeze(-1).to(msgs.dtype)  # 无门控时 gate 全 1
        edge = apply_noise(msgs, comm_mask, noise, binary=False)
        q = self.w_q(h)
        k = self.w_k(edge)
        score = torch.einsum("bnh,bnmh->bnm", q, k) / (self.hidden**0.5)
        vis = comm_mask & ~noise.drop & gate.to(torch.bool).unsqueeze(1)
        attn = self._masked_softmax(score, vis)
        v = self.w_v(edge)
        c = torch.einsum("bnm,bnmh->bnh", attn, v)
        bits = gate.float() * float(self.bits_per_msg)
        return CommOut(c=c, bits=bits, attn=attn, noise=noise)
