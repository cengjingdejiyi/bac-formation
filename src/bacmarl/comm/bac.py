"""BAC：带宽感知注意力通信（主方法，负责人 A）。

三阶段：门控（外部采样）→ 二值消息 sign + STE → Top-k 注意力聚合。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import CommModule, CommOut, register
from .channel import apply_noise, sample_noise


@register("bac")
class CommBAC(CommModule):
    has_gate = True

    def __init__(self, hidden: int, msg_dim: int = 8, msg_bits: int = 4, recv_k: int = 3,
                 hidden_dim: int | None = None, **kw):
        super().__init__(hidden, msg_dim, msg_bits, recv_k)
        self.msg_bits = int(msg_bits)
        self.bits_per_msg = self.msg_bits
        self.w_msg = nn.Linear(hidden, self.msg_bits)
        self.w_q = nn.Linear(hidden, hidden)
        self.w_k = nn.Linear(self.msg_bits, hidden)
        self.w_v = nn.Linear(self.msg_bits, hidden)

    def encode_msg(self, h: torch.Tensor) -> torch.Tensor:
        """二值化 + STE：前向为 ±1，反向梯度直通。"""
        logits = self.w_msg(h)                                  # [B, N, L]
        hard = torch.sign(logits)
        hard = torch.where(hard == 0, torch.ones_like(hard), hard)
        return logits + (hard - logits).detach()

    def forward(self, h, comm_mask, gate, channel, noise=None) -> CommOut:
        b, n, _ = h.shape
        L = self.msg_bits
        drop_p = float(channel.get("drop_p", 0.0)) if isinstance(channel, dict) else 0.0
        flip_p = float(channel.get("flip_p", 0.0)) if isinstance(channel, dict) else 0.0

        if noise is None:
            noise = sample_noise(comm_mask, L, drop_p, flip_p)

        msgs = self.encode_msg(h)                                # [B, N, L]
        msgs = msgs * gate.unsqueeze(-1).to(msgs.dtype)          # 未发言视为不发
        edge = apply_noise(msgs, comm_mask, noise, binary=True)  # [B, N, N, L]

        q = self.w_q(h)                                          # [B, N, H]
        k = self.w_k(edge)                                       # [B, N, N, H]
        score = torch.einsum("bnh,bnmh->bnm", q, k) / (self.hidden ** 0.5)
        vis = comm_mask & gate.to(torch.bool).unsqueeze(1).expand(b, n, n)  # 只有已发言邻居可见
        attn = self._masked_softmax(score, vis)
        v = self.w_v(edge)                                       # [B, N, N, H]
        c = torch.einsum("bnm,bnmh->bnh", attn, v)

        bits = gate.float() * float(L)                            # [B, N]
        return CommOut(c=c, bits=bits, attn=attn, noise=noise)
