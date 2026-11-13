"""无通信（MAPPO/IPPO 对照）。"""
from __future__ import annotations

import torch

from .base import CommModule, CommOut, register


@register("none")
class CommNone(CommModule):
    has_gate = False
    bits_per_msg = 0

    def __init__(self, hidden: int, msg_dim: int = 8, msg_bits: int = 4, recv_k: int = 3,
                 hidden_dim: int | None = None, **kw):
        super().__init__(hidden, msg_dim, msg_bits, recv_k)

    def forward(self, h, comm_mask, gate, channel, noise=None) -> CommOut:
        b, n, _ = h.shape
        zeros = torch.zeros_like(h)
        bits = torch.zeros((b, n), dtype=torch.float32, device=h.device)
        eye = torch.eye(n, dtype=torch.bool, device=h.device).unsqueeze(0).expand(b, n, n)
        return CommOut(c=zeros, bits=bits, attn=eye.clone(), noise=noise)
