"""BAC：带宽感知注意力通信（主方法，负责人 A）。

三阶段：门控（外部采样）→ 二值消息 sign + STE → Top-k 注意力聚合。

边约定：``comm_mask[b, i, j]`` 中 i 是接收者、j 是发送者；消息沿发送者轴 j 展开
（``edge[b, i, j] = msgs[b, j]``），门控、丢弃、翻转与注意力共用同一条 j→i 边。
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
        channel = channel if isinstance(channel, dict) else {}
        # 信道概率支持标量或逐环境向量 [B]，直接交给 sample_noise 广播
        drop_p = channel.get("drop_p", 0.0)
        flip_p = channel.get("flip_p", 0.0)

        if noise is None:
            noise = sample_noise(comm_mask, L, drop_p, flip_p)

        msgs = self.encode_msg(h)                                # [B, N, L]
        msgs = msgs * gate.unsqueeze(-1).to(msgs.dtype)          # 未发言视为不发
        edge = apply_noise(msgs, comm_mask, noise, binary=True)  # [B, N, N, L]

        q = self.w_q(h)                                          # [B, N, H]
        k = self.w_k(edge)                                       # [B, N, N, H]
        score = torch.einsum("bnh,bnmh->bnm", q, k) / (self.hidden ** 0.5)
        # 有效边 = 物理可达 ∧ 发送者发言 ∧ 消息未被丢弃；softmax 只在有效边上归一化
        vis = (comm_mask
               & gate.to(torch.bool).unsqueeze(1).expand(b, n, n)
               & ~noise.drop)                                    # [B, N, N]
        attn = self._masked_softmax(score, vis)
        # 整行没有有效边时显式置零：否则 softmax 会给均匀权重，w_v 的偏置会漏成非零上下文。
        attn = attn * vis.any(dim=-1, keepdim=True).to(attn.dtype)
        v = self.w_v(edge)                                       # [B, N, N, H]
        c = torch.einsum("bnm,bnmh->bnh", attn, v)               # 无有效边时 c 自然为 0

        bits = gate.float() * float(L)                            # [B, N]：发送侧计费，丢包不退款
        return CommOut(c=c, bits=bits, attn=attn, noise=noise)
