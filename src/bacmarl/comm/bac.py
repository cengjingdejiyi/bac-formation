"""BAC：带宽感知注意力通信（主方法，负责人 A）。

三阶段：门控（外部采样）→ 二值消息 sign + STE → 可见邻居加权聚合。
Top-K 不含在模块内，由环境给的 `comm_mask` 施加（半径 + 接收容量 K）。

聚合方式可由 `aggregate` 切换：
- `softmax`（默认）：q/k 注意力，`attn[b,i,j]` 是 i 对邻居 j 的权重；
- `mean`：均匀平均（消融 mean_agg），`attn[b,i,j]` 退化为"可见权重"，不是注意力。

`use_gate=False` 时模块自称无门控（消融 no_gate）：Actor 不再建门控头，gate 恒为 1，
`bits` 恒等于 `msg_bits`，此时 β 对偶对发言没有调节余地，只作对照。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import CommModule, CommOut, register
from .channel import apply_noise, sample_noise

AGGREGATES = ("softmax", "mean")


@register("bac")
class CommBAC(CommModule):
    has_gate = True

    def __init__(
        self,
        hidden: int,
        msg_dim: int = 8,
        msg_bits: int = 4,
        recv_k: int = 3,
        hidden_dim: int | None = None,
        aggregate: str = "softmax",
        use_gate: bool = True,
        **kw,
    ):
        super().__init__(hidden, msg_dim, msg_bits, recv_k)
        aggregate = str(aggregate).lower()
        if aggregate not in AGGREGATES:
            raise ValueError(f"aggregate 必须是 {AGGREGATES} 之一，收到 {aggregate!r}")
        self.aggregate = aggregate
        # 消融 no_gate：Actor 依据 has_gate 决定是否建门控头，这里必须落在实例上。
        self.has_gate = bool(use_gate)
        self.msg_bits = int(msg_bits)
        self.bits_per_msg = self.msg_bits
        self.w_msg = nn.Linear(hidden, self.msg_bits)
        self.w_q = nn.Linear(hidden, hidden)
        self.w_k = nn.Linear(self.msg_bits, hidden)
        self.w_v = nn.Linear(self.msg_bits, hidden)

    def encode_msg(self, h: torch.Tensor) -> torch.Tensor:
        """二值化 + STE：前向为 ±1，反向梯度直通。"""
        logits = self.w_msg(h)  # [B, N, L]
        hard = torch.sign(logits)
        hard = torch.where(hard == 0, torch.ones_like(hard), hard)
        return logits + (hard - logits).detach()

    def forward(self, h, comm_mask, gate, channel, noise=None) -> CommOut:
        """h [B,N,H]；comm_mask [B,N,N]（i 收 j）；gate [B,N] 或 None（视为全 1）。"""
        b, n, _ = h.shape
        L = self.msg_bits
        drop_p = channel.get("drop_p", 0.0) if isinstance(channel, dict) else 0.0
        flip_p = channel.get("flip_p", 0.0) if isinstance(channel, dict) else 0.0

        if noise is None:
            noise = sample_noise(comm_mask, L, drop_p, flip_p)
        if gate is None or not self.has_gate:
            # 无门控（消融 no_gate）或调用方未给门控时，全体按发言处理；
            # bits 也就恒为 L，不受调用方误传的零门控影响。
            gate = torch.ones((b, n), dtype=h.dtype, device=h.device)

        msgs = self.encode_msg(h)  # [B, N, L]
        msgs = msgs * gate.unsqueeze(-1).to(msgs.dtype)  # 未发言视为不发
        edge = apply_noise(
            msgs, comm_mask, noise, binary=True
        )  # [B,N,N,L]，edge[b,i,j]=j→i

        # 可见边：掩码允许、发送者已发言、未被整条丢弃；gate 落在发送者维 j 上。
        vis = comm_mask & ~noise.drop & gate.to(torch.bool).unsqueeze(1)

        v = self.w_v(edge)  # [B, N, N, H]
        if self.aggregate == "mean":
            # 消融 mean_agg：只在可见边上取平均，参数与 softmax 版保持同构。
            w = vis.to(edge.dtype)
            denom = w.sum(dim=-1, keepdim=True).clamp_min(1.0)  # 无邻居时权重为 0
            attn = w / denom
        else:
            q = self.w_q(h)  # [B, N, H]
            k = self.w_k(edge)  # [B, N, N, H]
            score = torch.einsum("bnh,bnmh->bnm", q, k) / (self.hidden**0.5)
            attn = self._masked_softmax(score, vis)

        c = torch.einsum("bnm,bnmh->bnh", attn, v)

        # 一次广播只计一次，丢包不退款；no_gate 时 gate 恒 1，bits 恒为 L。
        bits = gate.float() * float(L)  # [B, N]
        return CommOut(c=c, bits=bits, attn=attn, noise=noise)
