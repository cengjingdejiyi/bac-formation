"""循环 Actor：编码 → 门控 → 通信 → 策略（负责人 C，通信模块 A/D 提供）。"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Categorical

from .encoder import Encoder


@dataclass
class ActorOut:
    action_logits: torch.Tensor        # [B, N, 5]
    gate_logits: torch.Tensor | None   # [B, N]
    gate: torch.Tensor                 # [B, N]
    h: torch.Tensor                    # [B, N, H]
    comm_c: torch.Tensor               # [B, N, H]
    bits: torch.Tensor                 # [B, N]
    noise: object = None               # ChannelNoise，供 PPO 更新复用


class Actor(nn.Module):
    """所有无人机共享参数；输入只用相对量与邻居序，不附加智能体编号。"""

    def __init__(self, obs_dim: int, n_obs: int, n_actions: int, hidden: int, comm_module):
        super().__init__()
        self.hidden = hidden
        self.encoder = Encoder(obs_dim, n_obs, hidden)
        self.gru = nn.GRUCell(hidden, hidden)
        self.comm = comm_module
        self.gate_head = nn.Linear(hidden, 1) if comm_module.has_gate else None
        self.policy_head = nn.Sequential(
            nn.Linear(hidden + hidden, hidden), nn.Tanh(), nn.Linear(hidden, n_actions)
        )

    def init_hidden(self, b: int, n: int, device=None) -> torch.Tensor:
        device = device or next(self.parameters()).device
        return torch.zeros(b, n, self.hidden, device=device)

    def forward(self, obs, obs_mask, comm_mask, h_prev, channel,
                gate=None, noise=None) -> ActorOut:
        b, n, _ = obs.shape
        x = self.encoder(obs, obs_mask).reshape(b * n, self.hidden)
        h = self.gru(x, h_prev.reshape(b * n, self.hidden)).reshape(b, n, self.hidden)

        gate_logits = None
        if self.gate_head is not None:
            gate_logits = self.gate_head(h).squeeze(-1)
            if gate is None:
                gate = Bernoulli(logits=gate_logits).sample()
        else:
            gate = torch.ones((b, n), dtype=torch.float32, device=h.device)

        comm = self.comm(h, comm_mask, gate, channel, noise=noise)
        logits = self.policy_head(torch.cat([h, comm.c], dim=-1))
        return ActorOut(action_logits=logits, gate_logits=gate_logits, gate=gate, h=h,
                        comm_c=comm.c, bits=comm.bits, noise=comm.noise)

    @staticmethod
    def log_prob(out: ActorOut, action: torch.Tensor, gate: torch.Tensor | None = None) -> torch.Tensor:
        lp = Categorical(logits=out.action_logits).log_prob(action)
        if out.gate_logits is not None:
            g = out.gate if gate is None else gate
            lp = lp + Bernoulli(logits=out.gate_logits).log_prob(g.to(out.gate_logits.dtype))
        return lp
