"""Critic：central 读全局状态，local 读局部观测（IPPO）（负责人 C）。"""
from __future__ import annotations

import torch
import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, mode: str, state_dim: int, obs_dim: int, n_agents: int, hidden: int):
        super().__init__()
        self.mode = mode
        in_dim = state_dim if mode == "central" else obs_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.n_agents = n_agents

    def forward(self, state=None, obs=None) -> torch.Tensor:  # [B, N]
        if self.mode == "central":
            v = self.net(state).squeeze(-1).unsqueeze(-1)      # [B, 1]
        else:
            v = self.net(obs).squeeze(-1)                       # [B, N]
        return v.expand(-1, self.n_agents) if v.shape[-1] == 1 else v
