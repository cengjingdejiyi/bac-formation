"""观测编码器（负责人 C）。"""
from __future__ import annotations

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """obs + obs_mask → 隐状态输入。输入 [B, N, D_obs]、[B, N, n_obs]，输出 [B, N, H]。"""

    def __init__(self, obs_dim: int, n_obs: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + n_obs, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor, obs_mask: torch.Tensor) -> torch.Tensor:  # [B, N, H]
        x = torch.cat([obs, obs_mask.to(obs.dtype)], dim=-1)
        return self.net(x)
