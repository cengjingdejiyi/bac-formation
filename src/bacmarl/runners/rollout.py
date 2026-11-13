"""按真实转移收集 rollout；跨段延续状态，done 后清零对应 RNN（负责人 C）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.distributions import Categorical

from bacmarl.algos.buffer import RolloutBuffer


@dataclass
class RolloutResult:
    batch: dict
    hidden: torch.Tensor  # [B,N,H]，已清零真正终止的环境
    last_value: np.ndarray  # [B,N]，最新 next state 的 value
    episodes: list[dict]


@torch.no_grad()
def collect_rollout(
    env: Any,
    actor: Any,
    critic: Any,
    buffer: RolloutBuffer,
    batch: dict,
    hidden: torch.Tensor,
) -> RolloutResult:
    """写入 (obs_t, action_t, reward_{t+1}, done_{t+1})；不调用 env.reset()。

    obs [B,N,D]、hidden [B,N,H]；自动 reset 的新观测只作下一动作起点。
    """
    if buffer.ptr or buffer.T < 1:
        raise ValueError("collect_rollout requires a nonempty, unused buffer")
    device = hidden.device
    episodes = []
    for _ in range(buffer.T):
        obs = torch.as_tensor(batch["obs"], device=device)
        state = torch.as_tensor(batch["state"], device=device)
        channel = {
            k: torch.as_tensor(v, device=device) for k, v in batch["channel"].items()
        }
        out = actor(
            obs,
            torch.as_tensor(batch["obs_mask"], device=device),
            torch.as_tensor(batch["comm_mask"], device=device),
            hidden,
            channel,
        )
        action = Categorical(logits=out.action_logits).sample()
        logp = actor.log_prob(out, action)
        value = critic(state=state, obs=obs)
        nxt = env.step(action.cpu().numpy())
        noise_drop = np.zeros((buffer.B, buffer.N, buffer.N), dtype=bool)
        noise_flip = np.zeros(
            (buffer.B, buffer.N, buffer.N, buffer.noise_flip.shape[-1]), dtype=bool
        )
        if out.noise is not None:
            noise_drop = out.noise.drop.cpu().numpy()
            if noise_flip.shape[-1]:
                noise_flip = out.noise.flip.cpu().numpy()
        buffer.add(
            {
                "obs": batch["obs"],
                "obs_mask": batch["obs_mask"],
                "state": batch["state"],
                "comm_mask": batch["comm_mask"],
                "h_prev": hidden.cpu().numpy(),
                "action": action.cpu().numpy(),
                "gate": out.gate.cpu().numpy(),
                "logp_old": logp.cpu().numpy(),
                "value": value.cpu().numpy(),
                "reward_team": nxt["reward"],
                "bits": out.bits.cpu().numpy(),
                "done": nxt["done"],
                "noise_drop": noise_drop,
                "noise_flip": noise_flip,
            }
        )
        hidden = out.h.detach().masked_fill(
            torch.as_tensor(nxt["done"], device=device)[:, None, None], 0.0
        )
        episodes.extend(ep for ep in nxt.get("ep", []) if ep is not None)
        batch = nxt
    last_value = (
        critic(
            state=torch.as_tensor(batch["state"], device=device),
            obs=torch.as_tensor(batch["obs"], device=device),
        )
        .cpu()
        .numpy()
    )
    return RolloutResult(batch, hidden, last_value, episodes)
