"""Rollout 缓存（负责人 C）。字段见 docs/contracts.md 第 4 节。

骨架额外存 h_prev，使更新时无需 BPTT 也能复用采样期隐状态；全轨迹反传留给 C 扩展。
"""
from __future__ import annotations

import numpy as np


class RolloutBuffer:
    def __init__(self, t_max: int, b: int, n: int, obs_dim: int, n_obs: int,
                 state_dim: int, hidden: int, msg_bits: int):
        self.T, self.B, self.N = t_max, b, n
        self.hidden = hidden
        self.obs = np.zeros((t_max, b, n, obs_dim), dtype=np.float32)
        self.obs_mask = np.zeros((t_max, b, n, n_obs), dtype=bool)
        self.state = np.zeros((t_max, b, state_dim), dtype=np.float32)
        self.comm_mask = np.zeros((t_max, b, n, n), dtype=bool)
        self.h_prev = np.zeros((t_max, b, n, hidden), dtype=np.float32)
        self.action = np.zeros((t_max, b, n), dtype=np.int64)
        self.gate = np.zeros((t_max, b, n), dtype=np.float32)
        self.logp_old = np.zeros((t_max, b, n), dtype=np.float32)
        self.value = np.zeros((t_max, b, n), dtype=np.float32)
        self.reward_team = np.zeros((t_max, b, n), dtype=np.float32)
        self.bits = np.zeros((t_max, b, n), dtype=np.float32)
        self.done = np.zeros((t_max, b), dtype=bool)
        self.noise_drop = np.zeros((t_max, b, n, n), dtype=bool)
        self.noise_flip = np.zeros((t_max, b, n, n, msg_bits), dtype=bool)
        self.ptr = 0
        self.adv = np.zeros((t_max, b, n), dtype=np.float32)
        self.ret = np.zeros((t_max, b, n), dtype=np.float32)

    def add(self, step: dict) -> None:
        i = self.ptr
        self.obs[i] = step['obs']
        self.obs_mask[i] = step['obs_mask']
        self.state[i] = step['state']
        self.comm_mask[i] = step['comm_mask']
        self.h_prev[i] = step['h_prev']
        self.action[i] = step['action']
        self.gate[i] = step['gate']
        self.logp_old[i] = step['logp_old']
        self.value[i] = step['value']
        self.reward_team[i] = step['reward_team']
        self.bits[i] = step['bits']
        self.done[i] = step['done']
        if step.get('noise_drop') is not None:
            self.noise_drop[i] = step['noise_drop']
            self.noise_flip[i] = step['noise_flip']
        self.ptr += 1

    def compute_returns(self, shaped_reward: np.ndarray, last_value: np.ndarray,
                        gamma: float, lam: float) -> None:
        """GAE：shaped_reward [T,B,N]，last_value [B,N]。"""
        gae_ = np.zeros((self.B, self.N), dtype=np.float32)
        for t in reversed(range(self.ptr)):
            next_v = last_value if t == self.ptr - 1 else self.value[t + 1]
            nonterm = 1.0 - self.done[t].astype(np.float32)[:, None]
            delta = shaped_reward[t] + gamma * next_v * nonterm - self.value[t]
            gae_ = delta + gamma * lam * nonterm * gae_
            self.adv[t] = gae_
            self.ret[t] = gae_ + self.value[t]

    def batched(self, idx: np.ndarray) -> dict:
        """按扁平索引 (t, b) 取小批量。"""
        t, b = idx // self.B, idx % self.B
        return {
            'obs': self.obs[t, b], 'obs_mask': self.obs_mask[t, b], 'state': self.state[t, b],
            'comm_mask': self.comm_mask[t, b], 'h_prev': self.h_prev[t, b],
            'action': self.action[t, b], 'gate': self.gate[t, b], 'logp_old': self.logp_old[t, b],
            'adv': self.adv[t, b], 'ret': self.ret[t, b],
            'noise_drop': self.noise_drop[t, b], 'noise_flip': self.noise_flip[t, b],
        }

    def flat_indices(self) -> np.ndarray:
        return np.arange(self.ptr * self.B)
