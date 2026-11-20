"""β 的拉格朗日对偶更新（负责人 A）。

对偶上升：β ← clamp(β + η(b̂ − b_max), 0, β_max)，b̂ 为本轮实测平均比特数。

口径（与 `docs/contracts.md` 第 3、4 节一致，改之前先改文档）：
- b̂ 是「每机每步平均比特数」，即所有 T、B、N 上的 `bits.mean()`；
- 一次广播只计一次，不因接收边数涨价，丢包也不退款（bits 由门控决定）；
- β 非负、有上界 β_max：预算极紧时防止 β 线性发散把奖励项吃穿；
- 本轮 b̂ 不是有限数（空 buffer / NaN）时跳过更新，不动 β；
- β 随 checkpoint 一起存取，断点续训不重置对偶变量。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def _mean_scalar(value) -> float:
    """把标量、numpy、torch 张量统一成 float 均值。"""
    if isinstance(value, (int, float, np.floating, np.integer)):
        return float(value)
    if hasattr(value, "detach"):  # torch 张量
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return float(np.asarray(value, dtype=np.float64).mean())


@dataclass
class BetaDual:
    beta: float = 0.1
    lr: float = 0.05
    b_max: float = 2.0
    beta_max: float = 5.0

    def update(self, measured_bits) -> float:
        """用本轮的 b̂ 更新一次 β，返回更新后的 β。"""
        b_hat = _mean_scalar(measured_bits)
        if not math.isfinite(b_hat):
            return self.beta
        beta = self.beta + self.lr * (b_hat - self.b_max)
        self.beta = float(min(max(beta, 0.0), self.beta_max))
        return self.beta

    def shaped_reward(self, reward_team, bits):
        """训练奖励合成：r_i = reward_team - β · bits_i。"""
        return reward_team - self.beta * bits

    def state_dict(self) -> dict:
        """只存对偶状态与超参，供 checkpoint 断点续训。"""
        return {
            "beta": float(self.beta),
            "lr": float(self.lr),
            "b_max": float(self.b_max),
            "beta_max": float(self.beta_max),
        }

    def load_state_dict(self, state: dict) -> None:
        for key in ("beta", "lr", "b_max", "beta_max"):
            if key in state:
                setattr(self, key, float(state[key]))
