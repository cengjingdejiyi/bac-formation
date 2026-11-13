"""β 的拉格朗日对偶更新（负责人 A）。

对偶上升：β ← max(0, β + η( b̂ − b_max ))，b̂ 为本轮实测平均比特数。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BetaDual:
    beta: float = 0.1
    lr: float = 0.05
    b_max: float = 2.0

    def update(self, measured_bits: float) -> float:
        self.beta = max(0.0, self.beta + self.lr * (measured_bits - self.b_max))
        return self.beta

    def shaped_reward(self, reward_team, bits):
        """训练奖励合成：r_i = reward_team - β · bits_i。"""
        return reward_team - self.beta * bits
