"""奖励尺度归一化：Welford 在线均值/方差（numpy，无 torch 依赖）。

只缩放训练用的 shaped reward，不改环境返回给日志/评测的原始回合回报；
均匀正缩放不改变最优策略，但把价值目标的量级压到 O(1)，避免回合越长、
惩罚越大时 value loss 随量级平方放大。
"""

from __future__ import annotations

import numpy as np


class RunningStd:
    """在线标准差估计（Welford），样本数不足 2 时退回 1.0。"""

    def __init__(self, eps: float = 1e-8):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.eps = float(eps)

    def update(self, x) -> "RunningStd":
        values = np.asarray(x, dtype=np.float64).ravel()
        count = values.size
        if count == 0:
            return self
        batch_mean = float(values.mean())
        batch_m2 = float(((values - batch_mean) ** 2).sum())
        total = self.n + count
        delta = batch_mean - self.mean
        self.mean += delta * count / total
        self.m2 += batch_m2 + delta**2 * self.n * count / total
        self.n = total
        return self

    @property
    def std(self) -> float:
        if self.n < 2:
            return 1.0
        return float(max(np.sqrt(self.m2 / (self.n - 1)), self.eps))

    def scale(self, min_std: float = 1.0) -> float:
        """缩放系数 = 1/std，但只做缩小不做放大，避免小奖励被放大成爆炸值。"""
        return 1.0 / max(self.std, float(min_std))
