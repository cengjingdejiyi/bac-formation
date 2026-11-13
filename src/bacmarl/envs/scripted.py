"""脚本控制器：参数标定与 demo 用（负责人 B）。

不可学习，仅靠局部观测把无人机收拢到环上，用于标定 R_obs / R_comm / d_safe 的可行区间。
"""
from __future__ import annotations

import numpy as np


def _dir_to_action(vec: np.ndarray) -> int:
    """把二维方向向量离散成 MPE2 的 5 个动作之一。"""
    if np.linalg.norm(vec) < 1e-6:
        return 0
    if abs(vec[0]) >= abs(vec[1]):
        return 2 if vec[0] > 0 else 1
    return 4 if vec[1] > 0 else 3


class ScriptedController:
    """朝“以编队中心为圆心、半径 r 的环上自身当前射线方向”收敛。"""

    def __init__(self, n_agents: int, radius: float = 0.5, dead_zone: float = 0.02):
        self.n_agents = n_agents
        self.radius = radius
        self.dead_zone = dead_zone

    def act(self, obs: np.ndarray, obs_mask: np.ndarray | None = None) -> np.ndarray:
        # obs[:, 0:2] 自身速度；obs[:, 2:4] 中心相对位置（center - pos）
        rel_center = obs[:, 2:4]                    # = center - pos
        pos_from_center = -rel_center               # = pos - center
        cur_r = np.linalg.norm(pos_from_center, axis=1, keepdims=True) + 1e-8
        radial = pos_from_center / cur_r            # 单位径向
        target = radial * self.radius               # 环上同射线点（相对中心）
        err = target - pos_from_center              # 需要移动的向量
        acts = np.zeros(self.n_agents, dtype=np.int64)
        for i in range(self.n_agents):
            if np.linalg.norm(err[i]) <= self.dead_zone:
                acts[i] = 0
            else:
                acts[i] = _dir_to_action(err[i])
        return acts
