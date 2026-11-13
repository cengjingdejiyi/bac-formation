"""指标与槽位匹配（负责人 B）。

槽位定义见开题报告 1.4：以编队中心 p^c 为圆心、半径 r_f 的圆上 N 个等角槽位。
"""

from __future__ import annotations

import numpy as np

try:  # 无 scipy 时退化为贪心匹配，保证骨架在任何环境可跑
    from scipy.optimize import linear_sum_assignment

    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False


def formation_heading(pos: np.ndarray, center) -> float:
    """MPE2 朝向：相对中心极角 [0, 2π) 最小的无人机，pos [N,2]。"""
    relative = np.asarray(pos, dtype=np.float64) - np.asarray(center, dtype=np.float64)
    angles = np.mod(np.arctan2(relative[:, 1], relative[:, 0]), 2.0 * np.pi)
    return float(angles.min())


def slot_positions(
    center, radius: float, n_agents: int, heading: float = 0.0
) -> np.ndarray:
    """圆上 N 个槽位坐标，返回 [N, 2]。

    heading 为编队朝向 θ，缺省 0（沿 +x 轴起排）。
    """
    center = np.asarray(center, dtype=np.float32).reshape(2)
    ang = heading + 2.0 * np.pi * np.arange(n_agents) / max(n_agents, 1)
    return center[None, :] + radius * np.stack(
        [np.cos(ang), np.sin(ang)], axis=1
    ).astype(np.float32)


def match_slots(pos: np.ndarray, slots: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """把 N 架无人机匹配到 N 个槽位，返回 (perm, dist)。

    perm[i] 表示无人机 i 分到的槽位下标；dist[i] 为对应距离。用匈牙利算法，无 scipy 退贪心。
    """
    pos = np.asarray(pos, dtype=np.float32)
    cost = np.linalg.norm(pos[:, None, :] - slots[None, :, :], axis=-1)  # [N, N]
    if _HAS_SCIPY:
        rows, cols = linear_sum_assignment(cost)
        perm = np.zeros(len(pos), dtype=np.int64)
        perm[rows] = cols
    else:  # pragma: no cover
        perm = np.full(len(pos), -1, dtype=np.int64)
        free = list(range(len(slots)))
        for i in np.argsort(cost.min(axis=1)):
            j = min(free, key=lambda k: cost[i, k])
            perm[i] = j
            free.remove(j)
    dist = cost[np.arange(len(pos)), perm]
    return perm, dist


def formation_error(
    pos: np.ndarray, center, radius: float, heading: float = 0.0
) -> float:
    """编队误差 e_t：槽位匹配后的平均距离。"""
    slots = slot_positions(center, radius, len(pos), heading)
    _, dist = match_slots(pos, slots)
    return float(np.mean(dist))


def count_collisions(pos: np.ndarray, safe_dist: float) -> int:
    """违反安全距离的机对数（每步计数，回合内累加）。"""
    pos = np.asarray(pos, dtype=np.float32)
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    n = len(pos)
    upper = np.triu(np.ones((n, n), dtype=bool), 1)
    return int(np.sum((d < safe_dist) & upper))


def all_in_slots(
    pos: np.ndarray, center, radius: float, dist_thr: float, heading: float = 0.0
) -> bool:
    """所有无人机与各自槽位的距离是否都不超过 dist_thr（完成判定的单步条件）。"""
    slots = slot_positions(center, radius, len(pos), heading)
    _, dist = match_slots(pos, slots)
    return bool(np.all(dist <= dist_thr))
