"""GAE 优势估计：全仓库唯一实现（numpy，不依赖 torch）。

`RolloutBuffer.compute_returns` 直接调用本函数，训练与测试走同一条路径；
此前 mappo.gae 与 buffer.compute_returns 各写一份，改一处漏一处。
"""

from __future__ import annotations

import numpy as np


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    lam: float,
    last_value: np.ndarray,
    n_steps: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """按真实转移计算优势 adv 与回报 ret，形状均与 rewards 一致 [T, B, N]。

    rewards/values [T, B, N]；dones [T, B]；last_value [B, N]（段末 next state 的 value）。
    done[t] 同时切断 GAE 自举与跨回合优势传播；段末非终止环境用 last_value 自举。
    n_steps 用于只计算缓存中已写入的前 n 步（默认整段）。
    """
    total = len(rewards) if n_steps is None else int(n_steps)
    adv = np.zeros_like(rewards)
    ret = np.zeros_like(rewards)
    g = np.zeros_like(values[0], dtype=np.float32)
    for t in reversed(range(total)):
        next_v = last_value if t == total - 1 else values[t + 1]
        nonterm = 1.0 - dones[t].astype(np.float32)[:, None]
        delta = rewards[t] + gamma * next_v * nonterm - values[t]
        g = delta + gamma * lam * nonterm * g
        adv[t] = g
        ret[t] = g + values[t]
    return adv, ret
