"""并行环境封装：把 B 个环境堆成 [B, N, ...]（负责人 B）。"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from .formation_comm import EnvConfig, EnvStep, FormationCommEnv


class VecFormationEnv:
    """B 个同构环境的批量化封装，回合结束自动 reset。"""

    def __init__(self, n_envs: int, cfg: EnvConfig, seed: int = 0):
        self.n_envs = int(n_envs)
        if self.n_envs < 1:
            raise ValueError("n_envs must be positive")
        self.cfg = cfg
        self.envs = []
        try:
            for _ in range(self.n_envs):
                self.envs.append(FormationCommEnv(cfg))
        except Exception:
            self.close()
            raise
        self._seeds = [seed + 1000 * k for k in range(self.n_envs)]
        self._eps_stats = [
            {"ret": 0.0, "len": 0, "coll": 0, "success": False}
            for _ in range(self.n_envs)
        ]

    @property
    def n_agents(self) -> int:
        return self.cfg.n_agents

    @property
    def obs_dim(self) -> int:
        return self.envs[0].obs_dim

    @property
    def state_dim(self) -> int:
        return self.envs[0].state_dim

    def reset(self) -> dict:
        """复现初始批次，obs [B,N,D]；回合内自动 reset 不重新播种。"""
        steps = [e.reset(seed=s) for e, s in zip(self.envs, self._seeds)]
        for st in self._eps_stats:
            st.update({"ret": 0.0, "len": 0, "coll": 0, "success": False})
        return self._stack(steps)

    def step(self, actions: np.ndarray) -> dict:
        """actions [B,N]；终止转移的奖励与统计和新回合观测一起返回。"""
        actions = np.asarray(actions)
        if actions.shape != (self.n_envs, self.n_agents):
            raise ValueError(f"expected actions [{self.n_envs}, {self.n_agents}]")
        if not np.issubdtype(actions.dtype, np.integer) or np.any(
            (actions < 0) | (actions > 4)
        ):
            raise ValueError("actions must be integers in 0..4")
        steps = []
        for k, env in enumerate(self.envs):
            st = env.step(actions[k])
            cur = self._eps_stats[k]
            cur["ret"] += float(np.mean(st.reward))
            cur["len"] += 1
            cur["coll"] += st.info["collisions"]
            cur["success"] = cur["success"] or st.info["success"]
            if st.done:
                st.info["ep"] = {
                    "ret": cur["ret"],
                    "len": cur["len"],
                    "collisions": cur["coll"],
                    "success": cur["success"],
                    "formation_err": st.info["formation_err"],
                }
                terminal = st
                initial = env.reset()  # 延续 RNG，保持复现且不重复同一个回合。
                initial.info["ep"] = terminal.info["ep"]
                initial.info["terminal_step"] = terminal
                st = replace(initial, reward=terminal.reward, done=True)
                self._eps_stats[k] = {"ret": 0.0, "len": 0, "coll": 0, "success": False}
            else:
                st.info["ep"] = None
            steps.append(st)
        return self._stack(steps)

    def close(self) -> None:
        """关闭全部 B 个环境。"""
        for env in self.envs:
            env.close()

    def __enter__(self) -> VecFormationEnv:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def _stack(self, steps: list[EnvStep]) -> dict:
        return {
            "obs": np.stack([s.obs for s in steps], axis=0),
            "obs_mask": np.stack([s.obs_mask for s in steps], axis=0),
            "state": np.stack([s.state for s in steps], axis=0),
            "comm_mask": np.stack([s.comm_mask for s in steps], axis=0),
            "reward": np.stack([s.reward for s in steps], axis=0),
            "channel": {
                k: np.asarray([s.info["channel"][k] for s in steps], dtype=np.float32)
                for k in steps[0].info["channel"]
            },
            "done": np.asarray([s.done for s in steps], dtype=bool),
            "ep": [s.info.get("ep") for s in steps],
            "info_raw": steps,
        }
