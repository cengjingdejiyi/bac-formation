"""并行环境封装：把 B 个环境堆成 [B, N, ...]（负责人 B）。"""
from __future__ import annotations

import numpy as np

from .formation_comm import EnvConfig, FormationCommEnv


class VecFormationEnv:
    """B 个同构环境的批量化封装，回合结束自动 reset。"""

    def __init__(self, n_envs: int, cfg: EnvConfig, seed: int = 0):
        self.n_envs = int(n_envs)
        self.cfg = cfg
        self.envs = [FormationCommEnv(cfg) for _ in range(self.n_envs)]
        self._seeds = [seed + 1000 * k for k in range(self.n_envs)]
        self._eps_stats = [{"ret": 0.0, "len": 0, "coll": 0, "success": False} for _ in range(self.n_envs)]

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
        steps = [e.reset(seed=s) for e, s in zip(self.envs, self._seeds)]
        for st in self._eps_stats:
            st.update({"ret": 0.0, "len": 0, "coll": 0, "success": False})
        return self._stack(steps)

    def step(self, actions: np.ndarray) -> dict:
        assert actions.shape == (self.n_envs, self.n_agents), actions.shape
        steps, done = [], []
        for k, env in enumerate(self.envs):
            st = env.step(actions[k])
            cur = self._eps_stats[k]
            cur["ret"] += float(np.mean(st.reward))
            cur["len"] += 1
            cur["coll"] += st.info["collisions"]
            cur["success"] = cur["success"] or st.info["success"]
            done.append(st.done)
            if st.done:
                st.info["ep"] = {"ret": cur["ret"], "len": cur["len"],
                                 "collisions": cur["coll"], "success": cur["success"]}
                st = env.reset(seed=self._seeds[k])  # 自动 reset
                self._eps_stats[k] = {"ret": 0.0, "len": 0, "coll": 0, "success": False}
            else:
                st.info["ep"] = None
            steps.append(st)
        out = self._stack(steps)
        out["done"] = np.asarray(done, dtype=bool)
        return out

    def _stack(self, steps) -> dict:
        return {
            "obs": np.stack([s.obs for s in steps], axis=0),
            "obs_mask": np.stack([s.obs_mask for s in steps], axis=0),
            "state": np.stack([s.state for s in steps], axis=0),
            "comm_mask": np.stack([s.comm_mask for s in steps], axis=0),
            "reward": np.stack([s.reward for s in steps], axis=0),
            "channel": {k: np.asarray([s.info["channel"][k] for s in steps], dtype=np.float32)
                        for k in steps[0].info["channel"]},
            "done": np.zeros(len(steps), dtype=bool),
            "ep": [s.info.get("ep") for s in steps],
            "info_raw": steps,
        }
