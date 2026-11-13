"""环境封装（负责人 B）。

接口见 docs/contracts.md 第 2 节。骨架同时提供：
  * ``backend="mock"``：numpy 内置质点环境，零依赖，用于契约冒烟与模块离线开发；
  * ``backend="mpe2"``：MPE2 simple_formation 适配位（待 B 补全，形状须逐项核对）；
  * ``backend="auto"``：有 mpe2 用 mpe2，否则退回 mock 并打印提示。

环境只负责物理、观测与团队奖励；门控、消息、信道噪声都在模型侧。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import metrics

ACTIONS = {0: (0.0, 0.0), 1: (-1.0, 0.0), 2: (1.0, 0.0), 3: (0.0, -1.0), 4: (0.0, 1.0)}


@dataclass
class EnvConfig:
    scene: str = "formation"
    backend: str = "auto"
    n_agents: int = 6
    max_steps: int = 100
    formation_radius: float = 0.5
    obs_radius: float = 0.5
    n_obs: int = 4
    comm_radius: float = 0.4
    recv_k: int = 3
    safe_dist: float = 0.2
    collision_coef: float = 1.0
    max_speed: float = 1.0
    success_dist: float = 0.05
    success_hold: int = 10
    heading: float = 0.0
    end_on_success: bool = False
    channel: dict = field(default_factory=lambda: {"drop_p": [0.0, 0.3], "flip_p": [0.0, 0.1]})

    @classmethod
    def from_dict(cls, d: dict) -> "EnvConfig":
        d = dict(d or {})
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in fields})

    @property
    def slot_angle(self) -> float:
        return 2.0 * np.pi / max(self.n_agents, 1)


@dataclass
class EnvStep:
    obs: np.ndarray        # [N, 4 + 2*n_obs]  float32
    obs_mask: np.ndarray   # [N, n_obs]        bool
    state: np.ndarray      # [D_state]         float32
    comm_mask: np.ndarray  # [N, N]            bool
    reward: np.ndarray     # [N]               float32
    done: bool
    info: dict


class FormationCommEnv:
    """N 架同构无人机的编队通信环境。"""

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self._rng = np.random.default_rng(0)
        self._t = 0
        self._pos = np.zeros((cfg.n_agents, 2), dtype=np.float32)
        self._vel = np.zeros((cfg.n_agents, 2), dtype=np.float32)
        self._center = np.zeros(2, dtype=np.float32)
        self._hold = 0
        self._collisions_ep = 0
        self._channel = {"drop_p": 0.0, "flip_p": 0.0}
        self.backend = self._resolve_backend(cfg.backend)

    # ---- 后端选择 -------------------------------------------------
    def _resolve_backend(self, backend: str) -> str:
        if backend == "mock":
            return "mock"
        if backend == "mpe2":
            try:
                import mpe2  # noqa: F401
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("未安装 mpe2，请 pip install mpe2 或改用 backend=mock") from exc
            raise NotImplementedError(
                "mpe2 适配由负责人 B 实现：需把 simple_formation 的 obs/state 映射到 docs/contracts.md 第 2 节形状"
            )
        # auto
        if "mpe2" in self.__dict__.get("_probed", {}):
            return "mpe2"
        try:
            import mpe2  # noqa: F401
        except Exception:
            return "mock"
        return "mpe2"

    @property
    def obs_dim(self) -> int:
        return 4 + 2 * self.cfg.n_obs

    @property
    def state_dim(self) -> int:
        return 4 * self.cfg.n_agents + 2

    @property
    def n_agents(self) -> int:
        return self.cfg.n_agents

    # ---- 采样 -----------------------------------------------------
    def _sample_channel(self) -> dict:
        cfg = self.cfg.channel

        def _pick(key: str, default: float) -> float:
            v = cfg.get(key, default)
            if isinstance(v, (list, tuple)) and len(v) == 2:
                return float(self._rng.uniform(v[0], v[1]))
            return float(v)

        return {"drop_p": _pick("drop_p", 0.0), "flip_p": _pick("flip_p", 0.0)}

    def reset(self, seed: int | None = None) -> EnvStep:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        cfg = self.cfg
        self._t = 0
        self._hold = 0
        self._collisions_ep = 0
        self._channel = self._sample_channel()
        self._center = np.array([0.0, 0.0], dtype=np.float32)
        # 初始散布在半径 [1.5*r_f, 3*r_f] 的环内
        r = self._rng.uniform(1.5 * cfg.formation_radius, 3.0 * cfg.formation_radius, size=cfg.n_agents)
        ang = self._rng.uniform(0.0, 2.0 * np.pi, size=cfg.n_agents)
        self._pos = np.stack([r * np.cos(ang), r * np.sin(ang)], axis=1).astype(np.float32)
        self._vel = np.zeros_like(self._pos)
        return self._obs_step(reward=np.zeros(cfg.n_agents, dtype=np.float32), done=False, success=False)

    def step(self, action: np.ndarray) -> EnvStep:
        cfg = self.cfg
        action = np.asarray(action).astype(np.int64).reshape(cfg.n_agents)
        acc = np.array([ACTIONS[int(a)] for a in action], dtype=np.float32) * 0.5
        self._vel = 0.5 * self._vel + 0.5 * np.clip(self._vel + acc, -cfg.max_speed, cfg.max_speed)
        self._pos = (self._pos + self._vel).astype(np.float32)
        self._t += 1

        ferr = metrics.formation_error(self._pos, self._center, cfg.formation_radius, cfg.heading)
        coll = metrics.count_collisions(self._pos, cfg.safe_dist)
        self._collisions_ep += coll

        success = metrics.all_in_slots(
            self._pos, self._center, cfg.formation_radius, cfg.success_dist, cfg.heading
        )
        self._hold = self._hold + 1 if success else 0
        done_task = self._hold >= cfg.success_hold
        done = bool(self._t >= cfg.max_steps or (done_task and cfg.end_on_success))

        reward = np.full(cfg.n_agents, -(ferr + cfg.collision_coef * coll), dtype=np.float32)
        return self._obs_step(reward=reward, done=done, success=done_task, ferr=ferr, coll=coll)

    # ---- 观测组装 -------------------------------------------------
    def _obs_step(self, reward, done, success, ferr: float | None = None, coll: int = 0) -> EnvStep:
        cfg = self.cfg
        n = cfg.n_agents
        pos = self._pos
        d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)  # [N, N]
        np.fill_diagonal(d, np.inf)

        obs = np.zeros((n, self.obs_dim), dtype=np.float32)
        obs_mask = np.zeros((n, cfg.n_obs), dtype=bool)
        for i in range(n):
            obs[i, 0:2] = self._vel[i]
            obs[i, 2:4] = self._center - pos[i]
            cand = np.where(d[i] <= cfg.obs_radius)[0]
            cand = cand[np.argsort(d[i, cand])][: cfg.n_obs]
            for slot, j in enumerate(cand):
                obs[i, 4 + 2 * slot: 6 + 2 * slot] = pos[j] - pos[i]
                obs_mask[i, slot] = True

        comm_mask = np.zeros((n, n), dtype=bool)
        for i in range(n):
            cand = np.where(d[i] <= cfg.comm_radius)[0]
            cand = cand[np.argsort(d[i, cand])][: cfg.recv_k]
            comm_mask[i, cand] = True

        state = np.concatenate(
            [pos.reshape(-1), self._vel.reshape(-1), self._center.reshape(-1)], axis=0
        ).astype(np.float32)

        if ferr is None:
            ferr = metrics.formation_error(pos, self._center, cfg.formation_radius, cfg.heading)

        info = {
            "formation_err": float(ferr),
            "collisions": int(coll),
            "collisions_ep": int(self._collisions_ep),
            "success": bool(success),
            "hold_steps": int(self._hold),
            "t": int(self._t),
            "channel": dict(self._channel),
            "center": self._center.copy(),
            "pos": pos.copy(),
        }
        return EnvStep(
            obs=obs, obs_mask=obs_mask, state=state, comm_mask=comm_mask,
            reward=np.asarray(reward, dtype=np.float32), done=bool(done), info=info,
        )
