"""环境封装（负责人 B）。

接口见 docs/contracts.md 第 2 节。两种后端共用观测、指标和奖励：
  * ``backend="mock"``：numpy 内置质点环境，零依赖，用于契约冒烟与模块离线开发；
  * ``backend="mpe2"``：MPE2 simple_formation_v1 的真实物理；
  * ``backend="auto"``：有 mpe2 用 mpe2，否则退回 mock 并打印提示。

环境只负责物理、观测与团队奖励；门控、消息、信道噪声都在模型侧。
"""

from __future__ import annotations

import importlib
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import metrics

ACTIONS = {0: (0.0, 0.0), 1: (-1.0, 0.0), 2: (1.0, 0.0), 3: (0.0, -1.0), 4: (0.0, 1.0)}


@dataclass
class EnvConfig:
    scene: str = "formation"
    backend: str = "mpe2"
    n_agents: int = 6
    max_steps: int = 100
    formation_radius: float = 0.5
    obs_radius: float = 0.8
    n_obs: int = 4
    comm_radius: float = 0.8
    recv_k: int = 3
    safe_dist: float = 0.2
    collision_coef: float = 1.0
    max_speed: float = 1.0
    success_dist: float = 0.05
    success_hold: int = 10
    heading: float = 0.0
    end_on_success: bool = False
    channel: dict = field(
        default_factory=lambda: {"drop_p": [0.0, 0.3], "flip_p": [0.0, 0.1]}
    )

    def validate(self) -> None:
        """检查粒子数、距离、时限和信道区间，避免生成无效张量。"""
        for name in ("n_agents", "max_steps", "success_hold"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("n_obs", "recv_k"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in (
            "formation_radius",
            "obs_radius",
            "comm_radius",
            "safe_dist",
            "collision_coef",
            "success_dist",
            "max_speed",
        ):
            value = getattr(self, name)
            if (
                not np.isfinite(value)
                or value < 0
                or (name == "max_speed" and value == 0)
            ):
                raise ValueError(f"invalid {name}: {value}")
        if not np.isfinite(self.heading):
            raise ValueError("heading must be finite")
        for key in ("drop_p", "flip_p"):
            value = self.channel.get(key, 0.0)
            bounds = np.asarray(value, dtype=float)
            if bounds.ndim > 1 or (bounds.ndim == 1 and bounds.shape != (2,)):
                raise ValueError(f"channel.{key} must be a scalar or [low, high]")
            if not np.all(np.isfinite(bounds) & (bounds >= 0) & (bounds <= 1)):
                raise ValueError(f"channel.{key} must be in [0, 1]")
            if bounds.ndim == 1 and bounds[0] > bounds[1]:
                raise ValueError(f"channel.{key} requires low <= high")

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
    obs: np.ndarray  # [N, 4 + 2*n_obs]  float32
    obs_mask: np.ndarray  # [N, n_obs]        bool
    state: np.ndarray  # [D_state]         float32
    comm_mask: np.ndarray  # [N, N]            bool
    reward: np.ndarray  # [N]               float32
    done: bool
    info: dict


class FormationCommEnv:
    """N 架同构无人机的编队通信环境。"""

    def __init__(self, cfg: EnvConfig):
        cfg.validate()
        if cfg.scene != "formation":
            raise NotImplementedError(
                f"scene={cfg.scene!r} is not implemented; use formation"
            )
        self.cfg = cfg
        self._rng = np.random.default_rng(0)
        self._t = 0
        self._pos = np.zeros((cfg.n_agents, 2), dtype=np.float32)
        self._vel = np.zeros((cfg.n_agents, 2), dtype=np.float32)
        self._center = np.zeros(2, dtype=np.float32)
        self._hold = 0
        self._collisions_ep = 0
        self._channel = {"drop_p": 0.0, "flip_p": 0.0}
        self._mpe_env: Any | None = None
        self._agent_names: tuple[str, ...] = ()
        self._closed = False
        self._needs_reset = True
        self.backend = self._resolve_backend(cfg.backend)

    # ---- 后端选择 -------------------------------------------------
    def _resolve_backend(self, backend: str) -> str:
        if backend == "mock":
            return "mock"
        if backend not in ("auto", "mpe2"):
            raise ValueError(f"unknown backend: {backend!r}")
        try:
            scene = importlib.import_module("mpe2.simple_formation_v1")
        except ModuleNotFoundError as exc:
            if exc.name != "mpe2":
                raise  # 已安装但依赖或场景损坏，不能悄悄切换物理后端。
            if backend == "mpe2":
                raise RuntimeError(
                    "未安装 mpe2，请 pip install mpe2 或改用 backend=mock"
                ) from exc
            warnings.warn(
                "mpe2 is not installed; using mock physics",
                RuntimeWarning,
                stacklevel=3,
            )
            return "mock"
        self._mpe_env = scene.parallel_env(
            N=self.n_agents,
            max_cycles=self.cfg.max_steps,
            continuous_actions=False,
            terminate_on_success=False,
        )
        try:
            self._agent_names = tuple(self._mpe_env.possible_agents)
            world = self._mpe_env.unwrapped.world
            if (
                len(self._agent_names) != self.n_agents
                or tuple(a.name for a in world.agents) != self._agent_names
                or len(world.landmarks) != 1
            ):
                raise RuntimeError("incompatible MPE2 agent order or landmark count")
            for name, agent in zip(self._agent_names, world.agents):
                if self._mpe_env.observation_space(name).shape != (6,):
                    raise RuntimeError(f"unexpected MPE2 observation space for {name}")
                space = self._mpe_env.action_space(name)
                if space.shape != () or getattr(space, "n", None) != 5:
                    raise RuntimeError(f"expected MPE2 Discrete(5) actions for {name}")
                agent.max_speed = self.cfg.max_speed
        except Exception:
            self._mpe_env.close()
            raise
        return "mpe2"

    def _sync_world(self) -> None:
        """读取 world 的位置/速度 [N, 2] 与中心 [2]；不使用原生奖励或观测。"""
        world = self._mpe_env.unwrapped.world
        self._pos = np.asarray([a.state.p_pos for a in world.agents], dtype=np.float32)
        self._vel = np.asarray([a.state.p_vel for a in world.agents], dtype=np.float32)
        self._center = np.asarray(
            world.landmarks[0].state.p_pos, dtype=np.float32
        ).copy()

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
            v = np.asarray(cfg.get(key, default), dtype=float)
            if v.ndim == 1:
                return float(self._rng.uniform(v[0], v[1]))
            return float(v)

        return {"drop_p": _pick("drop_p", 0.0), "flip_p": _pick("flip_p", 0.0)}

    def reset(self, seed: int | None = None) -> EnvStep:
        """开始新回合，返回 obs [N, 4+2*n_obs]；同一 seed 可复现。"""
        if self._closed:
            raise RuntimeError("environment is closed")
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        cfg = self.cfg
        self._t = 0
        self._hold = 0
        self._collisions_ep = 0
        self._channel = self._sample_channel()
        if self.backend == "mpe2":
            self._mpe_env.reset(seed=seed)
            self._sync_world()
            self._needs_reset = False
            return self._obs_step(
                reward=np.zeros(cfg.n_agents, dtype=np.float32),
                done=False,
                success=False,
            )
        self._center = np.array([0.0, 0.0], dtype=np.float32)
        # 初始散布在半径 [1.5*r_f, 3*r_f] 的环内
        r = self._rng.uniform(
            1.5 * cfg.formation_radius, 3.0 * cfg.formation_radius, size=cfg.n_agents
        )
        ang = self._rng.uniform(0.0, 2.0 * np.pi, size=cfg.n_agents)
        self._pos = np.stack([r * np.cos(ang), r * np.sin(ang)], axis=1).astype(
            np.float32
        )
        self._vel = np.zeros_like(self._pos)
        self._needs_reset = False
        return self._obs_step(
            reward=np.zeros(cfg.n_agents, dtype=np.float32), done=False, success=False
        )

    def step(self, action: np.ndarray) -> EnvStep:
        """推进物理，action [N] 整数 0..4，返回统一 EnvStep。"""
        if self._closed or self._needs_reset:
            raise RuntimeError("call reset() before stepping a new episode")
        cfg = self.cfg
        action = np.asarray(action)
        if (
            action.shape != (cfg.n_agents,)
            or not np.issubdtype(action.dtype, np.integer)
            or np.any((action < 0) | (action > 4))
        ):
            raise ValueError(
                f"action must be an integer array of shape ({cfg.n_agents},) in 0..4"
            )
        backend_done = False
        if self.backend == "mpe2":
            _, _, term, trunc, _ = self._mpe_env.step(
                {name: int(a) for name, a in zip(self._agent_names, action)}
            )
            backend_done = any(term.values()) or any(trunc.values())
            self._sync_world()
        else:
            acc = np.array([ACTIONS[int(a)] for a in action], dtype=np.float32) * 0.5
            self._vel = 0.5 * self._vel + 0.5 * np.clip(
                self._vel + acc, -cfg.max_speed, cfg.max_speed
            )
            speed = np.linalg.norm(self._vel, axis=-1, keepdims=True)
            self._vel /= np.maximum(1.0, speed / cfg.max_speed)
            self._pos = (self._pos + self._vel).astype(np.float32)
        self._t += 1

        ferr = metrics.formation_error(
            self._pos, self._center, cfg.formation_radius, cfg.heading
        )
        coll = metrics.count_collisions(self._pos, cfg.safe_dist)
        self._collisions_ep += coll

        success = metrics.all_in_slots(
            self._pos, self._center, cfg.formation_radius, cfg.success_dist, cfg.heading
        )
        self._hold = self._hold + 1 if success else 0
        done_task = self._hold >= cfg.success_hold
        done = bool(
            backend_done
            or self._t >= cfg.max_steps
            or (done_task and cfg.end_on_success)
        )
        self._needs_reset = done

        reward = np.full(
            cfg.n_agents, -(ferr + cfg.collision_coef * coll), dtype=np.float32
        )
        return self._obs_step(
            reward=reward, done=done, success=done_task, ferr=ferr, coll=coll
        )

    def close(self) -> None:
        """释放真实后端的资源；重复调用安全。"""
        if not self._closed and self._mpe_env is not None:
            self._mpe_env.close()
        self._closed = True

    def __enter__(self) -> FormationCommEnv:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    # ---- 观测组装 -------------------------------------------------
    def _obs_step(
        self,
        reward: np.ndarray,
        done: bool,
        success: bool,
        ferr: float | None = None,
        coll: int = 0,
    ) -> EnvStep:
        """两种物理共用 obs [N, 4+2*n_obs]、state [4*N+2]、mask [N,N]。"""
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
            cand = cand[np.argsort(d[i, cand], kind="stable")][: cfg.n_obs]
            for slot, j in enumerate(cand):
                obs[i, 4 + 2 * slot : 6 + 2 * slot] = pos[j] - pos[i]
                obs_mask[i, slot] = True

        comm_mask = np.zeros((n, n), dtype=bool)
        for i in range(n):
            cand = np.where(d[i] <= cfg.comm_radius)[0]
            cand = cand[np.argsort(d[i, cand], kind="stable")][: cfg.recv_k]
            comm_mask[i, cand] = True

        state = np.concatenate(
            [pos.reshape(-1), self._vel.reshape(-1), self._center.reshape(-1)], axis=0
        ).astype(np.float32)

        if ferr is None:
            ferr = metrics.formation_error(
                pos, self._center, cfg.formation_radius, cfg.heading
            )

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
            "backend": self.backend,
        }
        return EnvStep(
            obs=obs,
            obs_mask=obs_mask,
            state=state,
            comm_mask=comm_mask,
            reward=np.asarray(reward, dtype=np.float32),
            done=bool(done),
            info=info,
        )
