"""B 参数标定：真实环境的随机/悬停/局部控制与全局槽位可行性诊断。

python scripts/calibrate_env.py --env-config formation_calibrated
slot_oracle 使用全局位置，只验证动力学/阈值可行性，不是分布式算法基线。
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import platform
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bacmarl.envs import EnvConfig, EnvStep, FormationCommEnv  # noqa: E402
from bacmarl.envs.metrics import match_slots, slot_positions  # noqa: E402
from bacmarl.envs.scripted import ScriptedController, _dir_to_action  # noqa: E402
from bacmarl.utils.config import load_config  # noqa: E402


class SlotOracle:
    """固定匈牙利槽位目标 + 速度反馈；需要全局位置 [N,2]。仅作标定。"""

    def __init__(self, cfg: EnvConfig, initial: EnvStep):
        slots = slot_positions(
            initial.info["center"], cfg.formation_radius, cfg.n_agents, cfg.heading
        )
        perm, _ = match_slots(initial.info["pos"], slots)
        self.targets = slots[perm]

    def act(self, step: EnvStep) -> np.ndarray:
        """输出 [N] int64；0.2 速度反馈与 0.03 死区针对 MPE2 dt=0.1。"""
        correction = self.targets - step.info["pos"] - 0.2 * step.obs[:, :2]
        return np.asarray(
            [0 if np.linalg.norm(e) < 0.03 else _dir_to_action(e) for e in correction],
            dtype=np.int64,
        )


def ideal_degrees(cfg: EnvConfig, radius: float, limit: int) -> np.ndarray:
    """理想槽位上每架无人机的可用邻居数 [N]，不包含自己。"""
    pos = slot_positions([0, 0], cfg.formation_radius, cfg.n_agents, cfg.heading)
    distance = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    np.fill_diagonal(distance, np.inf)
    return np.minimum(np.sum(distance <= radius, axis=1), limit)


def run_episode(cfg: EnvConfig, seed: int, policy: str) -> dict:
    """输出单回合标量指标；返回/碰撞统计覆盖整个回合。"""
    rng = np.random.default_rng(seed)
    with FormationCommEnv(cfg) as env:
        step = env.reset(seed=seed)
        initial_error = step.info["formation_err"]
        radial = ScriptedController(cfg.n_agents, cfg.formation_radius)
        oracle = SlotOracle(cfg, step)
        total_return, first_success = 0.0, None
        obs_degrees, comm_degrees, isolated = [], [], []
        while not step.done:
            obs_degrees.append(float(step.obs_mask.sum(axis=-1).mean()))
            degrees = step.comm_mask.sum(axis=-1)
            comm_degrees.append(float(degrees.mean()))
            isolated.append(float(np.mean(degrees == 0)))
            if policy == "hover":
                action = np.zeros(cfg.n_agents, dtype=np.int64)
            elif policy == "random":
                action = rng.integers(0, 5, size=cfg.n_agents, dtype=np.int64)
            elif policy == "radial":
                action = radial.act(step.obs, step.obs_mask)
            elif policy == "slot_oracle":
                action = oracle.act(step)
            else:
                raise ValueError(f"unknown policy: {policy}")
            step = env.step(action)
            total_return += float(step.reward.mean())
            if step.info["success"] and first_success is None:
                first_success = step.info["t"]

        slots = slot_positions(
            step.info["center"], cfg.formation_radius, cfg.n_agents, cfg.heading
        )
        _, distances = match_slots(step.info["pos"], slots)
        return {
            "backend": env.backend,
            "n_agents": cfg.n_agents,
            "seed": seed,
            "policy": policy,
            "steps": step.info["t"],
            "initial_error": initial_error,
            "formation_err": step.info["formation_err"],
            "max_slot_dist": float(distances.max()),
            "return": total_return,
            "collisions": step.info["collisions_ep"],
            "success": first_success is not None,
            "first_success_step": first_success,
            "obs_neighbors": float(np.mean(obs_degrees)),
            "comm_neighbors": float(np.mean(comm_degrees)),
            "isolated_fraction": float(np.mean(isolated)),
            "ideal_obs_neighbors": float(
                ideal_degrees(cfg, cfg.obs_radius, cfg.n_obs).mean()
            ),
            "ideal_comm_neighbors": float(
                ideal_degrees(cfg, cfg.comm_radius, cfg.recv_k).mean()
            ),
            "drop_p": step.info["channel"]["drop_p"],
            "flip_p": step.info["channel"]["flip_p"],
        }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-config", default="formation")
    parser.add_argument("--backend", choices=["mpe2", "mock"], default="mpe2")
    parser.add_argument("--agents", nargs="+", type=int, default=[4, 6, 8])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=["hover", "random", "radial", "slot_oracle"],
        default=["hover", "random", "radial", "slot_oracle"],
    )
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "calibration")
    args = parser.parse_args(argv)
    if not (ROOT / "configs" / "env" / f"{args.env_config}.yaml").is_file():
        parser.error(f"unknown env config: {args.env_config}")
    base = EnvConfig.from_dict(load_config([f"env={args.env_config}"])["env"])
    base = replace(
        base,
        backend=args.backend,
        max_steps=base.max_steps if args.steps is None else args.steps,
    )
    rows = []
    for n in args.agents:
        cfg = replace(base, n_agents=n)
        for policy in args.policies:
            for seed in args.seeds:
                rows.append(run_episode(cfg, seed, policy))

    summary = []
    for n in args.agents:
        for policy in args.policies:
            group = [r for r in rows if r["n_agents"] == n and r["policy"] == policy]
            summary.append(
                {
                    "n_agents": n,
                    "policy": policy,
                    "episodes": len(group),
                    "success_rate": float(np.mean([r["success"] for r in group])),
                    "formation_err_mean": float(
                        np.mean([r["formation_err"] for r in group])
                    ),
                    "formation_err_std": float(
                        np.std([r["formation_err"] for r in group])
                    ),
                    "return_mean": float(np.mean([r["return"] for r in group])),
                    "collisions_mean": float(np.mean([r["collisions"] for r in group])),
                    "comm_neighbors_mean": float(
                        np.mean([r["comm_neighbors"] for r in group])
                    ),
                    "isolated_fraction_mean": float(
                        np.mean([r["isolated_fraction"] for r in group])
                    ),
                    "ideal_obs_neighbors": group[0]["ideal_obs_neighbors"],
                    "ideal_comm_neighbors": group[0]["ideal_comm_neighbors"],
                }
            )

    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "episodes.csv", rows)
    write_csv(args.out / "summary.csv", summary)
    packages = {}
    for name in ("numpy", "scipy", "mpe2", "pettingzoo"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    manifest = {
        "python": platform.python_version(),
        "packages": packages,
        "env": asdict(base),
        "agents": args.agents,
        "seeds": args.seeds,
        "policies": args.policies,
        "note": "slot_oracle requires global state; feasibility diagnostic only",
    }
    (args.out / "config.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for row in summary:
        print(
            f'N={row["n_agents"]} {row["policy"]:11} '
            f'success={row["success_rate"]:.0%} error={row["formation_err_mean"]:.4f} '
            f'collisions={row["collisions_mean"]:.1f} '
            f'ideal_comm={row["ideal_comm_neighbors"]:.1f}'
        )
    print(f"[DONE] {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
