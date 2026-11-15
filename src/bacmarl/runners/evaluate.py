"""评测：确定性回合评测（供 train 定期调用）与按丢包率网格复测。

评测入口（负责人 E）与训练内评测（负责人 C）共用 `evaluate_policy`，避免两套策略语义。

用法：
    python -m bacmarl.runners.evaluate --run results/main/formation_6n_bac_L4_K3_B2.0_s0 --episodes 8
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from bacmarl.envs.formation_comm import EnvConfig, FormationCommEnv  # noqa: E402

_METRIC_KEYS = (
    "return",
    "success_rate",
    "formation_err",
    "collisions",
    "length",
    "bits",
    "speak_rate",
)


def evaluate_policy(
    actor, env_cfg: EnvConfig, episodes: int = 8, seed: int = 0
) -> dict:
    """用确定性策略跑固定回合，返回各指标均值。

    动作取 argmax；门控取 p>=0.5（`Actor.forward(deterministic=True)`）；
    每回合隐状态从零开始，回合内不重置。actor 为 None 时退化为均匀随机动作。
    """
    import torch

    n = env_cfg.n_agents
    acc = {k: [] for k in _METRIC_KEYS}
    for ep in range(episodes):
        env = FormationCommEnv(env_cfg)
        try:
            step = env.reset(seed=seed + ep)
            torch.manual_seed(seed + ep)
            hidden = (
                actor.init_hidden(1, n) if actor is not None else torch.zeros(1, n, 1)
            )
            rng = np.random.default_rng(seed + ep)
            total, length, bits, gates = 0.0, 0, [], []
            while True:
                if actor is not None:
                    with torch.no_grad():
                        out = actor(
                            torch.as_tensor(step.obs)[None],
                            torch.as_tensor(step.obs_mask)[None],
                            torch.as_tensor(step.comm_mask)[None],
                            hidden,
                            dict(step.info["channel"]),
                            deterministic=True,
                        )
                    hidden = out.h
                    action = out.action_logits.argmax(-1)[0].numpy()
                    bits.append(float(out.bits.mean()))
                    if out.gate is not None:
                        gates.append(float(out.gate.mean()))
                else:
                    action = rng.integers(0, 5, size=n)
                nxt = env.step(action)
                total += float(np.mean(nxt.reward))
                length += 1
                done, step = nxt.done, nxt
                if done:
                    break
            acc["return"].append(total)
            acc["length"].append(length)
            acc["success_rate"].append(float(step.info["success"]))
            acc["formation_err"].append(float(step.info["formation_err"]))
            acc["collisions"].append(float(step.info["collisions_ep"]))
            acc["bits"].append(float(np.mean(bits)) if bits else 0.0)
            acc["speak_rate"].append(float(np.mean(gates)) if gates else 0.0)
        finally:
            env.close()
    return {k: float(np.mean(v)) if v else 0.0 for k, v in acc.items()}


def _read_cfg(run_dir: Path) -> dict:
    p = run_dir / "config.yaml"
    if not p.exists():
        from bacmarl.utils.config import load_config

        return load_config([])
    try:
        import yaml

        return yaml.safe_load(p.read_text(encoding="utf-8"))
    except Exception:
        import json

        return json.loads(p.read_text(encoding="utf-8"))


def _load_actor(run_dir: Path, cfg: dict, ckpt: str):
    try:
        from bacmarl.comm.base import build_comm
        from bacmarl.models.actor import Actor
        from bacmarl.utils.checkpoint import load_checkpoint

        hidden = int((cfg.get("train") or {}).get("hidden", 64))
        env_cfg = EnvConfig.from_dict(cfg.get("env", {}))
        comm = build_comm(cfg["algo"]["comm"], hidden, cfg["algo"])
        prot = FormationCommEnv(env_cfg)
        try:
            actor = Actor(prot.obs_dim, env_cfg.n_obs, 5, hidden, comm)
        finally:
            prot.close()
        path = run_dir / ckpt
        if not path.exists():
            print(f"[WARN] 未找到 {path}，退化为随机策略")
            return None
        load_checkpoint(path, actor=actor)
        actor.eval()
        return actor
    except Exception as exc:  # 缺 torch / ckpt 损坏时仍要能出对照行
        print(f"[WARN] 加载模型失败（{exc}），退化为随机策略")
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--ckpt", default="ckpt/last.ckpt")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    cfg = _read_cfg(run_dir)
    grid = (
        ((cfg.get("experiment") or {}).get("robustness") or {}).get("drop_p") or [0.0]
    )

    actor = _load_actor(run_dir, cfg, args.ckpt)
    rows = []
    for drop_p in grid:
        ec = EnvConfig.from_dict(cfg.get("env", {}))
        ec.channel = {"drop_p": float(drop_p), "flip_p": 0.0}
        m = evaluate_policy(actor, ec, episodes=args.episodes)
        rows.append({"drop_p": drop_p, **{k: m[k] for k in _METRIC_KEYS}})

    out_csv = run_dir / "robust.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[DONE] {out_csv}")
    for r in rows:
        print(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
