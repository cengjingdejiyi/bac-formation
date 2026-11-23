"""按"回合内达标"口径重测成功率，并做容差敏感性扫描（不改权重、不重训）。

为什么需要这个脚本
------------------
仓库里的 `evaluate_policy` 只在回合**最后一步**读取 `info["success"]`，而环境
`end_on_success=False`：一个回合即使在中途已经"六机全部入槽并保持 10 步"，只要最后
一步漂出容差，就会被记成 0。契约原文是"六机全部进入容差并保持 10 步"，并未要求
"第 100 步仍在容差内"，所以旧口径偏严，导致六个方法成功率全 0。

本脚本按契约本意统计：**回合内任意时刻满足判据即算该回合成功**，同时给出：
- `success_any`：回合内是否出现过达标（>=hold 步）
- `first_success_step`：首次达标步号（未达标记 -1）
- `min_ferr` / `final_ferr`：回合内最小与末步队形误差
并做容差扫描（默认 0.09 / 0.15 / 0.25 / 0.35），说明"放宽到与环境尺度相称的容差后
成功率如何变化"。

QMIX 有独立的价值网络策略（GreedyQPolicy），不能用 PPO 的 Actor 加载，脚本内按
`algo.name == "qmix"` 分流。

用法：
    python scripts/eval_success_any.py --results results/main \
        --tolerances 0.09,0.15,0.25,0.35 --episodes 16 \
        --out results/_logs/success_any_sweep.csv
"""

from __future__ import annotations

import argparse
import copy
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from bacmarl.envs.formation_comm import EnvConfig, FormationCommEnv  # noqa: E402
from bacmarl.runners.evaluate import _load_actor, _read_cfg  # noqa: E402

# 多个进程并行跑时限制单进程线程数，避免线程超订把每个进程都拖慢
torch.set_num_threads(1)

FIELDS = (
    "method",
    "seed",
    "tolerance",
    "episodes",
    "success_any",
    "success_safe",
    "first_success_step_mean",
    "min_ferr",
    "final_ferr",
    "collisions",
    "return",
)

# "安全成功"门槛：达标且回合碰撞不超过 2 对（奖励本身就是 -(ferr + coll)，只看精度
# 会把"挤成一团但撞个不停"的策略算成成功，与任务目标不符）
SAFE_COLL = 2.0


def parse_list(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def method_of(name: str) -> str:
    parts = name.split("_")
    return parts[2] if len(parts) > 2 else name


def seed_of(name: str) -> str:
    return name.rsplit("_", 1)[-1].lstrip("s") or "0"


def build_policy(run_dir: Path, cfg: dict, ckpt: str):
    """按算法分流构造可评测的策略对象。"""
    algo = cfg.get("algo") or {}
    if algo.get("name") == "qmix":
        from bacmarl.runners.qmix import GreedyQPolicy, QMixAgent, load_qmix_checkpoint

        env_cfg = EnvConfig.from_dict(cfg.get("env", {}))
        prot = FormationCommEnv(env_cfg)
        try:
            agent = QMixAgent(
                obs_dim=prot.obs_dim,
                n_obs=env_cfg.n_obs,
                n_actions=5,
                hidden=int((cfg.get("train") or {}).get("hidden", 64)),
                n_agents=env_cfg.n_agents,
                state_dim=prot.state_dim,
                mixer_embed=int(algo.get("mixer_embed", 32)),
                hypernet_embed=int(algo.get("hypernet_embed", 64)),
                use_agent_id=bool(algo.get("use_agent_id", True)),
            )
        finally:
            prot.close()
        load_qmix_checkpoint(run_dir / ckpt, agent, None)
        agent.eval()
        return GreedyQPolicy(agent)
    return _load_actor(run_dir, cfg, ckpt)


def rollout_episode(policy, env_cfg: EnvConfig, seed: int) -> dict:
    """跑一个回合，按"回合内达标"口径记录成功与误差。"""
    env = FormationCommEnv(env_cfg)
    n = env_cfg.n_agents
    try:
        step = env.reset(seed=seed)
        torch.manual_seed(seed)
        hidden = policy.init_hidden(1, n)
        rng = np.random.default_rng(seed)
        total, length = 0.0, 0
        ferrs, first = [], -1
        success_any = False
        bits = []
        while True:
            with torch.no_grad():
                out = policy(
                    torch.as_tensor(step.obs)[None],
                    torch.as_tensor(step.obs_mask)[None],
                    torch.as_tensor(step.comm_mask)[None],
                    hidden,
                    dict(step.info["channel"]),
                    deterministic=True,
                )
            hidden = out.h
            action = out.action_logits.argmax(-1)[0].numpy()
            bits.append(float(out.bits.mean()) if out.bits is not None else 0.0)
            nxt = env.step(action)
            total += float(np.mean(nxt.reward))
            length += 1
            ferrs.append(float(nxt.info["formation_err"]))
            if bool(nxt.info["success"]) and not success_any:
                success_any = True
                first = length
            done, step = nxt.done, nxt
            if done:
                break
        return {
            "success_any": 1.0 if success_any else 0.0,
            "first_success_step": first,
            "min_ferr": float(np.min(ferrs)) if ferrs else float("nan"),
            "final_ferr": float(ferrs[-1]) if ferrs else float("nan"),
            "collisions": float(nxt.info["collisions_ep"]),
            "return": total,
            "bits": float(np.mean(bits)) if bits else 0.0,
        }
    finally:
        env.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/main")
    ap.add_argument("--tolerances", default="0.09,0.15,0.25,0.35")
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--hold", type=int, default=None, help="覆盖成功保持步数（默认用 run 配置里的值）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default="ckpt/last.ckpt")
    ap.add_argument("--out", default="results/_logs/success_any_sweep.csv")
    ap.add_argument(
        "--filter",
        default=None,
        help="只跑名字包含该子串的 run（如 _bac_），用于按方法拆分并行",
    )
    args = ap.parse_args(argv)

    root = Path(args.results) if Path(args.results).is_absolute() else ROOT / args.results
    runs = sorted(p for p in root.iterdir() if p.is_dir())
    if args.filter:
        runs = [p for p in runs if args.filter in p.name]
    if not runs:
        print(f"[FAIL] {root} 下没有 run 目录")
        return 2

    rows: list[dict] = []
    for run_dir in runs:
        cfg = _read_cfg(run_dir)
        try:
            policy = build_policy(run_dir, cfg, args.ckpt)
        except Exception as exc:
            print(f"[FAIL] {run_dir.name}: {exc}")
            return 4
        if policy is None:
            print(f"[FAIL] {run_dir.name}: 缺 {args.ckpt}")
            return 3
        method, seed = method_of(run_dir.name), seed_of(run_dir.name)
        for tol in parse_list(args.tolerances):
            env_dict = copy.deepcopy(cfg.get("env", {}))
            env_dict["success_dist"] = tol
            if args.hold:
                env_dict["success_hold"] = args.hold
            env_dict["channel"] = {"drop_p": 0.0, "flip_p": 0.0}
            env_cfg = EnvConfig.from_dict(env_dict)
            eps = [rollout_episode(policy, env_cfg, args.seed + i) for i in range(args.episodes)]
            succ = [e["success_any"] for e in eps]
            safe = [1.0 if (e["success_any"] > 0 and e["collisions"] <= SAFE_COLL) else 0.0 for e in eps]
            firsts = [e["first_success_step"] for e in eps if e["first_success_step"] > 0]
            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "tolerance": f"{tol:.3f}",
                    "episodes": args.episodes,
                    "success_any": f"{np.mean(succ):.4f}",
                    "success_safe": f"{np.mean(safe):.4f}",
                    "first_success_step_mean": f"{np.mean(firsts):.1f}" if firsts else "-1",
                    "min_ferr": f"{np.mean([e['min_ferr'] for e in eps]):.4f}",
                    "final_ferr": f"{np.mean([e['final_ferr'] for e in eps]):.4f}",
                    "collisions": f"{np.mean([e['collisions'] for e in eps]):.3f}",
                    "return": f"{np.mean([e['return'] for e in eps]):.3f}",
                }
            )
            print(
                f"[{method} s{seed}] tol={tol:.2f} success={np.mean(succ):.1%} "
                f"safe={np.mean(safe):.1%} "
                f"min_ferr={np.mean([e['min_ferr'] for e in eps]):.3f} "
                f"final_ferr={np.mean([e['final_ferr'] for e in eps]):.3f}",
                flush=True,
            )

    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"[DONE] {out} ({len(rows)} rows)")

    print("\n== 成功率（跨 3 seed 均值） ==")
    tols = parse_list(args.tolerances)
    print("method   " + "".join(f"{t:>9}" for t in tols))
    for method in sorted({r["method"] for r in rows}):
        line = f"{method:<9}"
        for tol in tols:
            vals = [
                float(r["success_any"])
                for r in rows
                if r["method"] == method and r["tolerance"] == f"{tol:.3f}"
            ]
            line += f"{np.mean(vals):>9.1%}" if vals else f"{'-':>9}"
        print(line)

    print(f"\n== 安全成功率（达标且碰撞<={SAFE_COLL:.0f} 对/回合） ==")
    print("method   " + "".join(f"{t:>9}" for t in tols))
    for method in sorted({r["method"] for r in rows}):
        line = f"{method:<9}"
        for tol in tols:
            vals = [
                float(r["success_safe"])
                for r in rows
                if r["method"] == method and r["tolerance"] == f"{tol:.3f}"
            ]
            line += f"{np.mean(vals):>9.1%}" if vals else f"{'-':>9}"
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
