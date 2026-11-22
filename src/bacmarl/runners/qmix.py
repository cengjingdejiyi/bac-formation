"""QMIX 训练与评测入口（对比基线，负责人 D）。

用法（与策略梯度入口同一套配置/命名/日志口径）：

    python -m bacmarl.runners.qmix env=formation algo=qmix seed=0
    python -m bacmarl.runners.qmix env=formation algo=qmix \\
        train.total_env_steps=3200 experiment.name=ci seed=0
    python -m bacmarl.runners.qmix env=formation algo=qmix seed=0 --robust

输出与 `runners/train.py` 完全同构：`results/{实验组}/{运行名}/` 下有 `config.yaml`、
`metrics.csv`、`tb/`、`ckpt/`；`--robust` 额外写 `robust.csv`（契约 5 列），供 E 的
`make_figures.py` 直接读取。

为什么单独一个入口而不是塞进 `train.py`：QMIX 是 off-policy 值分解，没有 rollout
缓存、GAE、PPO 截断目标和 β 对偶，硬塞进 PPO 主循环会让两条链路的语义互相污染。
两条入口共享的是配置、运行名、评测信道与日志字段，这些都已经抽到 utils 里。
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

try:  # 允许在没有 torch 的机器上只看配置/排班，不在这里就崩
    import torch
except Exception:  # pragma: no cover - 仅缺依赖时才走这里
    torch = None

from bacmarl.algos.qmix import QMixAgent, epsilon_at, qmix_td_loss  # noqa: E402
from bacmarl.algos.replay import NStepWindow, TransitionReplay  # noqa: E402
from bacmarl.envs.formation_comm import EnvConfig  # noqa: E402
from bacmarl.envs.vec_env import VecFormationEnv  # noqa: E402
from bacmarl.runners.evaluate import _METRIC_KEYS, evaluate_policy  # noqa: E402
from bacmarl.utils.config import load_config  # noqa: E402
from bacmarl.utils.logger import RunLogger  # noqa: E402
from bacmarl.utils.naming import run_name  # noqa: E402
from bacmarl.utils.runtime import dump_yaml, eval_channel  # noqa: E402
from bacmarl.utils.seeding import set_seed  # noqa: E402

_EVAL_KEYS = _METRIC_KEYS
N_ACTIONS = 5  # MPE2 离散动作：悬停/左/右/下/上
ROBUST_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)


class GreedyQPolicy:
    """把 QMIX 包成 `evaluate_policy` 认识的最小策略接口（执行期 argmax，无混合网络）。

    评测口径与策略梯度侧一致：每回合零隐状态、回合内不重置、动作确定性。
    QMIX 不通信，`bits`/`gate` 恒为 0，因此 `eval/bits`、`eval/speak_rate` 也是 0，
    与 `comm/bits=0` 对得上。
    """

    def __init__(self, agent: QMixAgent):
        self.agent = agent

    def init_hidden(self, b: int, n: int, device=None):
        return self.agent.init_hidden(b, n, device)

    def __call__(
        self,
        obs,
        obs_mask,
        comm_mask,
        h_prev,
        channel,
        gate=None,
        noise=None,
        deterministic: bool = True,
    ):
        q, h = self.agent.q_values(obs, obs_mask, h_prev)
        zeros = torch.zeros(obs.shape[0], obs.shape[1], device=obs.device)
        return SimpleNamespace(
            action_logits=q,
            gate_logits=None,
            gate=None,
            h=h,
            comm_c=zeros,
            bits=zeros,
            noise=None,
        )


def _episode_mean(episodes: list[dict], key: str) -> float:
    """回合字段均值；缺字段（如 toy 环境没给 formation_err_mean）返回 nan，不抛 KeyError。"""
    if not episodes:
        return float("nan")
    values = [ep.get(key) for ep in episodes if ep.get(key) is not None]
    if not values:
        return float("nan")
    return float(np.mean(values))


def _save_ckpt(
    path: Path,
    agent: QMixAgent,
    target: QMixAgent,
    opt,
    cfg: dict,
    steps: int,
    epsilon: float,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "agent": agent.state_dict(),
            "target": target.state_dict(),
            "optimizer": opt.state_dict(),
            "meta": {"cfg": cfg, "steps": int(steps), "epsilon": float(epsilon)},
        },
        path,
    )
    return path


def load_qmix_checkpoint(
    path: str | Path, agent: QMixAgent | None = None, optimizer=None
) -> dict:
    """读 QMIX checkpoint；给了 agent/optimizer 就原地恢复。"""
    ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
    if agent is not None and "agent" in ckpt:
        agent.load_state_dict(ckpt["agent"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt


def build_agent(cfg: dict, vec: VecFormationEnv) -> QMixAgent:
    """按配置构造 QMIX（供训练与离线评测共用，避免两处超参不一致）。"""
    algo, tr = cfg["algo"], cfg["train"]
    return QMixAgent(
        obs_dim=vec.obs_dim,
        n_obs=vec.cfg.n_obs,
        n_actions=N_ACTIONS,
        hidden=int(tr["hidden"]),
        n_agents=vec.cfg.n_agents,
        state_dim=vec.state_dim,
        mixer_embed=int(algo.get("mixer_embed", 32)),
        hypernet_embed=int(algo.get("hypernet_embed", 64)),
        use_agent_id=bool(algo.get("use_agent_id", True)),
    )


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    flags = [a for a in argv if a.startswith("--")]
    tokens = [a for a in argv if not a.startswith("--")]
    cfg = load_config(tokens)
    if torch is None:
        print(
            "[ERR] QMIX 训练需要 PyTorch：pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 1
    if (cfg.get("algo") or {}).get("comm") != "none":
        print(
            "[ERR] QMIX 只能配 algo.comm=none（QMIX 不使用通信模块）", file=sys.stderr
        )
        return 1
    set_seed(int(cfg["seed"]))
    tr = cfg["train"]
    env_cfg = EnvConfig.from_dict(cfg["env"])
    with VecFormationEnv(int(tr["n_envs"]), env_cfg, seed=int(cfg["seed"])) as vec:
        return _train(cfg, vec, robust="--robust" in flags)


def _train(cfg: dict, vec: VecFormationEnv, robust: bool = False) -> int:
    tr, algo = cfg["train"], cfg["algo"]
    b, n = int(tr["n_envs"]), vec.cfg.n_agents
    hidden = int(tr["hidden"])
    gamma = float(tr["gamma"])

    agent = build_agent(cfg, vec)
    target = agent.target_copy()
    opt = torch.optim.Adam(agent.parameters(), lr=float(algo.get("lr", 5e-4)))
    replay = TransitionReplay(
        capacity=int(algo.get("replay_size", 200000)),
        obs_dim=vec.obs_dim,
        n_obs=vec.cfg.n_obs,
        state_dim=vec.state_dim,
        hidden=hidden,
        n_agents=n,
    )

    batch_size = max(1, int(algo.get("batch_size", 64)))
    n_step = max(1, int(algo.get("n_step", 1)))
    train_start = max(1, int(algo.get("train_start", 2000)))
    train_every = max(1, int(algo.get("train_every", 4)))
    updates_per_train = max(1, int(algo.get("updates_per_train", 1)))
    target_update = max(1, int(algo.get("target_update", 500)))
    eps_start, eps_end = float(algo.get("eps_start", 1.0)), float(
        algo.get("eps_end", 0.05)
    )
    eps_anneal = int(algo.get("eps_anneal_steps", 100000))
    grad_clip = float(algo.get("grad_clip", 10.0))
    double_q = bool(algo.get("double_q", True))
    # 稳定化三件套（2026-11-21 加入，见 algos/qmix.py 的说明与 docs/baselines.md §5.1）：
    # Huber 损失把离群 TD 误差压成线性、target 夹逼阻断自举发散、梯度范数进日志。
    td_huber = bool(algo.get("td_huber", True))
    huber_beta = float(algo.get("huber_beta", 1.0))
    _clip = algo.get("target_clip", 50.0)
    target_clip = float(_clip) if _clip not in (None, "none", "None") else None
    if target_clip is not None and target_clip <= 0:
        target_clip = None
    # DQN 系常规做法：把每步奖励缩放到 O(1)，价值量级才不会到 -100（100 步回合、
    # 每步 -1 时）。只缩放学习用的回报，环境返回的回合回报照旧记账。
    reward_scale = float(algo.get("reward_scale", 1.0))

    exp_name = (cfg.get("experiment") or {}).get("name", "main")
    run_dir = ROOT / "results" / exp_name / run_name(cfg)
    if (run_dir / "metrics.csv").exists() and not tr.get("resume_from"):
        print(
            f"[WARN] {run_dir} 已有 metrics.csv，本次会覆盖（换 experiment.name 可避开）"
        )
    logger = RunLogger(run_dir)
    dump_yaml(cfg, run_dir / "config.yaml")

    eval_every = int(tr.get("eval_every", 0) or 0)
    eval_episodes = int(tr.get("eval_episodes", 8))
    eval_cfg = (
        EnvConfig.from_dict({**cfg["env"], "channel": eval_channel(cfg["env"], tr)})
        if eval_every > 0
        else None
    )
    log_every = max(b, int(tr.get("log_every", b * int(tr["rollout_len"]))))

    steps, start_steps = 0, 0
    resume = tr.get("resume_from")
    if resume:
        path = Path(str(resume))
        path = path if path.is_absolute() else ROOT / path
        ckpt = load_qmix_checkpoint(path, agent, opt)
        start_steps = int((ckpt.get("meta") or {}).get("steps", 0))
        steps = start_steps
        target.load_state_dict(ckpt.get("target") or ckpt["agent"])
        print(f"[INFO] resumed {path.name}: steps={steps}")

    total = int(tr["total_env_steps"])
    if steps >= total:
        print(f"[WARN] resume 起点 {steps} >= total_env_steps={total}，本次不训练")
        logger.close()
        return 0
    next_eval = ((steps // eval_every) + 1) * eval_every if eval_every > 0 else 0

    rng = np.random.default_rng(int(cfg["seed"]) + 7919)
    window = NStepWindow(n_step, gamma, b)
    batch = vec.reset()
    h_prev = agent.init_hidden(b, n)
    recent_episodes: list[dict] = []
    episode_count = 0
    ret_sum, ret_n = 0.0, 0
    updates = 0
    last_stats = None
    last_grad_norm = float("nan")
    window_rewards: list[float] = []
    last_log = 0

    print(
        f"[INFO] run_dir={run_dir}\n[INFO] comm=none(QMIX) N={n} B={b} hidden={hidden} "
        f"backend={vec.envs[0].backend}"
    )
    try:
        while steps < total:
            obs = torch.as_tensor(batch["obs"])
            obs_mask = torch.as_tensor(batch["obs_mask"])
            with torch.no_grad():
                q, h_out = agent.q_values(obs, obs_mask, h_prev)
            epsilon = epsilon_at(steps, eps_start, eps_end, eps_anneal)
            greedy = q.argmax(-1)
            explore = torch.as_tensor(rng.random((b, n)) < epsilon)
            random_actions = torch.as_tensor(
                rng.integers(0, N_ACTIONS, size=(b, n)), dtype=torch.int64
            )
            action = torch.where(explore, random_actions, greedy)
            nxt = vec.step(action.numpy())

            reward = np.asarray(nxt["reward"], dtype=np.float32).mean(axis=1)  # [B]
            done = np.asarray(nxt["done"], dtype=np.float32)  # [B]
            # n 步回报：逐环境折叠，凑满 n 步或遇到终止就写回 replay。
            h_np = h_prev.numpy()
            for k in range(b):
                window.set_tail(
                    k,
                    nxt["obs"][k],
                    nxt["obs_mask"][k],
                    h_out[k].numpy(),
                    nxt["state"][k],
                )
                for rec in window.push(
                    k,
                    {
                        "obs": batch["obs"][k],
                        "obs_mask": batch["obs_mask"][k],
                        "action": action[k].numpy(),
                        "reward": float(reward[k]) * reward_scale,
                        "done": float(done[k]),
                        "state": batch["state"][k],
                        "h_prev": h_np[k],
                    },
                ):
                    replay.push(**rec)
            window_rewards.extend(float(r) for r in reward)
            h_prev = h_out.masked_fill(torch.as_tensor(nxt["done"])[:, None, None], 0.0)
            batch = nxt
            steps += b
            new_eps = [ep for ep in nxt.get("ep", []) if ep is not None]
            episode_count += len(new_eps)
            recent_episodes = (recent_episodes + new_eps)[-100:]
            ret_sum += float(sum(ep["ret"] for ep in new_eps))
            ret_n += len(new_eps)

            if len(replay) >= train_start and steps % train_every == 0:
                for _ in range(updates_per_train):
                    sample = replay.sample(batch_size, rng)
                    loss, last_stats = qmix_td_loss(
                        agent,
                        target,
                        sample,
                        gamma,
                        double_q=double_q,
                        huber=td_huber,
                        huber_beta=huber_beta,
                        target_clip=target_clip,
                    )
                    if not torch.isfinite(loss):
                        # 发散样本直接跳过，不清梯度也不更新参数。
                        opt.zero_grad(set_to_none=True)
                        continue
                    opt.zero_grad()
                    loss.backward()
                    last_grad_norm = float(
                        torch.nn.utils.clip_grad_norm_(agent.parameters(), grad_clip)
                    )
                    opt.step()
                    updates += 1
                    if updates % target_update == 0:
                        agent.sync_target(target)

            # 与 PPO 入口一样按固定步距落一行；最后一行强制写，保证曲线终点正好是
            # total_env_steps（18 个 run 的横轴才对得齐）。
            if steps - last_log >= log_every or steps >= total:
                eval_metrics = {k: float("nan") for k in _EVAL_KEYS}
                if eval_every > 0 and steps >= next_eval and eval_cfg is not None:
                    was_training = agent.training
                    agent.eval()
                    eval_metrics = evaluate_policy(
                        GreedyQPolicy(agent),
                        eval_cfg,
                        episodes=eval_episodes,
                        seed=int(cfg["seed"]),
                    )
                    agent.train(was_training)
                    next_eval = steps + eval_every
                    _save_ckpt(
                        run_dir / "ckpt" / f"step_{steps}.ckpt",
                        agent,
                        target,
                        opt,
                        cfg,
                        steps,
                        epsilon,
                    )
                row = {
                    "train/return": _episode_mean(recent_episodes, "ret"),
                    "train/return_all": (ret_sum / ret_n) if ret_n else float("nan"),
                    "train/success_rate": _episode_mean(recent_episodes, "success"),
                    "train/formation_err": _episode_mean(
                        recent_episodes, "formation_err"
                    ),
                    "train/formation_err_mean": _episode_mean(
                        recent_episodes, "formation_err_mean"
                    ),
                    "train/collisions": _episode_mean(recent_episodes, "collisions"),
                    "train/episode_length": _episode_mean(recent_episodes, "len"),
                    "train/episodes": episode_count,
                    "train/step_reward": (
                        float(np.mean(window_rewards))
                        if window_rewards
                        else float("nan")
                    ),
                    # 列与策略梯度方法保持同构：QMIX 不通信，这些恒为 0。
                    "comm/bits": 0.0,
                    "comm/speak_rate": 0.0,
                    "comm/beta": 0.0,
                    "qmix/epsilon": float(epsilon),
                    "qmix/updates": updates,
                    "qmix/replay": len(replay),
                    "loss/td": last_stats.loss if last_stats else float("nan"),
                    "qmix/td_abs": last_stats.td_abs if last_stats else float("nan"),
                    "qmix/q_tot": last_stats.q_tot if last_stats else float("nan"),
                    "qmix/target": last_stats.target if last_stats else float("nan"),
                    "qmix/target_clip": last_stats.clip_frac if last_stats else float("nan"),
                    "qmix/grad_norm": last_grad_norm,
                    **{f"eval/{k}": v for k, v in eval_metrics.items()},
                }
                logger.log(row, steps)
                window_rewards = []
                last_log = steps

        _save_ckpt(
            run_dir / "ckpt" / "last.ckpt",
            agent,
            target,
            opt,
            cfg,
            steps,
            epsilon_at(steps, eps_start, eps_end, eps_anneal),
        )
        print(f"[DONE] steps={steps} updates={updates} csv={logger.csv_path}")

        if robust:
            _write_robust(agent, cfg, run_dir, episodes=eval_episodes)
    finally:
        logger.close()
    return 0


def _write_robust(
    agent: QMixAgent, cfg: dict, run_dir: Path, episodes: int = 8
) -> Path:
    """按丢包率网格复测并写 `robust.csv`（契约 5 列，供 E 的作图脚本直接用）。"""
    grid = ((cfg.get("experiment") or {}).get("robustness") or {}).get(
        "drop_p"
    ) or list(ROBUST_GRID)
    policy = GreedyQPolicy(agent)
    rows = []
    for drop_p in grid:
        ec = EnvConfig.from_dict(cfg.get("env", {}))
        ec.channel = {"drop_p": float(drop_p), "flip_p": 0.0}
        m = evaluate_policy(policy, ec, episodes=episodes, seed=int(cfg["seed"]))
        rows.append(
            {
                "drop_p": float(drop_p),
                "success_rate": m["success_rate"],
                "formation_err": m["formation_err"],
                "speak_rate": 0.0,  # QMIX 无门控
                "bits": 0.0,  # QMIX 不通信
            }
        )
    out = run_dir / "robust.csv"
    import csv

    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[DONE] {out}")
    for row in rows:
        print(row)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
