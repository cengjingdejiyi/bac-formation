"""统一训练入口。

用法：
    python -m bacmarl.runners.train env=formation algo=bac env.n_agents=6 algo.msg_bits=4 seed=0
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from bacmarl.algos.buffer import RolloutBuffer  # noqa: E402
from bacmarl.algos.dual import BetaDual  # noqa: E402
from bacmarl.algos.mappo import ppo_update  # noqa: E402
from bacmarl.comm.base import build_comm  # noqa: E402
from bacmarl.envs.formation_comm import EnvConfig  # noqa: E402
from bacmarl.envs.vec_env import VecFormationEnv  # noqa: E402
from bacmarl.runners.evaluate import _METRIC_KEYS, evaluate_policy  # noqa: E402
from bacmarl.runners.rollout import collect_rollout  # noqa: E402
from bacmarl.utils.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from bacmarl.utils.config import load_config  # noqa: E402
from bacmarl.utils.logger import RunLogger  # noqa: E402
from bacmarl.utils.naming import (
    run_name,
    variant_suffix as _variant_suffix,
)  # noqa: E402
from bacmarl.utils.normalizer import RunningStd  # noqa: E402
from bacmarl.utils.runtime import dump_yaml as _dump_yaml  # noqa: E402
from bacmarl.utils.runtime import eval_channel as _eval_channel  # noqa: E402
from bacmarl.utils.seeding import set_seed  # noqa: E402

_EVAL_KEYS = _METRIC_KEYS


def value_target_scale(recent_episodes) -> float:
    """价值目标缩放系数：回合回报标准差的倒数，把 O(100) 的回报压到 O(1)。

    只用于价值回归目标，策略梯度不受影响（优势在 ppo_update 内逐轮归一化）。
    不放大：std < 1 时返回 1.0；还没有回合数据时也返回 1.0。
    """
    if not recent_episodes:
        return 1.0
    rets = np.asarray([ep["ret"] for ep in recent_episodes], dtype=np.float64)
    std = float(rets.std())
    if not np.isfinite(std) or std < 1.0:
        return 1.0
    return 1.0 / std


def validate_norm_cfg(tr: dict) -> None:
    """reward_norm 与 value_norm 互斥：两者都在缩放价值目标尺度，同时开会缩两次。"""
    if tr.get("value_norm", False) and tr.get("reward_norm", True):
        raise ValueError(
            "train.value_norm 与 train.reward_norm 互斥：请显式设 train.reward_norm=false"
        )


def _restore_run(tr: dict, actor, critic, opt, dual, scaler) -> int:
    """按 `train.resume_from` 恢复权重、优化器、β 与奖励缩放统计，返回起始步数。

    β 是唯一的对偶状态（`meta["beta"]`），不恢复它会让续训后的预算约束断档；
    优化器与 Welford 统计同理，缺了它们只能算「换权重接着跑」，不是续训。
    回合级日志窗口（recent_episodes / episode_count）不恢复，只影响曲线开头的平滑。
    """
    resume = tr.get("resume_from")
    if not resume:
        return 0
    path = Path(str(resume))
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise ValueError(f"resume_from 指向的 checkpoint 不存在：{path}")
    ckpt = load_checkpoint(path, actor, critic, optimizer=opt)
    meta = ckpt.get("meta") or {}
    if "beta" in meta:
        dual.load_state_dict({"beta": meta["beta"]})
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    steps = int(meta.get("steps", 0))
    print(f"[INFO] resumed {path.name}: steps={steps} beta={dual.beta:.3f}")
    return steps


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cfg = load_config(argv)
    try:
        import torch
    except Exception:
        print(
            "[ERR] 训练需要 PyTorch：pip install -r requirements.txt", file=sys.stderr
        )
        return 1

    set_seed(int(cfg["seed"]))
    tr = cfg["train"]
    validate_norm_cfg(tr)
    env_cfg = EnvConfig.from_dict(cfg["env"])
    with VecFormationEnv(int(tr["n_envs"]), env_cfg, seed=int(cfg["seed"])) as vec:
        return _train(cfg, vec, torch)


def _train(cfg: dict, vec: VecFormationEnv, torch) -> int:
    """环境和循环状态只初始化一次；rollout 分段不改变回合边界。"""
    tr = cfg["train"]
    env_cfg = vec.cfg
    b, n = int(tr["n_envs"]), env_cfg.n_agents
    hidden = int(tr["hidden"])

    comm = build_comm(cfg["algo"]["comm"], hidden, cfg["algo"])
    # 只有二值消息（BAC）需要翻转掩码
    flip_bits = (
        int(cfg["algo"].get("msg_bits", 0)) if cfg["algo"]["comm"] == "bac" else 0
    )

    from bacmarl.models.actor import Actor
    from bacmarl.models.critic import Critic

    actor = Actor(vec.obs_dim, env_cfg.n_obs, 5, hidden, comm)
    critic = Critic(
        cfg["algo"].get("critic", "central"), vec.state_dim, vec.obs_dim, n, hidden
    )
    opt = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=float(tr["lr"])
    )

    dual = BetaDual(
        beta=float(cfg["algo"].get("beta_init", 0.0)),
        lr=float(cfg["algo"].get("beta_lr", 0.0)),
        b_max=float(cfg["algo"].get("budget_bits", 0.0)),
        beta_max=float(cfg["algo"].get("beta_max", 5.0)),
    )

    exp_name = (cfg.get("experiment") or {}).get("name", "main")
    run_dir = ROOT / "results" / exp_name / run_name(cfg)
    if (run_dir / "metrics.csv").exists() and not tr.get("resume_from"):
        print(
            f"[WARN] {run_dir} 已有 metrics.csv，本次会覆盖（换 experiment.name 可避开）"
        )
    logger = RunLogger(run_dir)
    _dump_yaml(cfg, run_dir / "config.yaml")

    # 训练内评测：信道取固定值（训练区间中点，可由 train.eval_drop_p/eval_flip_p 覆盖）。
    eval_every = int(tr.get("eval_every", 0) or 0)
    eval_episodes = int(tr.get("eval_episodes", 8))
    eval_cfg = (
        EnvConfig.from_dict({**cfg["env"], "channel": _eval_channel(cfg["env"], tr)})
        if eval_every > 0
        else None
    )
    # 奖励尺度归一化：只缩放训练用的 shaped reward，价值目标与自举同尺度。
    scaler = RunningStd() if tr.get("reward_norm", True) else None
    start_steps = _restore_run(tr, actor, critic, opt, dual, scaler)

    try:
        steps, total = start_steps, int(tr["total_env_steps"])
        rollout_len = int(tr["rollout_len"])
        if total < 1 or rollout_len < 1:
            raise ValueError("total_env_steps and rollout_len must be positive")
        if steps >= total:
            print(f"[WARN] resume 起点 {steps} >= total_env_steps={total}，本次不训练")
            return 0
        # 续训时不能把评测计时器从 0 重新数，否则会立刻多跑一次评测。
        next_eval = ((steps // eval_every) + 1) * eval_every if eval_every > 0 else 0
        print(
            f'[INFO] run_dir={run_dir}\n[INFO] comm={cfg["algo"]["comm"]} N={n} B={b} hidden={hidden} backend={vec.envs[0].backend}'
        )
        batch = vec.reset()
        h = actor.init_hidden(b, n)
        recent_episodes = []
        episode_count = 0
        while steps < total:
            length = min(rollout_len, (total - steps + b - 1) // b)
            buf = RolloutBuffer(
                length,
                b,
                n,
                vec.obs_dim,
                env_cfg.n_obs,
                vec.state_dim,
                hidden,
                flip_bits,
            )
            rollout = collect_rollout(vec, actor, critic, buf, batch, h)
            batch, h = rollout.batch, rollout.hidden
            steps += buf.ptr * b
            shaped_all = buf.reward_team - dual.beta * buf.bits
            if scaler is not None:
                scaler.update(shaped_all)
                shaped_all = shaped_all * scaler.scale()
            buf.compute_returns(
                shaped_all,
                rollout.last_value,
                float(tr["gamma"]),
                float(tr["gae_lambda"]),
            )
            # value_norm：只缩放价值回归目标（ret / v_old），策略梯度不受影响
            # （优势在 ppo_update 里逐轮归一化）。目的是把 O(100) 的回合回报压到 O(1)，
            # 否则 value loss 随量级平方放大（实测 76~400），后段一次尖峰就把策略带崩。
            value_scale = (
                value_target_scale(recent_episodes)
                if tr.get("value_norm", False)
                else 1.0
            )
            if value_scale != 1.0:
                buf.ret[: buf.ptr] *= value_scale
                buf.value[: buf.ptr] *= value_scale
            tb = {
                k: torch.as_tensor(v)
                for k, v in buf.batched(buf.flat_indices()).items()
            }
            logs = ppo_update(actor, critic, opt, buf, tr, tb)
            beta = dual.update(float(buf.bits.mean()))
            episode_count += len(rollout.episodes)
            recent_episodes = (recent_episodes + rollout.episodes)[-100:]

            def episode_mean(key: str) -> float:
                return (
                    float(np.mean([ep[key] for ep in recent_episodes]))
                    if recent_episodes
                    else float("nan")
                )

            if eval_every > 0 and steps >= next_eval and eval_cfg is not None:
                was_training = actor.training
                actor.eval()
                eval_metrics = evaluate_policy(
                    actor, eval_cfg, episodes=eval_episodes, seed=int(cfg["seed"])
                )
                actor.train(was_training)
                next_eval = steps + eval_every
                ckpt_path = run_dir / "ckpt" / f"step_{steps}.ckpt"
                save_checkpoint(
                    ckpt_path,
                    actor,
                    critic,
                    meta={"cfg": cfg, "steps": steps, "beta": dual.beta},
                    optimizer=opt,
                    extra={"scaler": scaler.state_dict()} if scaler else None,
                )
                print(f"[INFO] eval@{steps} {eval_metrics} -> {ckpt_path.name}")
            else:
                # 字段始终存在，CSV 表头才不会被首行定型后丢掉 eval 列。
                eval_metrics = {k: float("nan") for k in _EVAL_KEYS}

            logger.log(
                {
                    "train/return": episode_mean("ret"),
                    "train/success_rate": episode_mean("success"),
                    "train/formation_err": episode_mean("formation_err"),
                    "train/collisions": episode_mean("collisions"),
                    "train/episode_length": episode_mean("len"),
                    "train/episodes": episode_count,
                    "train/step_reward": float(buf.reward_team.mean()),
                    "train/reward_scale": float(scaler.scale()) if scaler else 1.0,
                    "train/value_scale": value_scale,
                    "comm/bits": float(buf.bits.mean()),
                    "comm/speak_rate": float(buf.gate.mean()),
                    "comm/beta": beta,
                    **{f"eval/{k}": v for k, v in eval_metrics.items()},
                    **logs,
                },
                steps,
            )
        save_checkpoint(
            run_dir / "ckpt" / "last.ckpt",
            actor,
            critic,
            meta={"cfg": cfg, "steps": steps, "beta": dual.beta},
            optimizer=opt,
            extra={"scaler": scaler.state_dict()} if scaler else None,
        )
        print(f"[DONE] steps={steps} beta={dual.beta:.3f} csv={logger.csv_path}")
    finally:
        logger.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
