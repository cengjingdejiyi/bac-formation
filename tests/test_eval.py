"""训练内评测（P1）验收：确定性评测、固定信道选择、train 真的写出 eval 指标与周期 ckpt。"""

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

torch = pytest.importorskip("torch")

from bacmarl.comm.base import build_comm  # noqa: E402
from bacmarl.envs.formation_comm import EnvConfig, FormationCommEnv  # noqa: E402
from bacmarl.models.actor import Actor  # noqa: E402
from bacmarl.runners import train as train_mod  # noqa: E402
from bacmarl.runners.evaluate import _METRIC_KEYS, evaluate_policy  # noqa: E402


def _actor(env: FormationCommEnv, seed: int = 0) -> Actor:
    torch.manual_seed(seed)
    comm = build_comm("bac", 8, {"msg_bits": 2, "msg_dim": 4, "recv_k": 2})
    return Actor(env.obs_dim, env.cfg.n_obs, 5, 8, comm)


def test_evaluate_policy_is_deterministic_and_reports_every_metric():
    cfg = EnvConfig(backend="mock", n_agents=3, max_steps=6, channel={"drop_p": 0.1})
    with FormationCommEnv(cfg) as env:
        actor = _actor(env)
    first = evaluate_policy(actor, cfg, episodes=2, seed=3)
    second = evaluate_policy(actor, cfg, episodes=2, seed=3)

    assert set(first) == set(_METRIC_KEYS)
    for key in _METRIC_KEYS:
        assert np.isfinite(first[key]) or np.isnan(first[key])
        assert first[key] == pytest.approx(second[key]), key
    assert first["length"] > 0


def test_eval_channel_uses_training_midpoint_and_honours_override():
    env_dict = {"channel": {"drop_p": [0.0, 0.4], "flip_p": [0.0, 0.1]}}
    assert train_mod._eval_channel(env_dict, {}) == {
        "drop_p": pytest.approx(0.2),
        "flip_p": pytest.approx(0.05),
    }
    overridden = train_mod._eval_channel(
        env_dict, {"eval_drop_p": 0.35, "eval_flip_p": 0.0}
    )
    assert overridden == {"drop_p": 0.35, "flip_p": 0.0}


def test_train_writes_eval_metrics_and_periodic_checkpoint(tmp_path, monkeypatch):
    """eval_every 生效：metrics.csv 有 eval/* 列，且落盘 step_*.ckpt。"""
    monkeypatch.setattr(train_mod, "ROOT", tmp_path)
    argv = [
        "env=formation",
        "algo=bac",
        "env.backend=mock",
        "env.n_agents=2",
        "env.max_steps=8",
        "train.n_envs=2",
        "train.rollout_len=8",
        "train.total_env_steps=32",
        "train.minibatch=8",
        "train.hidden=8",
        "train.eval_every=16",
        "train.eval_episodes=1",
        "experiment.name=pytest_eval",
        "seed=0",
    ]
    assert train_mod.main(argv) == 0

    csv_path = next((tmp_path / "results").rglob("metrics.csv"))
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    assert rows, "metrics.csv 为空"
    for key in ("eval/return", "eval/success_rate", "eval/formation_err"):
        assert key in rows[0], f"缺少列 {key}"
    assert any(
        row["eval/return"] not in ("", "nan") for row in rows
    ), "eval 指标没有真正写入"

    ckpts = sorted((csv_path.parent / "ckpt").glob("step_*.ckpt"))
    assert len(ckpts) >= 2, [p.name for p in ckpts]

    # 新增指标：全程累计回报与回合内 ferr 均值必须落列，且与旧口径分开
    for key in ("train/return_all", "train/formation_err_mean", "train/value_scale"):
        assert key in rows[0], f"缺少列 {key}"
    assert all(row["train/return_all"] not in ("", "nan") for row in rows)

    # 续训窗口：ckpt 里必须带 episodes 段，续训才能接上曲线
    import torch

    ckpt = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
    assert "episodes" in ckpt and "recent" in ckpt["episodes"]
    assert int(ckpt["episodes"]["count"]) >= 1
    assert int(ckpt["episodes"]["ret_n"]) >= 1


def test_gate_mode_sample_is_supported_and_invalid_mode_rejected():
    cfg = EnvConfig(backend="mock", n_agents=3, max_steps=6, channel={"drop_p": 0.0})
    with FormationCommEnv(cfg) as env:
        actor = _actor(env)
    sampled = evaluate_policy(actor, cfg, episodes=2, seed=5, gate_mode="sample")
    assert set(sampled) == set(_METRIC_KEYS)
    for key in _METRIC_KEYS:
        assert np.isfinite(sampled[key]) or np.isnan(sampled[key])
    with pytest.raises(ValueError, match="threshold or sample"):
        evaluate_policy(actor, cfg, episodes=1, seed=0, gate_mode="bogus")


def _no_comm_actor(env: FormationCommEnv, seed: int = 0) -> Actor:
    """无通信方法（IPPO/MAPPO）：comm 模块没有门控头，gate 是占位 ones。"""
    torch.manual_seed(seed)
    comm = build_comm("none", 8, {"msg_bits": 0, "msg_dim": 0, "recv_k": 0})
    return Actor(env.obs_dim, env.cfg.n_obs, 5, 8, comm)


def test_no_comm_policy_reports_zero_speak_rate():
    """无通信方法必须报 speak_rate=0；报成 1.0 会和 `bits=0` 自相矛盾。

    2026-11-22 修正：Actor 在 comm=none 时 gate 恒为 ones，评测侧原来按 `gate is not None`
    统计，于是 IPPO/MAPPO 的 `speak_rate` 全是 1.0（C 交付的 MAPPO 三 seed 就是这样）。
    """
    cfg = EnvConfig(backend="mock", n_agents=3, max_steps=6, channel={"drop_p": 0.1})
    with FormationCommEnv(cfg) as env:
        actor = _no_comm_actor(env)
    metrics = evaluate_policy(actor, cfg, episodes=2, seed=0)
    assert metrics["bits"] == pytest.approx(0.0)
    assert metrics["speak_rate"] == pytest.approx(0.0)


def test_tarmac_policy_reports_full_speak_rate():
    """反向用例：没有门控头但恒发（tarmac 1024 bit）时发言率必须是 1.0。

    只按 "gate_logits is None → 0" 修会把 tarmac 变成 `bits=1024 但 speak_rate=0`
    的新矛盾，所以口径要按"到底发没发消息"判。
    """
    cfg = EnvConfig(backend="mock", n_agents=3, max_steps=6, channel={"drop_p": 0.1})
    with FormationCommEnv(cfg) as env:
        torch.manual_seed(0)
        comm = build_comm("tarmac", 8, {"msg_dim": 32, "msg_bits": 4, "recv_k": 3})
        actor = Actor(env.obs_dim, env.cfg.n_obs, 5, 8, comm)
    metrics = evaluate_policy(actor, cfg, episodes=2, seed=0)
    assert metrics["bits"] == pytest.approx(1024.0, rel=0.02)
    assert metrics["speak_rate"] == pytest.approx(1.0)


@pytest.mark.parametrize("algo,expect", [("tarmac", 1.0), ("ippo", 0.0)])
def test_train_speak_rate_follows_comm_kind(tmp_path, monkeypatch, algo, expect):
    """训练日志里的 speak_rate 必须按"到底发没发消息"落数：

    tarmac（无门控、恒发 1024 bit）→ 1.0；ippo（comm=none）→ 0.0。
    旧逻辑取占位 gate 均值，两个都会写成 1.0，是 E 的 ippo 表里那处矛盾的来源。
    """
    monkeypatch.setattr(train_mod, "ROOT", tmp_path / algo)
    argv = [
        "env=formation",
        f"algo={algo}",
        "env.backend=mock",
        "env.n_agents=2",
        "env.max_steps=8",
        "train.n_envs=2",
        "train.rollout_len=8",
        "train.total_env_steps=32",
        "train.minibatch=8",
        "train.hidden=8",
        "train.eval_every=16",
        "train.eval_episodes=1",
        f"experiment.name=pytest_speak_{algo}",
        "seed=0",
    ]
    assert train_mod.main(argv) == 0
    csv_path = next((tmp_path / algo / "results").rglob("metrics.csv"))
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    speak = [float(r["comm/speak_rate"]) for r in rows if r["comm/speak_rate"] not in ("", "nan")]
    assert speak, "comm/speak_rate 没有落值"
    assert all(v == pytest.approx(expect) for v in speak), (algo, speak)
    bits = [float(r["comm/bits"]) for r in rows if r["comm/bits"] not in ("", "nan")]
    if algo == "tarmac":
        assert all(v > 0 for v in bits), bits
    else:
        assert all(v == pytest.approx(0.0) for v in bits), bits

