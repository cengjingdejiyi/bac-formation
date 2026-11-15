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
