"""续训：resume_from 必须把 β、优化器、奖励缩放统计和步数一起接回来（负责人 A）。

不含环境的纯单元测试 + 一次真实入口的端到端续训。
"""

import sys
from pathlib import Path

import pytest
import torch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:  # 允许单文件运行 pytest
    sys.path.insert(0, str(SRC))

from bacmarl.algos.dual import BetaDual  # noqa: E402
from bacmarl.runners import train as train_mod  # noqa: E402
from bacmarl.utils.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from bacmarl.utils.normalizer import RunningStd  # noqa: E402


class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))


def _opt(module):
    return torch.optim.Adam(module.parameters(), lr=0.1)


def _ckpt(
    path: Path,
    steps: int,
    beta: float,
    scaler: dict | None = None,
    episodes: dict | None = None,
):
    actor, critic = _Tiny(), _Tiny()
    opt = _opt(actor)
    with torch.no_grad():
        actor.w.fill_(3.0)
    actor.w.sum().backward()  # 走一步，Adam 才有动量可存
    opt.step()
    saved = actor.w.detach().clone()
    extra = {}
    if scaler:
        extra["scaler"] = scaler
    if episodes:
        extra["episodes"] = episodes
    save_checkpoint(
        path,
        actor,
        critic,
        meta={"steps": steps, "beta": beta},
        optimizer=opt,
        extra=extra or None,
    )
    return path, saved


def test_restore_brings_back_beta_optimizer_scaler_and_steps(tmp_path):
    path, saved_w = _ckpt(
        tmp_path / "s8.ckpt",
        8,
        0.37,
        {"n": 5, "mean": 1.5, "m2": 2.0, "eps": 1e-8},
        {"recent": [{"ret": -1.0, "len": 4}], "count": 3, "ret_sum": -7.0, "ret_n": 3},
    )
    actor, critic = _Tiny(), _Tiny()
    opt = _opt(actor)
    dual = BetaDual(beta=0.0, lr=0.05, b_max=2.0)
    scaler = RunningStd()
    state = train_mod._restore_run(
        {"resume_from": str(path), "reward_norm": True},
        actor,
        critic,
        opt,
        dual,
        scaler,
    )
    assert state["steps"] == 8
    assert state["count"] == 3
    assert state["ret_sum"] == pytest.approx(-7.0) and state["ret_n"] == 3
    assert len(state["recent"]) == 1
    assert dual.beta == pytest.approx(0.37)
    assert torch.allclose(actor.w, saved_w)  # 权重已载入
    assert opt.state_dict()["state"]  # 优化器动量不是空的
    assert scaler.state_dict()["n"] == 5
    assert scaler.state_dict()["m2"] == pytest.approx(2.0)


def test_restore_is_a_noop_without_resume_from():
    actor, critic = _Tiny(), _Tiny()
    opt = _opt(actor)
    dual = BetaDual(beta=0.1, lr=0.05, b_max=2.0)
    state = train_mod._restore_run({}, actor, critic, opt, dual, RunningStd())
    assert state == {"steps": 0, "recent": [], "count": 0, "ret_sum": 0.0, "ret_n": 0}
    assert dual.beta == pytest.approx(0.1)


def test_restore_tolerates_old_ckpt_without_episode_window(tmp_path):
    """旧 ckpt 没有 episodes 段：不报错，窗口与累计值从空/0 开始。"""
    path, _ = _ckpt(tmp_path / "old.ckpt", 6, 0.2)
    state = train_mod._restore_run(
        {"resume_from": str(path)}, _Tiny(), _Tiny(), None, BetaDual(), None
    )
    assert state["steps"] == 6 and state["recent"] == [] and state["ret_n"] == 0


def test_restore_rejects_a_missing_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="不存在"):
        train_mod._restore_run(
            {"resume_from": str(tmp_path / "nope.ckpt")},
            _Tiny(),
            _Tiny(),
            None,
            BetaDual(),
            None,
        )


def _argv(experiment: str, total_steps: int, resume: Path | None = None):
    argv = [
        "env=formation",
        "algo=bac",
        "env.backend=mock",
        "env.n_agents=3",
        "train.n_envs=1",
        "train.rollout_len=4",
        f"train.total_env_steps={total_steps}",
        "train.ppo_epochs=1",
        "train.minibatch=4",
        "train.hidden=8",
        "env.max_steps=6",
        f"experiment.name={experiment}",
        "seed=0",
    ]
    if resume is not None:
        argv.append(f"train.resume_from={resume}")
    return argv


def test_training_entry_continues_from_the_last_checkpoint(capsys):
    first = train_mod.main(_argv("ci_resume_a", 8))
    assert first == 0
    ckpt = (
        train_mod.ROOT
        / "results"
        / "ci_resume_a"
        / train_mod.run_name(train_mod.load_config(_argv("ci_resume_a", 8)))
        / "ckpt"
        / "last.ckpt"
    )
    meta = load_checkpoint(ckpt)["meta"]
    assert meta["steps"] == 8 and "beta" in meta

    second = train_mod.main(_argv("ci_resume_b", 16, resume=ckpt))
    assert second == 0
    out = capsys.readouterr().out
    assert "resumed last.ckpt: steps=8 beta=" in out
    last = (
        train_mod.ROOT
        / "results"
        / "ci_resume_b"
        / train_mod.run_name(
            train_mod.load_config(_argv("ci_resume_b", 16, resume=ckpt))
        )
        / "ckpt"
        / "last.ckpt"
    )
    assert load_checkpoint(last)["meta"]["steps"] == 16
