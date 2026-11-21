"""价值目标尺度归一化（value_norm）的单元测试（负责人 C/A）。

背景：`reward_norm` 的 `RunningStd.scale()` 只缩小不放大（min_std=1.0），而本任务的
每步 shaped reward 量级在 0.5–1.3，std 恒 < 1 → 缩放系数永远是 1.0，价值目标仍是
O(100) 的回合回报，`loss/value` 实测 76~400。`value_norm` 改成缩放**价值回归目标**，
不动策略梯度。
"""

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bacmarl.runners import train as train_mod  # noqa: E402
from bacmarl.utils.normalizer import RunningStd  # noqa: E402


def test_reward_norm_scale_is_a_noop_for_this_task():
    """记录现状：每步奖励 std≈0.5 时 scale() 仍为 1.0，所以 reward_norm 形同虚设。"""
    rng = np.random.default_rng(0)
    step_rewards = rng.normal(loc=-0.9, scale=0.5, size=(64, 8, 6))
    scaler = RunningStd()
    for _ in range(4):
        scaler.update(step_rewards)
    assert scaler.std < 1.0
    assert scaler.scale() == 1.0


def test_value_scale_shrinks_long_horizon_returns():
    """回合回报 -60~-120（std≈30）→ 目标压到 O(1)；真实 run 的 std 量级就是这样。"""
    rets = [-60.0, -150.0, -90.0, -80.0, -120.0]
    eps = [{"ret": v} for v in rets]
    scale = train_mod.value_target_scale(eps)
    assert scale == pytest.approx(1.0 / float(np.std(rets)), rel=1e-9)
    assert abs(scale * -90.0) < 10.0


def test_value_scale_never_amplifies_and_handles_empty():
    assert train_mod.value_target_scale([]) == 1.0
    flat = [{"ret": -3.0} for _ in range(10)]
    assert train_mod.value_target_scale(flat) == 1.0


def test_reward_norm_and_value_norm_conflict_raises():
    with pytest.raises(ValueError, match="互斥"):
        train_mod.validate_norm_cfg({"reward_norm": True, "value_norm": True})
    train_mod.validate_norm_cfg({"reward_norm": False, "value_norm": True})
    train_mod.validate_norm_cfg({"reward_norm": True, "value_norm": False})
    train_mod.validate_norm_cfg({})
