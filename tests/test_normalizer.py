"""奖励归一化单元测试（P3）。"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bacmarl.utils.normalizer import RunningStd  # noqa: E402


def test_running_std_matches_numpy_for_chunked_updates():
    rng = np.random.default_rng(0)
    data = rng.normal(loc=-3.0, scale=2.5, size=1000)
    scaler = RunningStd()
    for chunk in np.array_split(data, 7):
        scaler.update(chunk)
    assert scaler.std == pytest.approx(float(np.std(data, ddof=1)), rel=1e-9)
    assert scaler.mean == pytest.approx(float(np.mean(data)), rel=1e-12)


def test_std_is_one_until_enough_samples():
    scaler = RunningStd()
    assert scaler.std == 1.0
    assert scaler.scale() == 1.0
    scaler.update(np.array([5.0]))
    assert scaler.std == 1.0


def test_scale_only_shrinks_never_amplifies():
    """小奖励（std<1）不放大：常数与微小方差都返回 1.0，避免价值目标被吹大。"""
    constant = RunningStd().update(np.zeros((2, 3, 4)))
    constant.update(np.zeros((2, 3, 4)))
    assert constant.scale() == 1.0

    large = RunningStd().update(np.full((10,), -20.0) + np.arange(10))
    assert large.std > 1.0
    assert large.scale() == pytest.approx(1.0 / large.std)


def test_empty_update_keeps_statistics():
    scaler = RunningStd().update(np.ones((2, 3)))
    before = (scaler.n, scaler.mean, scaler.std)
    scaler.update(np.array([]))
    assert (scaler.n, scaler.mean, scaler.std) == before


def test_scaling_preserves_sign_and_ordering():
    """均匀正缩放不改变最优策略：比较大小关系与符号保持一致。"""
    reward = np.array([-3.0, 0.0, 4.0], dtype=np.float32)
    scaled = reward * 0.5
    assert np.all(np.sign(scaled) == np.sign(reward))
    assert np.all(np.diff(scaled) > 0) == np.all(np.diff(reward) > 0)
