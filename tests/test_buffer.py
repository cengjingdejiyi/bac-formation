"""Rollout 缓存与 β 对偶测试（contracts 第 4 节）。"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from bacmarl.algos.buffer import RolloutBuffer  # noqa: E402
from bacmarl.algos.dual import BetaDual  # noqa: E402


def _buf(t=4, b=2, n=3, h=8):
    return RolloutBuffer(t, b, n, obs_dim=4, n_obs=2, state_dim=4 * n + 2, hidden=h, msg_bits=4)


def _fill(buf, rewards, bits=None):
    for i in range(buf.T):
        buf.add({
            'obs': np.zeros((buf.B, buf.N, 4), dtype=np.float32),
            'obs_mask': np.zeros((buf.B, buf.N, 2), dtype=bool),
            'state': np.zeros((buf.B, 4 * buf.N + 2), dtype=np.float32),
            'comm_mask': np.zeros((buf.B, buf.N, buf.N), dtype=bool),
            'h_prev': np.zeros((buf.B, buf.N, buf.hidden), dtype=np.float32),
            'action': np.zeros((buf.B, buf.N), dtype=np.int64),
            'gate': np.ones((buf.B, buf.N), dtype=np.float32),
            'logp_old': np.zeros((buf.B, buf.N), dtype=np.float32),
            'value': np.zeros((buf.B, buf.N), dtype=np.float32),
            'reward_team': rewards[i],
            'bits': bits[i] if bits is not None else np.zeros((buf.B, buf.N), dtype=np.float32),
            'done': np.zeros(buf.B, dtype=bool),
        })


def test_ptr_and_batched_shape():
    buf = _buf()
    _fill(buf, np.zeros((4, 2, 3), dtype=np.float32))
    assert buf.ptr == 4
    assert buf.batched(np.arange(8))['action'].shape == (8, 3)


def test_gae_returns_decrease_over_time():
    buf = _buf()
    rewards = np.ones((4, 2, 3), dtype=np.float32)
    _fill(buf, rewards, bits=np.zeros_like(rewards))
    buf.compute_returns(rewards, np.zeros((2, 3)), gamma=0.99, lam=0.95)
    assert np.all(buf.adv >= 0)
    assert buf.ret[0].mean() > buf.ret[-1].mean()


def test_beta_dual_clips_at_zero_and_tracks_budget():
    dual = BetaDual(beta=0.1, lr=0.1, b_max=2.0)
    assert dual.update(4.0) > 0.1
    dual.beta = 0.0
    assert dual.update(0.0) == 0.0
