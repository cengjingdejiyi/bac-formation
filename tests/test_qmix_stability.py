"""QMIX 数值稳定性：TD 目标夹逼、Huber 损失、日志列（2026-11-21 修复的回归测试）。

背景：旧配置（MSE + lr 5e-4 + 每次 4 个梯度步）在 200k 步内把 `qmix/q_tot` 从 −5
顶到 5360、`loss/td` 峰值 9.4e6，与 PPO 侧 `value_norm` 未生效同一类"价值目标量级
失控"。这里锁定三件事：目标有界、离群样本的损失有界、日志列存在。
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from bacmarl.algos.qmix import QMixAgent, QMixLossStats, qmix_td_loss, td_target

OBS_DIM, N_OBS, N, H, STATE, B = 10, 7, 4, 16, 12, 8
N_ACTIONS = 5


def _agent(seed: int = 0) -> QMixAgent:
    torch.manual_seed(seed)
    return QMixAgent(
        obs_dim=OBS_DIM,
        n_obs=N_OBS,
        n_actions=N_ACTIONS,
        hidden=H,
        n_agents=N,
        state_dim=STATE,
        mixer_embed=8,
        hypernet_embed=16,
        use_agent_id=True,
    )


def _batch(size: int = B, reward: float = -0.09, q_scale: float = 1.0) -> dict:
    torch.manual_seed(1)
    return {
        "obs": torch.randn(size, N, OBS_DIM),
        "obs_mask": torch.ones(size, N, N_OBS, dtype=torch.bool),
        "action": torch.zeros(size, N, dtype=torch.long),
        "reward": torch.full((size,), reward),
        "done": torch.zeros(size),
        "state": torch.randn(size, STATE) * q_scale,
        "h_prev": torch.zeros(size, N, H),
        "next_obs": torch.randn(size, N, OBS_DIM),
        "next_obs_mask": torch.ones(size, N, N_OBS, dtype=torch.bool),
        "next_h": torch.zeros(size, N, H),
        "next_state": torch.randn(size, STATE) * q_scale,
    }


def test_td_target_clips_when_requested():
    """夹逼只拦越界值，合法的目标原样保留。"""
    r = torch.tensor([-0.09, -0.09])
    done = torch.zeros(2)
    q_next = torch.tensor([5000.0, -1.0])
    raw = td_target(r, done, q_next, gamma=0.99)
    clipped = td_target(r, done, q_next, gamma=0.99, clip=50.0)
    assert raw[0].item() > 4900.0
    assert clipped[0].item() == pytest.approx(50.0)
    assert clipped[1].item() == pytest.approx(-0.09 + 0.99 * -1.0)


def test_td_target_without_clip_is_unchanged():
    """默认（clip=None）行为必须与修复前完全一致，避免影响已有脚本。"""
    r = torch.tensor([1.0])
    done = torch.tensor([0.0])
    q_next = torch.tensor([10.0])
    assert td_target(r, done, q_next, gamma=0.9).item() == pytest.approx(10.0)


def test_huber_loss_stays_bounded_on_outlier_target():
    """同一个坏 target：MSE 的 loss 上百万，Huber 必须落在个位数量级。"""
    agent = _agent()
    target = agent.target_copy()
    batch = _batch(size=B)
    # 把 next_state 放大到极端，制造一个量级失控的 Q_tot(s')
    batch["next_state"] = batch["next_state"] * 300.0
    batch["next_obs"] = batch["next_obs"] * 300.0
    batch["next_obs_mask"] = torch.ones(B, N, N_OBS, dtype=torch.bool)
    with torch.no_grad():
        # 先把 mixer 的偏置抬高，模拟已经发散到几千的 target 网络
        for p in target.mixer.parameters():
            p.add_(torch.randn_like(p) * 5.0)

    mse_loss, mse_stats = qmix_td_loss(
        agent, target, batch, gamma=0.99, huber=False, target_clip=None
    )
    hub_loss, hub_stats = qmix_td_loss(
        agent, target, batch, gamma=0.99, huber=True, huber_beta=1.0,
        target_clip=None,
    )
    assert np.isfinite(mse_loss.item()) and np.isfinite(hub_loss.item())
    assert hub_loss.item() < mse_stats.td_abs * 5.0  # 线性段：loss ~ |TD|
    assert hub_loss.item() < mse_loss.item() or mse_stats.td_abs < 1.0


def test_target_clip_frac_is_reported():
    """夹逼比例要能进日志，用于判断"到底有多少样本越界"。"""
    agent = _agent()
    target = agent.target_copy()
    batch = _batch()
    with torch.no_grad():
        for p in target.mixer.parameters():
            p.add_(torch.randn_like(p) * 10.0)
    _, stats = qmix_td_loss(
        agent, target, batch, gamma=0.99, target_clip=0.5, huber=True
    )
    assert isinstance(stats, QMixLossStats)
    assert 0.0 <= stats.clip_frac <= 1.0
    assert stats.clip_frac > 0.0  # 0.5 的界太紧，一定有样本被夹
    _, no_clip_stats = qmix_td_loss(agent, target, batch, gamma=0.99, target_clip=None)
    assert no_clip_stats.clip_frac == 0.0


def test_gradients_stay_finite_with_bad_sample():
    """坏样本下梯度必须有限：NaN 一旦进 Adam，后面整条曲线都作废。"""
    agent = _agent()
    target = agent.target_copy()
    batch = _batch()
    batch["next_state"] = batch["next_state"] * 1e4
    batch["next_obs"] = batch["next_obs"] * 1e4
    loss, stats = qmix_td_loss(
        agent, target, batch, gamma=0.99, huber=True, target_clip=50.0
    )
    loss.backward()
    for p in agent.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()
    assert np.isfinite(stats.loss)
