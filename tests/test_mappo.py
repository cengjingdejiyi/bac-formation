"""PPO 更新测试（contracts 第 4 节，负责人 C）。

覆盖：优势整轮归一化、价值目标裁断、存储噪声复用、门控项、日志字段。
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

torch = pytest.importorskip("torch")

from bacmarl.algos.buffer import RolloutBuffer  # noqa: E402
from bacmarl.algos.mappo import ppo_update  # noqa: E402
from bacmarl.comm.base import CommModule, CommOut  # noqa: E402
from bacmarl.models.actor import Actor  # noqa: E402
from bacmarl.models.critic import Critic  # noqa: E402

N, H, OBS, NOBS, STATE, N_ACTION = 3, 8, 4, 2, 4 * 3 + 2, 5


class ZeroComm(CommModule):
    """零通信上下文，便于隔离 PPO 本身的数学；可选带门控以覆盖 Bernoulli 项。"""

    has_gate = False
    bits_per_msg = 0

    def __init__(self, hidden: int, gated: bool = False):
        super().__init__(hidden=hidden, msg_dim=0, msg_bits=0, recv_k=1)
        self.has_gate = gated
        self.seen_noise = []

    def forward(self, h, comm_mask, gate, channel, noise=None):
        self.seen_noise.append(noise)
        b, n, _ = h.shape
        return CommOut(
            c=torch.zeros_like(h),
            bits=torch.zeros(b, n),
            attn=comm_mask.to(h.dtype),
            noise=noise,
        )


def _cfg(**overrides):
    cfg = {
        "clip_eps": 0.2,
        "entropy_coef": 0.0,
        "ppo_epochs": 2,
        "minibatch": 8,
        "value_clip_eps": 0.1,
        "norm_adv": True,
    }
    cfg.update(overrides)
    return cfg


def _buffer(t=4, b=2, gated=False, seed=0):
    torch.manual_seed(seed)
    comm = ZeroComm(H, gated=gated)
    actor = Actor(OBS, NOBS, N_ACTION, H, comm)
    critic = Critic("local", STATE, OBS, N, H)
    opt = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=1e-2
    )
    buf = RolloutBuffer(t, b, N, OBS, NOBS, STATE, H, msg_bits=0)
    rng = np.random.default_rng(seed)
    buf.ptr = t
    buf.obs[:] = rng.normal(size=(t, b, N, OBS)).astype(np.float32)
    buf.obs_mask[:] = True
    buf.state[:] = rng.normal(size=(t, b, STATE)).astype(np.float32)
    buf.comm_mask[:] = False
    buf.h_prev[:] = rng.normal(size=(t, b, N, H)).astype(np.float32)
    buf.action[:] = rng.integers(0, N_ACTION, size=(t, b, N))
    buf.gate[:] = 1.0
    buf.logp_old[:] = 0.0
    buf.value[:] = 0.5
    buf.ret[:] = 0.5
    buf.adv[:] = 0.0
    return actor, critic, opt, buf, comm


def _batch(buf, **arrays):
    for name, value in arrays.items():
        getattr(buf, name)[:] = value
    return {k: torch.as_tensor(v) for k, v in buf.batched(buf.flat_indices()).items()}


def test_ppo_update_moves_parameters_and_reports_stats():
    actor, critic, opt, buf, _ = _buffer()
    batch = _batch(
        buf, adv=np.linspace(-1, 1, 4 * 2 * N).reshape(4, 2, N).astype(np.float32)
    )
    before = [p.detach().clone() for p in actor.parameters()]
    logs = ppo_update(actor, critic, opt, buf, _cfg(), batch)

    for key in (
        "loss/policy",
        "loss/value",
        "loss/entropy",
        "entropy",
        "ppo/approx_kl",
        "ppo/clip_frac",
        "ppo/adv_std",
    ):
        assert key in logs, key
    assert all(np.isfinite(v) for v in logs.values())
    assert any(
        not torch.equal(a, b)
        for a, b in zip(before, [p.detach() for p in actor.parameters()])
    )


def test_constant_advantages_normalize_to_zero_policy_loss():
    """整轮归一化的直接后果：优势全相等时归一化为 0，policy loss 恰为 0。"""
    actor, critic, opt, buf, _ = _buffer()
    batch = _batch(buf, adv=np.full((4, 2, N), 7.0, dtype=np.float32))
    logs = ppo_update(actor, critic, opt, buf, _cfg(), batch)
    assert logs["ppo/adv_std"] == pytest.approx(0.0, abs=1e-6)
    assert logs["loss/policy"] == pytest.approx(0.0, abs=1e-6)


def test_stored_channel_noise_is_reused_instead_of_resampled():
    """更新必须复用采样期噪声：drop 全为 True 时不可能来自 drop_p=0 的重新采样。"""
    actor, critic, opt, buf, comm = _buffer()
    batch = _batch(buf, adv=np.ones((4, 2, N), dtype=np.float32))
    batch["noise_drop"] = torch.ones_like(batch["noise_drop"])
    batch["noise_flip"] = torch.zeros_like(batch["noise_flip"])

    ppo_update(actor, critic, opt, buf, _cfg(ppo_epochs=1, minibatch=4), batch)

    assert comm.seen_noise, "通信模块没有收到 noise"
    for noise in comm.seen_noise:
        assert noise is not None
        assert bool(noise.drop.all()), "stored drop mask 没有被复用"


def test_value_target_clipping_bounds_single_update_step():
    """价值目标只在 v_old ± value_clip_eps 内可动；不裁剪时会被大回报推走。"""
    eps = 0.1
    moved = {}
    for clip in (eps, 1e9):
        actor, critic, _, buf, _ = _buffer(t=1, b=4)
        obs = np.zeros((4, N, OBS), dtype=np.float32)
        state = np.zeros((4, STATE), dtype=np.float32)
        with torch.no_grad():
            v_old = critic(state=torch.as_tensor(state), obs=torch.as_tensor(obs))
        buf.obs[:] = obs[None]
        buf.state[:] = state[None]
        batch = _batch(
            buf,
            adv=np.zeros((1, 4, N), dtype=np.float32),
            ret=(v_old.numpy() + 10.0)[None],
            value=v_old.numpy()[None],
        )
        cfg = _cfg(value_clip_eps=clip, ppo_epochs=50, minibatch=8)
        opt = torch.optim.SGD(
            list(actor.parameters()) + list(critic.parameters()), lr=1e-3
        )
        ppo_update(actor, critic, opt, buf, cfg, batch)
        with torch.no_grad():
            v_new = critic(state=torch.as_tensor(state), obs=torch.as_tensor(obs))
        moved[clip] = float((v_new - v_old).abs().max())

    assert moved[eps] <= eps * 1.5, moved
    assert moved[1e9] > 3 * moved[eps], moved


def test_gated_comm_adds_bernoulli_term_and_stays_finite():
    actor, critic, opt, buf, _ = _buffer(gated=True)
    batch = _batch(
        buf, adv=np.linspace(-1, 1, 4 * 2 * N).reshape(4, 2, N).astype(np.float32)
    )
    logs = ppo_update(actor, critic, opt, buf, _cfg(), batch)
    assert np.isfinite(logs["loss/policy"])
    assert np.isfinite(logs["entropy"])


def test_minibatch_and_epochs_cover_every_sample():
    """小批量切分必须覆盖全部样本：ppo_epochs=1、minibatch 整除时报错或跑满。"""
    actor, critic, opt, buf, comm = _buffer(t=3, b=2)
    batch = _batch(buf, adv=np.ones((3, 2, N), dtype=np.float32))
    ppo_update(actor, critic, opt, buf, _cfg(ppo_epochs=1, minibatch=4), batch)
    # 6 个样本按 minibatch=4 切成 4+2 两块
    assert len(comm.seen_noise) == 2


def test_value_clip_off_by_default_and_reported_when_enabled():
    """value_clip_eps 为 None 时不裁断；启用时报告裁断比例。"""
    actor, critic, opt, buf, _ = _buffer()
    batch = _batch(buf, adv=np.zeros((4, 2, N), dtype=np.float32))

    disabled = ppo_update(
        actor, critic, opt, buf, _cfg(ppo_epochs=1, value_clip_eps=None), batch
    )
    assert disabled["ppo/value_clip_frac"] == 0.0

    # buf.value=0.5 而 critic 初始输出约 0，|v-v_old| 远大于 1e-6，裁断必然触发。
    enabled = ppo_update(
        actor, critic, opt, buf, _cfg(ppo_epochs=1, value_clip_eps=1e-6), batch
    )
    assert enabled["ppo/value_clip_frac"] == pytest.approx(1.0)
