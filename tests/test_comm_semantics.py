"""A 验收回归：BAC 通信语义（边方向 / 空邻居 / 丢包归一化 / 逐环境信道）。

对应 issue #2 的 3 项 P1 与 1 项 P2，固定为仓库回归测试；无 torch 时整体跳过。

边约定：``comm_mask[b, i, j]`` 中 i 是接收者、j 是发送者，``edge[b, i, j] = msgs[b, j]``。
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

torch = pytest.importorskip('torch')

from bacmarl.algos.dual import BetaDual  # noqa: E402
from bacmarl.comm.bac import CommBAC  # noqa: E402
from bacmarl.comm.channel import ChannelNoise, apply_noise, sample_noise  # noqa: E402
from bacmarl.comm.np_channel import apply_noise_np, sample_noise_np  # noqa: E402
from bacmarl.models.actor import Actor  # noqa: E402


def simple_comm(bias: float = 0.0) -> CommBAC:
    """hidden=1、msg_bits=1 的定参 BAC：消息 = sign(h)，上下文 = 有效发送者消息。"""
    comm = CommBAC(hidden=1, msg_bits=1)
    with torch.no_grad():
        comm.w_msg.weight.fill_(1.0)
        comm.w_msg.bias.zero_()
        comm.w_q.weight.zero_()
        comm.w_q.bias.zero_()
        comm.w_k.weight.zero_()
        comm.w_k.bias.zero_()
        comm.w_v.weight.fill_(1.0)
        comm.w_v.bias.fill_(bias)
    return comm


def zero_noise(mask, length: int = 1) -> ChannelNoise:
    return ChannelNoise(torch.zeros_like(mask),
                        torch.zeros((*mask.shape, length), dtype=torch.bool))


# ---- P1-1：接收者读发送者的消息（torch / numpy 两条信道实现）----------
def test_torch_edge_contains_sender_message():
    msgs = torch.tensor([[[-1.0], [1.0]]])                 # 发送者 0 = -1，发送者 1 = +1
    mask = torch.tensor([[[False, True], [False, False]]])  # 接收者 0 收发送者 1
    edge = apply_noise(msgs, mask, zero_noise(mask))
    assert edge[0, 0, 1, 0].item() == 1.0
    assert edge[0, 1, 0, 0].item() == 0.0                   # 无物理边 = 0


def test_numpy_edge_contains_sender_message():
    msgs = np.array([[[-1.0], [1.0]]], dtype=np.float32)
    mask = np.array([[[False, True], [False, False]]])
    noise = {"drop": np.zeros_like(mask), "flip": np.zeros((*mask.shape, 1), dtype=bool)}
    edge = apply_noise_np(msgs, mask, noise)
    assert edge[0, 0, 1, 0] == 1.0
    assert edge[0, 1, 0, 0] == 0.0


def test_bac_receiver_responds_to_sender_message():
    comm = simple_comm()
    mask = torch.tensor([[[False, True], [False, False]]])
    h = torch.tensor([[[-1.0], [1.0]]])
    out = comm(h, mask, torch.ones(1, 2), {}, noise=zero_noise(mask))
    assert out.c[0, 0, 0].item() == 1.0
    assert out.attn[0, 0, 1].item() == 1.0


def test_gate_uses_sender_axis():
    """门控按发送者 j 生效：只有 j=1 发言时，i=0 只看到 j=1 的边。"""
    comm = simple_comm()
    mask = torch.zeros(1, 3, 3, dtype=torch.bool)
    mask[0, 0, 1:] = True                                   # 0 收 1、2
    gate = torch.tensor([[0.0, 1.0, 0.0]])
    out = comm(torch.ones(1, 3, 1), mask, gate, {}, noise=zero_noise(mask))
    assert out.attn[0, 0, 1].item() == 1.0
    assert out.attn[0, 0, 2].item() == 0.0
    assert out.c[0, 0, 0].item() == 1.0


# ---- P1-2：无消息时上下文和注意力都为零 --------------------------------
@pytest.mark.parametrize("case", ["no_neighbors", "all_silent", "all_dropped"])
def test_no_received_messages_means_zero_context_and_attention(case):
    comm = simple_comm(bias=2.0)                            # 偏置非零，仍应为全零
    mask = ~torch.eye(3, dtype=torch.bool)[None]
    gate = torch.ones(1, 3)
    noise = zero_noise(mask)
    if case == "no_neighbors":
        mask = torch.zeros_like(mask)
    elif case == "all_silent":
        gate.zero_()
    else:
        noise.drop.copy_(mask)
    out = comm(torch.ones(1, 3, 1), mask, gate, {}, noise=noise)
    assert torch.count_nonzero(out.c).item() == 0, f"{case}: context={out.c.tolist()}"
    assert torch.count_nonzero(out.attn).item() == 0, f"{case}: attention={out.attn.tolist()}"


# ---- P1-3：丢包边不参与注意力归一化 ------------------------------------
def test_dropped_edge_is_excluded_from_attention_normalization():
    comm = simple_comm()
    mask = torch.zeros(1, 3, 3, dtype=torch.bool)
    mask[0, 0, 1:] = True
    noise = zero_noise(mask)
    noise.drop[0, 0, 1] = True
    out = comm(torch.ones(1, 3, 1), mask, torch.ones(1, 3), {}, noise=noise)
    assert out.attn[0, 0, 1].item() == 0.0
    assert out.attn[0, 0, 2].item() == 1.0
    assert out.attn[0, 0].sum().item() == 1.0


# ---- P2：逐环境信道概率不被取均值 --------------------------------------
def test_each_parallel_environment_keeps_its_channel_probability():
    comm = simple_comm()
    mask = (~torch.eye(3, dtype=torch.bool)[None]).repeat(2, 1, 1)
    out = comm(torch.ones(2, 3, 1), mask, torch.ones(2, 3),
               {"drop_p": np.array([0.0, 1.0], dtype=np.float32), "flip_p": np.zeros(2)})
    assert not out.noise.drop[0].any()
    assert torch.equal(out.noise.drop[1], mask[1])


@pytest.mark.parametrize("api", ["torch", "numpy"])
def test_sample_noise_accepts_per_env_probability(api):
    mask = (~np.eye(3, dtype=bool)[None]).repeat(2, 0)
    if api == "torch":
        noise = sample_noise(torch.as_tensor(mask), 1, np.array([0.0, 1.0]),
                             np.zeros(2))
        assert not noise.drop[0].any() and torch.equal(noise.drop[1], torch.as_tensor(mask[1]))
    else:
        noise = sample_noise_np(np.random.default_rng(0), mask, 1, np.array([0.0, 1.0]), 0.0)
        assert not noise["drop"][0].any() and np.array_equal(noise["drop"][1], mask[1])


# ---- 既有语义不回退 ----------------------------------------------------
def test_sign_output_and_straight_through_gradient():
    comm = simple_comm()
    h = torch.tensor([[[-2.0], [0.0], [2.0]]], requires_grad=True)
    msg = comm.encode_msg(h)
    assert torch.equal(msg, torch.tensor([[[-1.0], [1.0], [1.0]]]))
    msg.sum().backward()
    assert torch.equal(h.grad, torch.ones_like(h))


def test_bits_charge_sender_once_even_when_packet_is_dropped():
    comm = CommBAC(hidden=1, msg_bits=4)
    mask = ~torch.eye(3, dtype=torch.bool)[None]
    gate = torch.tensor([[1.0, 0.0, 1.0]])
    noise = zero_noise(mask, length=4)
    noise.drop.copy_(mask)
    out = comm(torch.ones(1, 3, 1), mask, gate, {}, noise=noise)
    assert torch.equal(out.bits, torch.tensor([[4.0, 0.0, 4.0]]))


def test_stored_noise_reproduces_context_and_attention():
    torch.manual_seed(7)
    comm = CommBAC(hidden=4, msg_bits=2)
    mask = ~torch.eye(3, dtype=torch.bool)[None]
    noise = sample_noise(mask, 2, 0.3, 0.2)
    h = torch.randn(1, 3, 4)
    gate = torch.ones(1, 3)
    first = comm(h, mask, gate, {}, noise=noise)
    second = comm(h, mask, gate, {}, noise=noise)
    assert torch.equal(first.c, second.c)
    assert torch.equal(first.attn, second.attn)


def test_binary_flip_applies_to_each_bit():
    msgs = torch.tensor([[[-1.0], [1.0]]])
    mask = torch.tensor([[[False, True], [False, False]]])
    noise = zero_noise(mask)
    noise.flip[0, 0, 1, 0] = True
    before = apply_noise(msgs, mask, zero_noise(mask))
    after = apply_noise(msgs, mask, noise)
    assert after[0, 0, 1, 0] == -before[0, 0, 1, 0]


def test_actor_joint_log_probability_contains_gate_action():
    torch.manual_seed(3)
    actor = Actor(obs_dim=6, n_obs=1, n_actions=5, hidden=4,
                  comm_module=CommBAC(hidden=4, msg_bits=2))
    mask = ~torch.eye(2, dtype=torch.bool)[None]
    gate = torch.tensor([[0.0, 1.0]])
    out = actor(torch.zeros(1, 2, 6), torch.ones(1, 2, 1, dtype=torch.bool),
                mask, actor.init_hidden(1, 2), {}, gate=gate, noise=zero_noise(mask, 2))
    action = torch.tensor([[0, 1]])
    expected = (torch.distributions.Categorical(logits=out.action_logits).log_prob(action)
                + torch.distributions.Bernoulli(logits=out.gate_logits).log_prob(gate))
    assert torch.allclose(actor.log_prob(out, action), expected)


def test_beta_update_and_individual_reward_match_contract():
    dual = BetaDual(beta=0.1, lr=0.05, b_max=2.0)
    reward = dual.shaped_reward(np.ones((1, 2)), np.array([[0.0, 4.0]]))
    assert np.allclose(reward, [[1.0, 0.6]])
    assert dual.update(4.0) == pytest.approx(0.2)
    assert dual.update(0.0) == pytest.approx(0.1)
    assert dual.update(0.0) == pytest.approx(0.0)
    assert dual.update(0.0) == 0.0
