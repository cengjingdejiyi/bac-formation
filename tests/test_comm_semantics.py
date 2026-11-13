"""用可解释的消息/有向边验证信息流，而不只验证张量形状。"""

import numpy as np
import pytest
import torch

from bacmarl.algos.dual import BetaDual
from bacmarl.comm.base import build_comm
from bacmarl.comm.channel import ChannelNoise, apply_noise, sample_noise
from bacmarl.comm.np_channel import apply_noise_np, sample_noise_np
from bacmarl.envs import EnvConfig, FormationCommEnv


def module(name):
    comm = build_comm(name, 4, {"msg_dim": 4, "msg_bits": 2})
    with torch.no_grad():
        comm.w_msg.weight.zero_()
        comm.w_msg.bias.zero_()
        if name == "bac":
            comm.w_msg.weight[:, 0] = 1.0
        else:
            comm.w_msg.weight.copy_(torch.eye(4))
        if name == "ic3net":
            comm.w_out.weight.copy_(torch.eye(4))
            comm.w_out.bias.fill_(0.7)
        else:
            comm.w_q.weight.zero_()
            comm.w_q.bias.zero_()
            comm.w_v.weight.zero_()
            comm.w_v.weight[: comm.w_v.in_features, :].copy_(
                torch.eye(comm.w_v.in_features)
            )
            comm.w_v.bias.fill_(0.5)
    return comm


def inputs():
    h = torch.tensor(
        [[[0.3, 0.2, 0.1, 0.4], [-1.0, -0.5, 0.6, 0.7], [0.5, 1.0, -0.2, 0.1]]]
    )
    mask = torch.zeros(1, 3, 3, dtype=torch.bool)
    mask[0, 0, 1] = True  # 只有接收者 0 能收到发送者 1，反向边不存在。
    return h, mask, torch.ones(1, 3)


@pytest.mark.parametrize("name", ["bac", "ic3net", "tarmac"])
def test_receiver_gets_the_senders_message_on_a_directed_edge(name):
    comm = module(name)
    h, mask, gate = inputs()
    out = comm(h, mask, gate, {})
    if name == "bac":
        expected = comm.w_v(comm.encode_msg(h[:, 1]))
    elif name == "ic3net":
        expected = comm.w_out(comm.w_msg(h[:, 1]))
    else:
        expected = comm.w_v(comm.w_msg(h[:, 1]))
    torch.testing.assert_close(out.c[:, 0], expected)
    torch.testing.assert_close(out.c[:, 1:], torch.zeros_like(out.c[:, 1:]))
    assert out.attn[0, 0, 1] == 1
    assert torch.count_nonzero(out.attn) == 1


@pytest.mark.parametrize("name", ["bac", "ic3net", "tarmac"])
def test_neighbor_affects_receiver_but_a_masked_agent_does_not(name):
    comm = module(name)
    h, mask, gate = inputs()
    original = comm(h, mask, gate, {}).c[:, 0]
    changed = h.clone()
    changed[:, 1, 0] += 3.0  # BAC 的硬消息符号也会翻转。
    assert (
        torch.linalg.vector_norm(comm(changed, mask, gate, {}).c[:, 0] - original)
        > 1e-5
    )
    changed = h.clone()
    changed[:, 2] += 100
    torch.testing.assert_close(comm(changed, mask, gate, {}).c[:, 0], original)
    mask.zero_()
    torch.testing.assert_close(
        comm(changed, mask, gate, {}).c, comm(h, mask, gate, {}).c
    )


@pytest.mark.parametrize("name", ["bac", "ic3net", "tarmac"])
def test_silent_senders_are_invisible_and_receiver_gate_does_not_block_reception(name):
    comm = module(name)
    h, mask, gate = inputs()
    original = comm(h, mask, gate, {}).c[:, 0]
    gate[:, 0] = 0  # 自己不发言仍可接收。
    torch.testing.assert_close(comm(h, mask, gate, {}).c[:, 0], original)
    gate[:, 1] = 0
    silent = comm(h, mask, gate, {})
    changed = h.clone()
    changed[:, 1] += 100
    torch.testing.assert_close(comm(changed, mask, gate, {}).c, silent.c)
    assert torch.count_nonzero(silent.c) == 0
    assert torch.count_nonzero(silent.attn) == 0


@pytest.mark.parametrize("name", ["bac", "ic3net", "tarmac"])
def test_all_masked_or_dropped_edges_have_zero_context_and_finite_gradients(name):
    comm = module(name)
    h, mask, gate = inputs()
    h.requires_grad_()
    noise = ChannelNoise(
        drop=mask.clone(), flip=torch.zeros(1, 3, 3, 2, dtype=torch.bool)
    )
    out = comm(h, mask, gate, {}, noise=noise)
    assert torch.count_nonzero(out.c) == 0
    assert torch.count_nonzero(out.attn) == 0
    out.c.sum().backward()
    assert torch.isfinite(h.grad).all()
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in comm.parameters()
    )


@pytest.mark.parametrize("name", ["bac", "ic3net", "tarmac"])
def test_dropped_sender_is_excluded_from_normalization_and_still_pays_bits(name):
    comm = module(name)
    h, mask, gate = inputs()
    mask[0, 0, 2] = True
    drop = torch.zeros_like(mask)
    drop[0, 0, 1] = True
    noise = ChannelNoise(drop, torch.zeros(1, 3, 3, 2, dtype=torch.bool))
    out = comm(h, mask, gate, {}, noise=noise)
    assert out.attn[0, 0, 1] == 0 and out.attn[0, 0, 2] == 1
    only_second = mask.clone()
    only_second[0, 0, 1] = False
    torch.testing.assert_close(out.c, comm(h, only_second, gate, {}).c)
    torch.testing.assert_close(out.bits, gate * comm.bits_per_msg)


@pytest.mark.parametrize("name", ["bac", "ic3net", "tarmac"])
def test_attention_respects_environment_top_k(name):
    with FormationCommEnv(
        EnvConfig(backend="mock", n_agents=6, comm_radius=10.0, recv_k=2)
    ) as env:
        mask = torch.as_tensor(env.reset(seed=0).comm_mask)[None]
    out = module(name)(torch.ones(1, 6, 4), mask, torch.ones(1, 6), {})
    assert ((out.attn != 0).sum(-1) <= 2).all()
    assert not out.attn[~mask].any()
    torch.testing.assert_close(out.attn.sum(-1), torch.ones(1, 6))


@pytest.mark.parametrize("binary", [True, False])
def test_numpy_and_torch_channel_match_hand_computed_directed_noise(binary):
    msgs = np.array([[[1, -1], [-1, -1], [1, 1]]], dtype=np.float32)
    mask = np.zeros((1, 3, 3), dtype=bool)
    mask[0, 0, 1] = mask[0, 1, 2] = True
    drop = np.zeros_like(mask)
    drop[0, 1, 2] = True
    flip = np.zeros((1, 3, 3, 2), dtype=bool)
    flip[0, 0, 1, 0] = True
    expected = np.zeros_like(flip, dtype=np.float32)
    expected[0, 0, 1] = [1 if binary else -1, -1]
    actual_np = apply_noise_np(msgs, mask, {"drop": drop, "flip": flip}, binary=binary)
    actual_torch = apply_noise(
        torch.from_numpy(msgs),
        torch.from_numpy(mask),
        ChannelNoise(torch.from_numpy(drop), torch.from_numpy(flip)),
        binary=binary,
    )
    np.testing.assert_array_equal(actual_np, expected)
    np.testing.assert_array_equal(actual_torch.numpy(), expected)


def test_channel_samples_each_environments_probability_without_averaging():
    mask = torch.ones(2, 3, 3, dtype=torch.bool)
    noise = sample_noise(mask, 2, torch.tensor([0.0, 1.0]), torch.tensor([1.0, 0.0]))
    assert not noise.drop[0].any() and noise.drop[1].all()
    assert noise.flip[0].all() and not noise.flip[1].any()
    np_noise = sample_noise_np(
        np.random.default_rng(0),
        mask.numpy(),
        2,
        np.array([0.0, 1.0]),
        np.array([1.0, 0.0]),
    )
    np.testing.assert_array_equal(noise.drop.numpy(), np_noise["drop"])
    np.testing.assert_array_equal(noise.flip.numpy(), np_noise["flip"])


def test_bac_binary_forward_and_ste_reaches_the_neighbor_encoder():
    comm = module("bac")
    h, mask, gate = inputs()
    h.requires_grad_()
    message = comm.encode_msg(h)
    assert torch.all(message.abs() == 1)
    out = comm(h, mask, gate, {})
    out.c[0, 0, 0].backward()
    assert h.grad[0, 1].abs().sum() > 0
    assert comm.w_msg.weight.grad.abs().sum() > 0
    assert h.grad[0, 2].abs().sum() == 0


def test_broadcast_budget_counts_senders_once_independent_of_edge_count():
    comm = module("bac")
    h, mask, gate = inputs()
    gate[0, 1] = 0
    sparse = comm(h, mask, gate, {})
    dense = comm(h, torch.ones_like(mask), gate, {"drop_p": 1.0})
    torch.testing.assert_close(sparse.bits, torch.tensor([[2.0, 0.0, 2.0]]))
    torch.testing.assert_close(dense.bits, sparse.bits)
    mean_bits = float(torch.stack([sparse.bits, dense.bits]).mean())
    dual = BetaDual(beta=0.1, lr=0.3, b_max=1.0)
    assert dual.update(mean_bits) == pytest.approx(0.1 + 0.3 * (4 / 3 - 1))
