"""BAC 消融开关与 β 对偶口径的回归测试（负责人 A）。

覆盖：mean 聚合与 softmax 的差异、可见边归一化、no_gate 时门控头与 bits、
build_comm 的消融接线、β 的上下界/均值口径/断点状态。
"""

import math
import sys
from pathlib import Path

import pytest
import torch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:  # 允许单文件运行 pytest
    sys.path.insert(0, str(SRC))

from bacmarl.algos.dual import BetaDual  # noqa: E402
from bacmarl.comm.bac import CommBAC  # noqa: E402
from bacmarl.comm.base import build_comm  # noqa: E402
from bacmarl.comm.channel import ChannelNoise  # noqa: E402
from bacmarl.models.actor import Actor  # noqa: E402

HIDDEN, MSG_BITS, N = 4, 2, 3


def _comm(aggregate="softmax", use_gate=True):
    """固定权重：消息 = sign(h[:2])，value 投影取前两维，便于手算。"""
    comm = CommBAC(HIDDEN, msg_bits=MSG_BITS, aggregate=aggregate, use_gate=use_gate)
    with torch.no_grad():
        for layer in (comm.w_msg, comm.w_q, comm.w_k, comm.w_v):
            layer.weight.zero_()
            layer.bias.zero_()
        comm.w_msg.weight[0, 0] = 1.0
        comm.w_msg.weight[1, 1] = 1.0
        comm.w_v.weight[0, 0] = 1.0
        comm.w_v.weight[1, 1] = 1.0
    return comm


def _inputs():
    """0 号接收 1、2 号的消息：msg1=+1,+1；msg2=-1,+1。"""
    h = torch.tensor(
        [
            [
                [1.0, 0.5, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
                [-1.0, 1.0, 0.0, 0.0],
            ]
        ]
    )
    mask = torch.zeros(1, N, N, dtype=torch.bool)
    mask[0, 0, 1] = mask[0, 0, 2] = True
    return h, mask, torch.ones(1, N)


def test_mean_aggregation_is_a_plain_average_over_visible_senders():
    comm, (h, mask, gate) = _comm("mean"), _inputs()
    out = comm(h, mask, gate, {})
    assert out.attn[0, 0, 1] == pytest.approx(0.5)
    assert out.attn[0, 0, 2] == pytest.approx(0.5)
    # 一半 +1、一半 -1 的消息在第 0 维抵掉，第 1 维都是 +1。
    torch.testing.assert_close(out.c[0, 0], torch.tensor([0.0, 1.0, 0.0, 0.0]))
    assert torch.count_nonzero(out.attn[0, 1:]) == 0
    assert torch.count_nonzero(out.c[0, 1:]) == 0
    torch.testing.assert_close(out.attn.sum(-1), torch.tensor([[1.0, 0.0, 0.0]]))


def test_mean_aggregation_excludes_dropped_and_masked_senders():
    comm, (h, mask, gate) = _comm("mean"), _inputs()
    drop = torch.zeros_like(mask)
    drop[0, 0, 2] = True  # 2 号整条丢包
    noise = ChannelNoise(drop, torch.zeros(1, N, N, MSG_BITS, dtype=torch.bool))
    out = comm(h, mask, gate, {}, noise=noise)
    assert out.attn[0, 0, 1] == pytest.approx(1.0)
    assert out.attn[0, 0, 2] == 0
    torch.testing.assert_close(out.c[0, 0], torch.tensor([1.0, 1.0, 0.0, 0.0]))

    only_one = mask.clone()
    only_one[0, 0, 2] = False
    torch.testing.assert_close(out.c, comm(h, only_one, gate, {}).c)


def test_softmax_differs_from_mean_when_the_receiver_queries_differently():
    comm, (h, mask, gate) = _comm("softmax"), _inputs()
    with torch.no_grad():
        comm.w_q.weight[0, 0] = 1.0
        comm.w_k.weight[0, 0] = 1.0
    out = comm(h, mask, gate, {})
    mean_out = _comm("mean")(h, mask, gate, {})
    # score(0,1)=+1、score(0,2)=-1，注意力必然偏向 1 号。
    assert out.attn[0, 0, 1] > 0.5 > out.attn[0, 0, 2]
    assert not torch.allclose(out.c, mean_out.c)


def test_masked_rows_stay_zero_in_both_aggregations():
    for aggregate in ("softmax", "mean"):
        comm, (h, mask, gate) = _comm(aggregate), _inputs()
        mask.zero_()
        out = comm(h, mask, gate, {})
        assert torch.count_nonzero(out.c) == 0
        assert torch.count_nonzero(out.attn) == 0
        torch.testing.assert_close(out.bits, gate * MSG_BITS)


def test_no_gate_ablation_removes_the_gate_head_and_pins_bits_to_msg_bits():
    comm, (h, mask, _) = _comm("softmax", use_gate=False), _inputs()
    assert comm.has_gate is False
    gate = torch.ones(1, N)
    out = comm(h, mask, gate, {})
    assert torch.all(out.bits == MSG_BITS)
    # 门控参数被忽略：全静默的 gate 也照常收发。
    torch.testing.assert_close(comm(h, mask, torch.zeros(1, N), {}).c, out.c)

    actor = Actor(obs_dim=4, n_obs=2, n_actions=5, hidden=HIDDEN, comm_module=comm)
    assert actor.gate_head is None
    obs = torch.zeros(1, N, 4)
    obs_mask = torch.ones(1, N, 2, dtype=torch.bool)
    out_actor = actor(obs, obs_mask, mask, actor.init_hidden(1, N), {})
    assert out_actor.gate_logits is None
    assert torch.all(out_actor.gate == 1)
    assert torch.all(out_actor.bits == MSG_BITS)


def test_gated_bac_keeps_the_gate_head():
    comm = _comm("softmax", use_gate=True)
    assert comm.has_gate is True
    actor = Actor(obs_dim=4, n_obs=2, n_actions=5, hidden=HIDDEN, comm_module=comm)
    assert actor.gate_head is not None


def test_build_comm_wires_the_ablation_switches():
    default = build_comm("bac", HIDDEN, {"msg_bits": MSG_BITS})
    assert default.aggregate == "softmax" and default.has_gate is True
    ablated = build_comm(
        "bac", HIDDEN, {"msg_bits": MSG_BITS, "aggregate": "mean", "use_gate": False}
    )
    assert ablated.aggregate == "mean" and ablated.has_gate is False
    with pytest.raises(ValueError):
        CommBAC(HIDDEN, msg_bits=MSG_BITS, aggregate="avg")


def test_beta_dual_clamps_uses_the_mean_and_round_trips():
    dual = BetaDual(beta=0.1, lr=0.05, b_max=2.0, beta_max=1.0)
    assert dual.update(torch.tensor([[2.0, 2.0], [2.0, 2.0]])) == pytest.approx(0.1)
    assert dual.update(torch.full((2, 2, 3), 4.0)) == pytest.approx(0.2)
    assert dual.update(torch.full((2, 2, 3), 100.0)) == pytest.approx(1.0)  # 上界
    assert dual.update(torch.zeros(4)) == pytest.approx(0.9)
    negative = BetaDual(beta=0.0, lr=1.0, b_max=10.0)
    assert negative.update(0.0) == 0.0
    frozen = dual.beta
    assert dual.update(float("nan")) == pytest.approx(frozen)
    assert dual.update([float("inf")]) == pytest.approx(frozen)

    state = dual.state_dict()
    restored = BetaDual(beta=0.0, lr=0.0, b_max=0.0, beta_max=0.0)
    restored.load_state_dict(state)
    assert restored.state_dict() == state
    assert math.isclose(restored.beta, state["beta"])


def test_beta_shaped_reward_only_charges_bits_once():
    dual = BetaDual(beta=0.5, lr=0.0, b_max=2.0)
    reward = torch.tensor([[0.0, -1.0]])
    bits = torch.tensor([[4.0, 0.0]])
    torch.testing.assert_close(
        dual.shaped_reward(reward, bits), torch.tensor([[-2.0, -1.0]])
    )
