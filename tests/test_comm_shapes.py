"""四种通信模块的形状与比特数测试（contracts 第 3 节）。无 torch 时整体跳过。"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

torch = pytest.importorskip('torch')

from bacmarl.comm.base import build_comm  # noqa: E402
from bacmarl.comm.channel import sample_noise  # noqa: E402


def _inputs(b=2, n=6, h=32):
    h_t = torch.randn(b, n, h)
    mask = torch.zeros(b, n, n, dtype=torch.bool)
    for i in range(n):
        mask[:, i, (i + 1) % n] = True
        mask[:, i, (i + 2) % n] = True
    gate = torch.randint(0, 2, (b, n)).float()
    return h_t, mask, gate


@pytest.mark.parametrize('name', ['none', 'ic3net', 'tarmac', 'bac'])
def test_shapes(name):
    h, mask, gate = _inputs()
    comm = build_comm(name, 32, {'msg_dim': 8, 'msg_bits': 4, 'recv_k': 3})
    out = comm(h, mask, gate, {'drop_p': 0.2, 'flip_p': 0.1})
    assert out.c.shape == (2, 6, 32)
    assert out.bits.shape == (2, 6)
    assert out.attn.shape == (2, 6, 6)
    assert bool((out.bits >= 0).all())


def test_bac_bits_equal_gate_times_L():
    h, mask, gate = _inputs()
    comm = build_comm('bac', 32, {'msg_bits': 4})
    out = comm(h, mask, gate, {'drop_p': 0.0, 'flip_p': 0.0})
    assert bool(torch.allclose(out.bits, gate * 4.0))


def test_none_bits_zero():
    h, mask, gate = _inputs()
    out = build_comm('none', 32, {})(h, mask, gate, {'drop_p': 0.0, 'flip_p': 0.0})
    assert float(out.bits.sum()) == 0.0


def test_noise_reuse_is_deterministic():
    h, mask, gate = _inputs()
    comm = build_comm('bac', 32, {'msg_bits': 4})
    noise = sample_noise(mask, 4, 0.3, 0.2)
    a = comm(h, mask, gate, {'drop_p': 0.3, 'flip_p': 0.2}, noise=noise)
    b = comm(h, mask, gate, {'drop_p': 0.3, 'flip_p': 0.2}, noise=noise)
    assert bool(torch.allclose(a.c, b.c))
