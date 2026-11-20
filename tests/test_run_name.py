"""运行名与消融变体后缀测试（contracts 第 6 节）。"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bacmarl.runners.train import run_name  # noqa: E402
from bacmarl.utils.config import load_config  # noqa: E402


def _name(overrides: list[str]) -> str:
    return run_name(load_config(["env=formation", "algo=bac", *overrides]))


def test_default_name_has_no_variant_suffix():
    assert _name([]) == "formation_6n_bac_L4_K3_B2.0_s0"


def test_aggregate_variant_gets_suffix():
    assert _name(["algo.aggregate=mean"]).endswith("_s0_aggmean")


def test_no_gate_variant_gets_suffix():
    assert _name(["algo.use_gate=false"]).endswith("_s0_nogate")


def test_no_noise_variant_gets_suffix():
    assert _name(["env.channel.drop_p=0", "env.channel.flip_p=0"]).endswith(
        "_s0_nonoise"
    )


def test_single_channel_knob_gets_suffix():
    assert _name(["env.channel.drop_p=0"]).endswith("_s0_drop0")
    assert _name(["env.channel.flip_p=0"]).endswith("_s0_flip0")


def test_variants_never_share_a_directory():
    """#10：mean_agg 与 no_gate 曾写进同一目录并静默覆盖。"""
    names = {
        _name(["algo.aggregate=mean"]),
        _name(["algo.use_gate=false"]),
        _name(["env.channel.drop_p=0", "env.channel.flip_p=0"]),
        _name([]),
    }
    assert len(names) == 4


def test_combined_variants_are_still_distinct():
    assert _name(["algo.aggregate=mean", "algo.use_gate=false"]).endswith(
        "_s0_aggmean-nogate"
    )


@pytest.mark.parametrize("name", ["bac", "ippo", "ic3net", "tarmac"])
def test_run_name_algo_component_matches_config(name):
    cfg = load_config(["env=formation", f"algo={name}"])
    assert f"_{name}_" in run_name(cfg)
