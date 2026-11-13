"""配置加载与覆盖测试（contracts 第 5 节）。"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bacmarl.utils.config import load_config  # noqa: E402
from bacmarl.envs import EnvConfig  # noqa: E402


def test_default_groups_merge():
    cfg = load_config([])
    assert cfg["env"]["n_agents"] >= 1
    assert cfg["algo"]["comm"] in ("none", "ic3net", "tarmac", "bac")
    assert cfg["train"]["hidden"] >= 8


def test_cli_override():
    cfg = load_config(
        ["env=formation", "algo=ippo", "env.n_agents=9", "algo.msg_bits=8", "seed=3"]
    )
    assert cfg["algo"]["comm"] == "none"
    assert cfg["env"]["n_agents"] == 9
    assert cfg["algo"]["msg_bits"] == 8
    assert cfg["seed"] == 3


def test_formal_environment_matches_dataclass_and_compatibility_alias():
    cfg = load_config(["env=formation"])["env"]
    alias = load_config(["env=formation_calibrated"])["env"]
    defaults = EnvConfig()
    assert cfg == alias
    assert cfg["backend"] == defaults.backend == "mpe2"
    assert cfg["obs_radius"] == defaults.obs_radius == 0.8
    assert cfg["comm_radius"] == defaults.comm_radius == 0.8
    assert cfg["collision_coef"] == defaults.collision_coef == 1.0


def test_legacy_environment_only_overrides_historical_parameters():
    formal = load_config(["env=formation"])["env"]
    legacy = load_config(["env=formation_legacy"])["env"]
    assert legacy == {
        **formal,
        "backend": "auto",
        "obs_radius": 0.5,
        "comm_radius": 0.4,
    }


def test_inheritance_cycle_is_rejected(tmp_path):
    (tmp_path / "config.yaml").write_text("defaults: {env: first}\n")
    (tmp_path / "env").mkdir()
    (tmp_path / "env" / "first.yaml").write_text("extends: second\n")
    (tmp_path / "env" / "second.yaml").write_text("extends: first\n")
    with pytest.raises(ValueError, match="cyclic"):
        load_config([], config_dir=tmp_path)
