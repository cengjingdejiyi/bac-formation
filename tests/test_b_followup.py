"""Issue #4: rotated formations and success from local observations."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bacmarl.envs import EnvConfig, FormationCommEnv
from bacmarl.envs.metrics import slot_positions
from bacmarl.envs.scripted import ScriptedController


@pytest.mark.parametrize("backend", ["mock", "mpe2"])
@pytest.mark.parametrize("angle", [np.pi / 6, -0.7, 2.3])
def test_rotated_ring_gets_zero_error_and_completes(backend, angle):
    if backend == "mpe2":
        pytest.importorskip("mpe2.simple_formation_v1")
    cfg = EnvConfig(backend=backend, n_agents=6, success_hold=2)
    center = np.array([0.2, -0.3], dtype=np.float32)
    ring = slot_positions(center, cfg.formation_radius, cfg.n_agents, angle)[
        [2, 0, 4, 5, 1, 3]
    ]
    with FormationCommEnv(cfg) as env:
        env.reset(seed=0)
        if backend == "mpe2":
            world = env._mpe_env.unwrapped.world
            for agent, pos in zip(world.agents, ring):
                agent.state.p_pos = pos.astype(float)
                agent.state.p_vel = np.zeros(2)
            world.landmarks[0].state.p_pos = center.astype(float)
            env._sync_world()
        else:
            env._pos, env._vel, env._center = ring, np.zeros_like(ring), center
        initial = env._obs_step(np.zeros(6), False, False)
        assert initial.info["formation_err"] < 1e-6
        for hold in [1, 2]:
            step = env.step(np.zeros(6, dtype=np.int64))
            assert step.info["formation_err"] < 1e-6
            np.testing.assert_allclose(step.reward, 0, atol=1e-6)
            assert step.info["hold_steps"] == hold
            assert step.info["success"] == (hold == 2)


def test_fixed_heading_is_an_explicit_comparison_mode():
    cfg = EnvConfig(backend="mock", heading_mode="fixed")
    with FormationCommEnv(cfg) as env:
        env.reset(seed=0)
        env._pos = slot_positions([0, 0], 0.5, 6, np.pi / 6)
        step = env.step(np.zeros(6, dtype=np.int64))
        assert step.info["formation_err"] == pytest.approx(0.258819, abs=1e-6)
        assert step.info["heading"] == 0
        assert step.info["hold_steps"] == 0


def test_scripted_actions_use_only_each_agents_observation_row():
    pytest.importorskip("mpe2.simple_formation_v1")
    with FormationCommEnv(EnvConfig(backend="mpe2")) as env:
        step = env.reset(seed=2)
        ctrl = ScriptedController(env.n_agents)
        actions = ctrl.act(step.obs, step.obs_mask)
        assert actions.dtype == np.int64 and actions.shape == (6,)
        for agent in range(6):
            changed = step.obs.copy()
            changed[np.arange(6) != agent] = 1000
            mask = step.obs_mask.copy()
            mask[np.arange(6) != agent] = False
            assert ctrl.act(changed, mask)[agent] == actions[agent]
        masked = step.obs.copy()
        neighbors = masked[:, 4:].reshape(6, 4, 2)
        neighbors[~step.obs_mask] = 1000
        np.testing.assert_array_equal(ctrl.act(masked, step.obs_mask), actions)


def test_formal_alias_and_legacy_configs_preserve_their_metric_semantics():
    from bacmarl.utils.config import load_config

    formal = EnvConfig.from_dict(load_config(["env=formation"])["env"])
    alias = EnvConfig.from_dict(load_config(["env=formation_calibrated"])["env"])
    legacy = EnvConfig.from_dict(load_config(["env=formation_legacy"])["env"])
    assert formal == alias
    assert formal.heading_mode == "dynamic" and formal.success_dist == 0.09
    assert legacy.heading_mode == "fixed" and legacy.success_dist == 0.05


def test_reject_unknown_heading_mode():
    with pytest.raises(ValueError, match="heading_mode"):
        FormationCommEnv(EnvConfig(backend="mock", heading_mode="typo"))


@pytest.mark.parametrize("n_agents", [4, 6, 8])
@pytest.mark.parametrize("seed", [0, 5, 9, 17])
def test_local_scripted_controller_reaches_calibrated_success_threshold(n_agents, seed):
    pytest.importorskip("mpe2.simple_formation_v1")
    cfg = EnvConfig(backend="mpe2", n_agents=n_agents)
    with FormationCommEnv(cfg) as env:
        step = env.reset(seed=seed)
        ctrl = ScriptedController(n_agents, cfg.formation_radius)
        completed = False
        while not step.done:
            step = env.step(ctrl.act(step.obs, step.obs_mask))
            completed |= step.info["success"]
        assert completed, step.info
    assert cfg.success_dist == 0.09 and cfg.success_hold == 10 and cfg.max_steps == 100
