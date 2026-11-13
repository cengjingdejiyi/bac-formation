"""环境语义回归：真实物理、局部观测、终止转移与可复现性。"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bacmarl.envs import EnvConfig, FormationCommEnv, VecFormationEnv
from bacmarl.envs import formation_comm, metrics


@pytest.fixture(params=["mock", "mpe2"])
def backend(request):
    if request.param == "mpe2":
        pytest.importorskip("mpe2.simple_formation_v1")
    return request.param


def set_world(env, pos, vel=None, center=(0.0, 0.0)):
    """为可解释的几何/奖励断言设置确定状态，pos/vel [N,2]。"""
    pos = np.asarray(pos, dtype=np.float32)
    vel = np.zeros_like(pos) if vel is None else np.asarray(vel, dtype=np.float32)
    if env.backend == "mpe2":
        world = env._mpe_env.unwrapped.world
        for agent, p, v in zip(world.agents, pos, vel):
            agent.state.p_pos = p.astype(float)
            agent.state.p_vel = v.astype(float)
        world.landmarks[0].state.p_pos = np.asarray(center, dtype=float)
        env._sync_world()
    else:
        env._pos, env._vel = pos.copy(), vel.copy()
        env._center = np.asarray(center, dtype=np.float32)


def test_local_observations_and_directed_nearest_links(backend):
    cfg = EnvConfig(
        backend=backend, n_agents=4, n_obs=4, recv_k=1, obs_radius=0.5, comm_radius=0.5
    )
    with FormationCommEnv(cfg) as env:
        env.reset(seed=0)
        set_world(
            env,
            [[0, 0], [0.1, 0], [0.4, 0], [2, 0]],
            vel=[[0.2, -0.3]] * 4,
            center=[1, 1],
        )
        st = env._obs_step(np.zeros(4), False, False)
        np.testing.assert_allclose(st.obs[:, :2], [[0.2, -0.3]] * 4)
        np.testing.assert_allclose(st.obs[:, 2:4], [1, 1] - env._pos)
        np.testing.assert_allclose(st.obs[2, 4:8], [-0.3, 0, -0.4, 0])
        np.testing.assert_array_equal(st.obs_mask.sum(1), [2, 2, 2, 0])
        assert np.all(st.obs[:, 8:] == 0)
        assert not st.obs_mask[3].any() and np.all(st.obs[3, 4:] == 0)
        np.testing.assert_array_equal(
            st.comm_mask, [[0, 1, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]]
        )
        np.testing.assert_array_equal(
            st.state, np.r_[env._pos.ravel(), env._vel.ravel(), [1, 1]]
        )


def test_configured_radius_heading_and_team_reward(backend):
    cfg = EnvConfig(
        backend=backend,
        n_agents=4,
        formation_radius=0.8,
        heading=0.3,
        safe_dist=2.0,
        collision_coef=2.5,
    )
    with FormationCommEnv(cfg) as env:
        env.reset(seed=0)
        center = np.array([0.3, -0.4])
        set_world(env, metrics.slot_positions(center, 0.8, 4, 0.3), center=center)
        st = env.step(np.zeros(4, dtype=np.int64))
        assert st.info["formation_err"] < 1e-6
        assert st.info["collisions"] == 6  # 六个无序机对，各算一次。
        np.testing.assert_allclose(st.reward, [-15.0] * 4, atol=1e-6)
        assert st.info["collisions_ep"] == 6


@pytest.mark.parametrize("early_end", [True, False])
def test_success_requires_consecutive_hold_and_obeys_early_end(backend, early_end):
    cfg = EnvConfig(
        backend=backend,
        n_agents=4,
        success_hold=3,
        max_steps=8,
        end_on_success=early_end,
    )
    with FormationCommEnv(cfg) as env:
        env.reset(seed=0)
        slots = metrics.slot_positions([0, 0], 0.5, 4)
        set_world(env, slots)
        assert env.step(np.zeros(4, dtype=np.int64)).info["hold_steps"] == 1
        set_world(env, slots + [0.2, 0])
        failed = env.step(np.zeros(4, dtype=np.int64))
        assert failed.info["hold_steps"] == 0 and not failed.info["success"]
        set_world(env, slots)
        for hold in range(1, 4):
            st = env.step(np.zeros(4, dtype=np.int64))
            assert st.info["hold_steps"] == hold
            assert st.info["success"] == (hold == 3)
            assert st.done == (early_end and hold == 3)
        if not early_end:
            set_world(env, slots + [0.2, 0])
            assert not env.step(np.zeros(4, dtype=np.int64)).info["success"]


def test_speed_cap_uses_vector_norm(backend):
    cfg = EnvConfig(backend=backend, n_agents=2, max_speed=0.6, max_steps=80)
    with FormationCommEnv(cfg) as env:
        env.reset(seed=7)
        set_world(env, [[-1, -1], [2, 2]])
        for t in range(80):
            st = env.step(np.full(2, 2 if t % 2 else 4, dtype=np.int64))
            assert np.all(np.linalg.norm(st.obs[:, :2], axis=1) <= 0.6 + 1e-6)
        assert st.done
        with pytest.raises(RuntimeError, match="reset"):
            env.step(np.zeros(2, dtype=np.int64))


def test_empty_neighbor_slots_and_channel_lifetime(backend):
    cfg = EnvConfig(
        backend=backend,
        n_agents=1,
        n_obs=0,
        recv_k=0,
        channel={"drop_p": [0.2, 0.3], "flip_p": 0.1},
    )
    with FormationCommEnv(cfg) as env:
        initial = env.reset(seed=3)
        st = env.step(np.array([0]))
        assert st.obs.shape == (1, 4) and st.obs_mask.shape == (1, 0)
        assert not st.comm_mask.any()
        assert st.info["channel"] == initial.info["channel"]
        assert 0.2 <= st.info["channel"]["drop_p"] <= 0.3
        assert st.info["channel"]["flip_p"] == 0.1


def test_seed_reproduces_full_run_without_repeated_episodes(backend):
    cfg = EnvConfig(backend=backend, n_agents=4, max_steps=1)
    with (
        VecFormationEnv(2, cfg, seed=9) as left,
        VecFormationEnv(2, cfg, seed=9) as right,
    ):
        initial = left.reset()
        np.testing.assert_array_equal(initial["state"], right.reset()["state"])
        assert not np.array_equal(initial["state"][0], initial["state"][1])
        states = [initial["state"].copy()]
        for _ in range(3):
            a = left.step(np.zeros((2, 4), dtype=np.int64))
            b = right.step(np.zeros((2, 4), dtype=np.int64))
            assert a["done"].all()
            np.testing.assert_array_equal(a["state"], b["state"])
            np.testing.assert_array_equal(a["reward"], b["reward"])
            np.testing.assert_array_equal(
                a["channel"]["drop_p"], b["channel"]["drop_p"]
            )
            assert not np.array_equal(a["state"], states[-1])
            states.append(a["state"].copy())


def test_vector_autoreset_preserves_terminal_reward_and_statistics(backend):
    cfg = EnvConfig(
        backend=backend,
        n_agents=4,
        max_steps=1,
        success_hold=1,
        channel={"drop_p": [0.0, 1.0]},
    )
    with VecFormationEnv(1, cfg, seed=0) as vec:
        vec.reset()
        set_world(vec.envs[0], metrics.slot_positions([0, 0], 0.5, 4))
        batch = vec.step(np.zeros((1, 4), dtype=np.int64))
        initial = batch["info_raw"][0]
        terminal = initial.info["terminal_step"]
        assert batch["done"].tolist() == [True]
        assert batch["ep"][0]["len"] == 1 and batch["ep"][0]["success"]
        assert terminal.done and terminal.info["hold_steps"] == 1
        assert initial.info["t"] == 0
        np.testing.assert_array_equal(batch["reward"][0], terminal.reward)
        assert batch["ep"][0]["ret"] == pytest.approx(float(terminal.reward.mean()))
        np.testing.assert_array_equal(batch["obs"][0], initial.obs)
        assert not np.array_equal(initial.state, terminal.state)
        assert batch["channel"]["drop_p"][0] == pytest.approx(
            initial.info["channel"]["drop_p"]
        )
        assert terminal.info["channel"] != initial.info["channel"]


def test_terminal_reward_is_not_zeroed(backend):
    cfg = EnvConfig(
        backend=backend, n_agents=2, max_steps=1, safe_dist=1.0, collision_coef=3.0
    )
    with VecFormationEnv(1, cfg) as vec:
        vec.reset()
        set_world(vec.envs[0], [[0, 0], [0.3, 0]])
        batch = vec.step(np.zeros((1, 2), dtype=np.int64))
        assert np.all(batch["reward"] < -3.0)
        assert batch["ep"][0]["collisions"] == 1


def test_invalid_batch_does_not_partially_advance(backend):
    with VecFormationEnv(2, EnvConfig(backend=backend, n_agents=2)) as vec:
        initial = vec.reset()
        with pytest.raises(ValueError, match="actions"):
            vec.step(np.array([[0, 0], [0, 5]]))
        assert all(env._t == 0 for env in vec.envs)
        for i, env in enumerate(vec.envs):
            np.testing.assert_array_equal(env._pos, initial["info_raw"][i].info["pos"])


@pytest.mark.parametrize("n", [4, 6, 8])
def test_mpe2_is_real_physics_in_stable_agent_order(n):
    sf = pytest.importorskip("mpe2.simple_formation_v1")
    raw = sf.parallel_env(
        N=n, max_cycles=3, continuous_actions=False, terminate_on_success=False
    )
    try:
        with FormationCommEnv(
            EnvConfig(backend="mpe2", n_agents=n, max_steps=3)
        ) as env:
            st = env.reset(seed=42)
            raw.reset(seed=42)
            for agent in raw.unwrapped.world.agents:
                agent.max_speed = env.cfg.max_speed
            np.testing.assert_allclose(
                st.info["center"], raw.unwrapped.world.landmarks[0].state.p_pos
            )
            for t in range(3):
                actions = (np.arange(n, dtype=np.int64) + t) % 5
                raw.step(dict(zip(raw.possible_agents, actions.tolist())))
                st = env.step(actions)
                np.testing.assert_allclose(
                    st.info["pos"],
                    [a.state.p_pos for a in raw.unwrapped.world.agents],
                    atol=1e-7,
                )
                np.testing.assert_allclose(
                    st.obs[:, :2],
                    [a.state.p_vel for a in raw.unwrapped.world.agents],
                    atol=1e-7,
                )
                assert st.obs.shape == (n, 12) and st.obs.dtype == np.float32
                assert st.state.shape == (4 * n + 2,) and st.state.dtype == np.float32
                assert st.reward.dtype == np.float32 and st.reward.shape == (n,)
                assert st.done == (t == 2)
                assert st.info["backend"] == "mpe2"
    finally:
        raw.close()


def test_auto_selects_installed_mpe2():
    pytest.importorskip("mpe2.simple_formation_v1")
    with FormationCommEnv(EnvConfig()) as env:
        assert env.backend == "mpe2" and env._mpe_env is not None


@pytest.mark.parametrize("backend", ["auto", "mpe2"])
def test_missing_mpe2_fallback_is_explicit(monkeypatch, backend):
    def missing(_name):
        raise ModuleNotFoundError("missing mpe2", name="mpe2")

    monkeypatch.setattr(formation_comm.importlib, "import_module", missing)
    if backend == "auto":
        with pytest.warns(RuntimeWarning, match="mock physics"):
            with FormationCommEnv(EnvConfig(backend=backend)) as env:
                assert env.backend == "mock"
    else:
        with pytest.raises(RuntimeError, match="mpe2"):
            FormationCommEnv(EnvConfig(backend=backend))


@pytest.mark.parametrize("missing_name", ["pygame", "mpe2.simple_formation_v1"])
def test_auto_does_not_hide_broken_installation(monkeypatch, missing_name):
    def broken(_name):
        raise ModuleNotFoundError("broken install", name=missing_name)

    monkeypatch.setattr(formation_comm.importlib, "import_module", broken)
    with pytest.raises(ModuleNotFoundError):
        FormationCommEnv(EnvConfig())


@pytest.mark.parametrize(
    "action",
    [
        np.zeros((4, 1), dtype=int),
        np.zeros(4),
        np.array([-1, 0, 0, 0]),
        np.array([0, 5, 0, 0]),
    ],
)
def test_reject_invalid_action_before_advancing(backend, action):
    with FormationCommEnv(EnvConfig(backend=backend, n_agents=4)) as env:
        initial = env.reset(seed=0)
        with pytest.raises(ValueError, match="action"):
            env.step(action)
        np.testing.assert_array_equal(env._pos, initial.info["pos"])
        assert env._t == 0


@pytest.mark.parametrize(
    "kw",
    [
        {"n_agents": 0},
        {"max_steps": 0},
        {"n_obs": -1},
        {"recv_k": -1},
        {"max_speed": 0},
        {"success_hold": 0},
        {"comm_radius": -0.1},
        {"heading": float("nan")},
        {"channel": {"drop_p": [0.4, 0.1]}},
        {"channel": {"flip_p": 1.1}},
        {"channel": {"drop_p": [0.1]}},
    ],
)
def test_reject_invalid_config(kw):
    with pytest.raises(ValueError):
        FormationCommEnv(EnvConfig(backend="mock", **kw))


def test_unknown_backend_and_unimplemented_scene_are_not_silent():
    with pytest.raises(ValueError, match="backend"):
        FormationCommEnv(EnvConfig(backend="typo"))
    with pytest.raises(NotImplementedError, match="spread"):
        FormationCommEnv(EnvConfig(backend="mock", scene="spread"))


def test_close_releases_mpe2_once(monkeypatch):
    pytest.importorskip("mpe2.simple_formation_v1")
    env = FormationCommEnv(EnvConfig(backend="mpe2"))
    calls = []
    original = env._mpe_env.close

    def close():
        calls.append(True)
        original()

    monkeypatch.setattr(env._mpe_env, "close", close)
    with env:
        env.reset(seed=0)
    env.close()
    assert calls == [True]
    with pytest.raises(RuntimeError, match="closed"):
        env.reset()


def test_slot_matching_is_unique_and_permutation_invariant(monkeypatch):
    slots = metrics.slot_positions([0.2, -0.3], 0.7, 6, heading=0.4)
    pos = slots[[3, 1, 5, 0, 2, 4]]
    for scipy_available in [True, False]:
        if scipy_available and not metrics._HAS_SCIPY:
            continue
        monkeypatch.setattr(metrics, "_HAS_SCIPY", scipy_available)
        perm, dist = metrics.match_slots(pos, slots)
        assert len(set(perm)) == 6 and np.all(dist == 0)
        assert metrics.formation_error(pos, [0.2, -0.3], 0.7, heading=0.4) == 0


def test_safe_distance_boundary_and_pair_count():
    pos = np.array([[0, 0], [0.2, 0], [0.4, 0]], dtype=np.float32)
    assert metrics.count_collisions(pos, 0.2) == 0
    assert metrics.count_collisions(pos, 0.21) == 2
