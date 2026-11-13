"""环境接口形状测试（contracts 第 2 节）。"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from bacmarl.envs.formation_comm import EnvConfig, FormationCommEnv  # noqa: E402


def _env(**kw):
    d = {'n_agents': 5, 'max_steps': 10, 'n_obs': 3, 'backend': 'mock'}
    d.update(kw)
    return FormationCommEnv(EnvConfig.from_dict(d))


def test_shapes_and_dtypes():
    env = _env()
    step = env.reset(seed=0)
    n, n_obs = env.cfg.n_agents, env.cfg.n_obs
    assert step.obs.shape == (n, 4 + 2 * n_obs) and step.obs.dtype == np.float32
    assert step.obs_mask.shape == (n, n_obs) and step.obs_mask.dtype == bool
    assert step.state.shape == (4 * n + 2,) and step.state.dtype == np.float32
    assert step.comm_mask.shape == (n, n) and step.comm_mask.dtype == bool
    assert step.reward.shape == (n,) and step.reward.dtype == np.float32
    assert isinstance(step.done, bool)
    assert not np.any(np.diag(step.comm_mask))


def test_done_after_max_steps():
    env = _env()
    step = env.reset(seed=1)
    for _ in range(env.cfg.max_steps):
        step = env.step(np.zeros(env.cfg.n_agents, dtype=np.int64))
    assert step.done


def test_comm_mask_respects_radius_and_k():
    env = _env(n_agents=6, comm_radius=0.4, recv_k=2)
    step = env.reset(seed=2)
    pos = step.info['pos']
    d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    for i in range(env.cfg.n_agents):
        assert step.comm_mask[i].sum() <= 2
        for j in np.where(step.comm_mask[i])[0]:
            assert d[i, j] <= 0.4 + 1e-5


def test_actions_are_discrete_5():
    env = _env()
    step = env.reset(seed=3)
    out = env.step(np.array([0, 1, 2, 3, 4]))
    assert out.obs.shape == step.obs.shape
