"""契约冒烟：不装 torch / mpe2 也能跑通（环境 → 指标 → 信道 → CSV）。

验收：退出码 0，所有 [PASS] 成立，results/_smoke/metrics.csv 生成。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np  # noqa: E402

from bacmarl.comm.np_channel import apply_noise_np, sample_noise_np  # noqa: E402
from bacmarl.envs.formation_comm import EnvConfig, FormationCommEnv  # noqa: E402
from bacmarl.envs.scripted import ScriptedController  # noqa: E402
from bacmarl.envs.vec_env import VecFormationEnv  # noqa: E402
from bacmarl.utils.logger import RunLogger  # noqa: E402
from bacmarl.utils.seeding import set_seed  # noqa: E402

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = '') -> None:
    print(('[PASS] ' if ok else '[FAIL] ') + name + (f'  {detail}' if detail else ''))
    if not ok:
        FAILS.append(name)


def main() -> int:
    set_seed(0)
    cfg = EnvConfig.from_dict({
        'n_agents': 6, 'max_steps': 40, 'n_obs': 4, 'backend': 'mock',
        'channel': {'drop_p': [0.0, 0.3], 'flip_p': [0.0, 0.1]},
    })
    n, n_obs = cfg.n_agents, cfg.n_obs

    env = FormationCommEnv(cfg)
    step = env.reset(seed=0)
    check('obs 形状 [N, 4+2n_obs]', step.obs.shape == (n, 4 + 2 * n_obs), str(step.obs.shape))
    check('obs float32', step.obs.dtype == np.float32, str(step.obs.dtype))
    check('obs_mask [N, n_obs] bool', step.obs_mask.shape == (n, n_obs) and step.obs_mask.dtype == bool)
    check('state [4N+2]', step.state.shape == (4 * n + 2,), str(step.state.shape))
    check('comm_mask [N, N] 对角 False', step.comm_mask.shape == (n, n) and not np.any(np.diag(step.comm_mask)))
    check('reward [N] float32', step.reward.shape == (n,) and step.reward.dtype == np.float32)
    check('info 含 channel', 'channel' in step.info)
    check('动作合法 5 选 1', set(np.unique(env.step(np.zeros(n, dtype=np.int64)).obs.dtype.name)) == {'float32'})

    # 信道：整条丢弃 + 比特翻转（numpy 路径，无需 torch）
    rng = np.random.default_rng(0)
    L = 4
    noise = sample_noise_np(rng, step.comm_mask[None], L, drop_p=0.5, flip_p=0.3)
    check('noise.drop [B,N,N]', noise['drop'].shape == (1, n, n))
    check('noise.flip [B,N,N,L]', noise['flip'].shape == (1, n, n, L))
    msgs = rng.choice([-1.0, 1.0], size=(1, n, L)).astype(np.float32)
    edge = apply_noise_np(msgs, step.comm_mask[None], noise)
    check('apply_noise [B,N,N,L]', edge.shape == (1, n, n, L), str(edge.shape))
    check('被丢弃的边全为 0', bool(np.all(edge[noise['drop']] == 0)))
    check('消息为 ±1', bool(np.all(np.abs(msgs) == 1.0)))

    # 并行环境 + 脚本控制器
    vec = VecFormationEnv(4, cfg, seed=0)
    batch = vec.reset()
    check('vec obs [B,N,D]', batch['obs'].shape == (4, n, 4 + 2 * n_obs), str(batch['obs'].shape))
    check('vec step 返回 done [B]', batch['done'].shape == (4,) and batch['done'].dtype == bool)
    ctrl = ScriptedController(n, cfg.formation_radius)
    logger = RunLogger(ROOT / 'results' / '_smoke', use_tb=False)
    ret = 0.0
    for t in range(30):
        acts = np.stack([ctrl.act(batch['obs'][k], batch['obs_mask'][k]) for k in range(4)])
        batch = vec.step(acts)
        ret += float(batch['reward'].mean())
    logger.log({'smoke/return': ret, 'smoke/last_ferr': float(batch['info_raw'][0].info['formation_err'])},
               30 * 4)
    logger.close()
    check('metrics.csv 已生成', (ROOT / 'results' / '_smoke' / 'metrics.csv').exists())
    check('回报有限', bool(np.isfinite(ret)), f'{ret:.3f}')

    print('\n' + ('SMOKE OK' if not FAILS else 'SMOKE FAILED: ' + ', '.join(FAILS)))
    return 0 if not FAILS else 1


if __name__ == '__main__':
    raise SystemExit(main())
