"""演示：脚本控制器跑一轮，导出轨迹（负责人 F）。无 matplotlib 也能跑。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np  # noqa: E402

from bacmarl.envs.formation_comm import EnvConfig, FormationCommEnv  # noqa: E402
from bacmarl.envs.scripted import ScriptedController  # noqa: E402


def main() -> int:
    cfg = EnvConfig.from_dict({'n_agents': 6, 'max_steps': 80, 'backend': 'mock'})
    env = FormationCommEnv(cfg)
    step = env.reset(seed=1)
    ctrl = ScriptedController(cfg.n_agents, cfg.formation_radius)
    frames, errs = [], []
    done = False
    while not done:
        frames.append(step.info['pos'].copy())
        errs.append(step.info['formation_err'])
        step = env.step(ctrl.act(step.obs, step.obs_mask))
        done = step.done
    out = ROOT / 'results' / 'demo'
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / 'trajectory.npy', np.stack(frames))
    print(f'[DONE] {len(frames)} 帧，末步编队误差 {errs[-1]:.3f} → {out / "trajectory.npy"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
