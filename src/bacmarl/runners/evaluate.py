"""评测入口：读 checkpoint，跑确定性回合，按丢包率网格复测（负责人 E）。

用法：
    python -m bacmarl.runners.evaluate --run results/main/formation_6n_bac_L4_K3_B2.0_s0 --episodes 8
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

import numpy as np  # noqa: E402

from bacmarl.envs.formation_comm import EnvConfig, FormationCommEnv  # noqa: E402


def _read_cfg(run_dir: Path) -> dict:
    p = run_dir / 'config.yaml'
    if not p.exists():
        from bacmarl.utils.config import load_config

        return load_config([])
    try:
        import yaml

        return yaml.safe_load(p.read_text(encoding='utf-8'))
    except Exception:
        import json

        return json.loads(p.read_text(encoding='utf-8'))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--episodes', type=int, default=8)
    ap.add_argument('--ckpt', default='ckpt/last.ckpt')
    args = ap.parse_args()

    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    cfg = _read_cfg(run_dir)
    env_cfg = EnvConfig.from_dict(cfg.get('env', {}))
    hidden = int((cfg.get('train') or {}).get('hidden', 64))

    actor = None
    try:
        import torch

        from bacmarl.comm.base import build_comm
        from bacmarl.models.actor import Actor
        from bacmarl.utils.checkpoint import load_checkpoint

        comm = build_comm(cfg['algo']['comm'], hidden, cfg['algo'])
        prot = FormationCommEnv(env_cfg)
        actor = Actor(prot.obs_dim, env_cfg.n_obs, 5, hidden, comm)
        ck = run_dir / args.ckpt
        if ck.exists():
            load_checkpoint(ck, actor=actor)
            actor.eval()
        else:
            print(f'[WARN] 未找到 {ck}，退化为随机策略')
            actor = None
    except Exception as exc:
        print(f'[WARN] 加载模型失败（{exc}），退化为随机策略')

    grid = list(((cfg.get('experiment') or {}).get('robustness') or {}).get('drop_p') or [0.0])
    rows = []
    for drop_p in grid:
        ec = EnvConfig.from_dict(cfg.get('env', {}))
        ec.channel = {'drop_p': [drop_p, drop_p], 'flip_p': [0.0, 0.0]}
        succ, ferr, bits = [], [], []
        for ep in range(args.episodes):
            env = FormationCommEnv(ec)
            step = env.reset(seed=ep)
            done = False
            while not done:
                if actor is not None:
                    import torch

                    with torch.no_grad():
                        out = actor(torch.as_tensor(step.obs)[None],
                                    torch.as_tensor(step.obs_mask)[None],
                                    torch.as_tensor(step.comm_mask)[None],
                                    torch.zeros(1, ec.n_agents, hidden), {'drop_p': drop_p, 'flip_p': 0.0})
                        act = out.action_logits.argmax(-1)[0].numpy()
                    bits.append(float(out.bits.mean()))
                else:
                    act = np.random.default_rng(ep).integers(0, 5, size=ec.n_agents)
                step = env.step(act)
                done = step.done
            succ.append(float(step.info['success']))
            ferr.append(float(step.info['formation_err']))

        rows.append({'drop_p': drop_p,
                     'success_rate': float(np.mean(succ)),
                     'formation_err': float(np.mean(ferr)),
                     'bits': float(np.mean(bits)) if bits else 0.0})

    out_csv = run_dir / 'robust.csv'
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'[DONE] {out_csv}')
    for r in rows:
        print(r)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
