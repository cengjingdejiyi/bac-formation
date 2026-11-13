"""统一训练入口。

用法：
    python -m bacmarl.runners.train env=formation algo=bac env.n_agents=6 algo.msg_bits=4 seed=0
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

import numpy as np  # noqa: E402

from bacmarl.algos.buffer import RolloutBuffer  # noqa: E402
from bacmarl.algos.dual import BetaDual  # noqa: E402
from bacmarl.algos.mappo import ppo_update  # noqa: E402
from bacmarl.comm.base import build_comm  # noqa: E402
from bacmarl.envs.formation_comm import EnvConfig  # noqa: E402
from bacmarl.envs.vec_env import VecFormationEnv  # noqa: E402
from bacmarl.utils.config import load_config  # noqa: E402
from bacmarl.utils.logger import RunLogger  # noqa: E402
from bacmarl.utils.seeding import set_seed  # noqa: E402


def run_name(cfg: dict) -> str:
    """运行名规范见 docs/contracts.md 第 6 节。"""
    e, a = cfg['env'], cfg['algo']
    return (f"{e.get('scene', 'formation')}_{e['n_agents']}n_{a.get('name', a['comm'])}"
            f"_L{a.get('msg_bits', 0)}_K{e.get('recv_k', 1)}_B{a.get('budget_bits', 0)}"
            f"_s{cfg['seed']}")


def _dump_yaml(cfg: dict, path: Path) -> None:
    try:
        import yaml

        path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding='utf-8')
    except Exception:
        import json

        path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf-8')


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cfg = load_config(argv)
    try:
        import torch
        from torch.distributions import Categorical
    except Exception:
        print('[ERR] 训练需要 PyTorch：pip install -r requirements.txt', file=sys.stderr)
        return 1

    set_seed(int(cfg['seed']))
    tr = cfg['train']
    env_cfg = EnvConfig.from_dict(cfg['env'])
    vec = VecFormationEnv(int(tr['n_envs']), env_cfg, seed=int(cfg['seed']))
    b, n = int(tr['n_envs']), env_cfg.n_agents
    hidden = int(tr['hidden'])

    comm = build_comm(cfg['algo']['comm'], hidden, cfg['algo'])
    # 只有二值消息（BAC）需要翻转掩码
    flip_bits = int(cfg['algo'].get('msg_bits', 0)) if cfg['algo']['comm'] == 'bac' else 0

    from bacmarl.models.actor import Actor
    from bacmarl.models.critic import Critic

    actor = Actor(vec.obs_dim, env_cfg.n_obs, 5, hidden, comm)
    critic = Critic(cfg['algo'].get('critic', 'central'), vec.state_dim, vec.obs_dim, n, hidden)
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=float(tr['lr']))

    dual = BetaDual(beta=float(cfg['algo'].get('beta_init', 0.0)),
                    lr=float(cfg['algo'].get('beta_lr', 0.0)),
                    b_max=float(cfg['algo'].get('budget_bits', 0.0)))

    exp_name = (cfg.get('experiment') or {}).get('name', 'main')
    run_dir = ROOT / 'results' / exp_name / run_name(cfg)
    logger = RunLogger(run_dir)
    _dump_yaml(cfg, run_dir / 'config.yaml')

    steps, total = 0, int(tr['total_env_steps'])
    print(f'[INFO] run_dir={run_dir}\n[INFO] comm={cfg["algo"]["comm"]} N={n} B={b} hidden={hidden}')

    while steps < total:
        batch = vec.reset()
        h = actor.init_hidden(b, n)
        buf = RolloutBuffer(int(tr['rollout_len']), b, n, vec.obs_dim, env_cfg.n_obs,
                            vec.state_dim, hidden, flip_bits)

        for _ in range(int(tr['rollout_len'])):
            obs_t = torch.as_tensor(batch['obs'])
            mask_t = torch.as_tensor(batch['obs_mask'])
            cmask_t = torch.as_tensor(batch['comm_mask'])
            channel = {k: float(np.mean(v)) for k, v in batch['channel'].items()}
            with torch.no_grad():
                out = actor(obs_t, mask_t, cmask_t, h, channel)
                action = Categorical(logits=out.action_logits).sample()
                logp = actor.log_prob(out, action)
                value = critic(state=torch.as_tensor(batch['state']), obs=obs_t)

            noise_drop = (out.noise.drop.numpy() if out.noise is not None
                          else np.zeros((b, n, n), dtype=bool))
            noise_flip = (out.noise.flip.numpy() if (out.noise is not None and flip_bits > 0)
                          else np.zeros((b, n, n, flip_bits), dtype=bool))
            buf.add({
                'obs': batch['obs'], 'obs_mask': batch['obs_mask'], 'state': batch['state'],
                'comm_mask': batch['comm_mask'], 'h_prev': h.numpy(),
                'action': action.numpy(), 'gate': out.gate.numpy(),
                'logp_old': logp.numpy(), 'value': value.numpy(),
                'reward_team': batch['reward'], 'bits': out.bits.numpy(),
                'done': batch['done'], 'noise_drop': noise_drop, 'noise_flip': noise_flip,
            })
            h = out.h.detach()
            batch = vec.step(action.numpy())
            steps += b

        probe = torch.as_tensor(batch['obs'])
        last_v = critic(state=torch.as_tensor(batch['state']), obs=probe).detach().numpy()
        shaped_all = buf.reward_team - dual.beta * buf.bits
        buf.compute_returns(shaped_all, last_v, float(tr['gamma']), float(tr['gae_lambda']))

        raw = buf.batched(buf.flat_indices())
        tb = {}
        for k, v in raw.items():
            t = torch.as_tensor(v)
            tb[k] = t if t.dtype == torch.bool else t.float()
        logs = ppo_update(actor, critic, opt, buf, tr, tb)
        beta = dual.update(float(np.mean(buf.bits.sum(axis=-1)) / max(n, 1)))
        logger.log({'train/return': float(np.mean(buf.reward_team)),
                    'comm/bits': float(np.mean(buf.bits)),
                    'comm/speak_rate': float(np.mean(buf.gate)),
                    'comm/beta': beta, **logs}, steps)

    logger.close()
    try:
        from bacmarl.utils.checkpoint import save_checkpoint

        save_checkpoint(run_dir / 'ckpt' / 'last.ckpt', actor, critic, meta={'cfg': cfg, 'steps': steps})
    except Exception as exc:  # pragma: no cover
        print(f'[WARN] checkpoint 保存失败：{exc}')
    print(f'[DONE] steps={steps} beta={dual.beta:.3f} csv={logger.csv_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
