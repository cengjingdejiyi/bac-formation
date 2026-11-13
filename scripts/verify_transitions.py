"""直接验证训练使用的 collect_rollout，不重演历史错误的伪训练循环。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bacmarl.algos.buffer import RolloutBuffer  # noqa: E402
from bacmarl.comm.base import build_comm  # noqa: E402
from bacmarl.envs import EnvConfig, VecFormationEnv  # noqa: E402
from bacmarl.models.actor import Actor  # noqa: E402
from bacmarl.models.critic import Critic  # noqa: E402
from bacmarl.runners.rollout import collect_rollout  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["mock", "mpe2"], default="mpe2")
    args = parser.parse_args()
    torch.manual_seed(0)
    cfg = EnvConfig(backend=args.backend, n_agents=3, max_steps=3)
    with VecFormationEnv(1, cfg, seed=0) as env:
        actor = Actor(env.obs_dim, cfg.n_obs, 5, 8, build_comm("none", 8))
        critic = Critic("central", env.state_dim, env.obs_dim, cfg.n_agents, 8)
        buf = RolloutBuffer(6, 1, 3, env.obs_dim, cfg.n_obs, env.state_dim, 8, 0)
        current = [env.reset()]
        trace = []
        step = env.step

        def traced_step(action: np.ndarray) -> dict:
            obs = current[0]["obs"].copy()
            nxt = step(action)
            trace.append((obs, action.copy(), nxt["reward"].copy(), nxt["done"].copy()))
            current[0] = nxt
            return nxt

        env.step = traced_step
        rollout = collect_rollout(
            env, actor, critic, buf, current[0], actor.init_hidden(1, 3)
        )
        print("t | cached_reward | step_reward | done | pre_action_obs_matches")
        for t, (obs, action, reward, done) in enumerate(trace):
            np.testing.assert_array_equal(buf.obs[t], obs)
            np.testing.assert_array_equal(buf.action[t], action)
            np.testing.assert_array_equal(buf.reward_team[t], reward)
            np.testing.assert_array_equal(buf.done[t], done)
            print(
                f"{t} | {buf.reward_team[t].mean():+.6f} | {reward.mean():+.6f} | {bool(done[0])} | True"
            )
        assert buf.reward_team[0].mean() != 0.0
        assert [ep["len"] for ep in rollout.episodes] == [3, 3]
        print(
            "TRANSITIONS OK: both terminal rewards retained; no synthetic reset reward cached"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
