"""脚本化转移验证奖励对齐、自动 reset、RNN 与 GAE 的回合边界。"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.nn.functional import one_hot

from bacmarl.algos.buffer import RolloutBuffer
from bacmarl.algos.mappo import gae
from bacmarl.comm.base import build_comm
from bacmarl.comm.channel import ChannelNoise
from bacmarl.envs import EnvConfig, VecFormationEnv
from bacmarl.models.actor import Actor
from bacmarl.models.critic import Critic
from bacmarl.runners.rollout import collect_rollout


class ScriptedVec:
    """两个不同长度回合，奖励随真实步号/动作变化，自动重置后观测跳变。"""

    def __init__(self, lengths=(2, 3)):
        self.lengths = np.asarray(lengths)
        self.b = len(lengths)
        self.ticks = np.zeros(self.b, dtype=int)
        self.episode = np.zeros(self.b, dtype=int)
        self.calls, self.history, self.reset_calls = 0, [], 0

    def batch(self, reward=None, done=None, ep=None):
        obs = (1 + 10 * np.arange(self.b) + 100 * self.episode + self.ticks).astype(
            np.float32
        )
        return {
            "obs": obs[:, None, None],
            "state": obs[:, None],
            "obs_mask": np.zeros((self.b, 1, 0), dtype=bool),
            "comm_mask": np.zeros((self.b, 1, 1), dtype=bool),
            "channel": {
                "drop_p": np.linspace(0, 1, self.b, dtype=np.float32),
                "flip_p": np.zeros(self.b),
            },
            "reward": np.zeros((self.b, 1), np.float32) if reward is None else reward,
            "done": np.zeros(self.b, dtype=bool) if done is None else done,
            "ep": [None] * self.b if ep is None else ep,
        }

    def reset(self):
        self.reset_calls += 1
        return self.batch()

    def step(self, actions):
        before = self.batch()
        self.calls += 1
        self.ticks += 1
        reward = (10 * self.calls + actions).astype(np.float32)
        done = self.ticks == self.lengths
        terminal = self.batch()["obs"].copy()
        self.history.append(
            {
                "obs": before["obs"].copy(),
                "action": actions.copy(),
                "reward": reward.copy(),
                "done": done.copy(),
                "terminal": terminal,
            }
        )
        episodes = [
            {"len": int(self.ticks[i])} if done[i] else None for i in range(self.b)
        ]
        self.episode += done
        self.ticks[done] = 0
        return self.batch(reward, done, episodes)


class ScriptedActor:
    def __init__(self):
        self.channels = []

    def __call__(self, obs, obs_mask, comm_mask, hidden, channel):
        action = obs[..., 0].long() % 5
        self.channels.append(channel)
        gate = torch.ones_like(action, dtype=torch.float32)
        return SimpleNamespace(
            action_logits=one_hot(action, 5).float() * 1e6,
            gate=gate,
            h=hidden + 1,
            bits=gate * 4,
            noise=None,
        )

    @staticmethod
    def log_prob(out, action):
        return torch.zeros_like(action, dtype=torch.float32)


class ScriptedCritic:
    def __call__(self, state, obs):
        return state[:, :1].expand(obs.shape[:2])


class MovingActor(ScriptedActor):
    def __call__(self, *args):
        out = super().__call__(*args)
        out.action_logits.zero_()
        out.action_logits[..., 2] = 1e6
        return out


def make_buffer(t, b=2, n=1, d=1, slots=0, state_dim=1):
    return RolloutBuffer(t, b, n, d, slots, state_dim, hidden=1, msg_bits=0)


def test_every_cached_transition_uses_pre_action_obs_and_the_current_step_reward_done():
    env, actor = ScriptedVec(), ScriptedActor()
    buf = make_buffer(5)
    result = collect_rollout(
        env, actor, ScriptedCritic(), buf, env.reset(), torch.zeros(2, 1, 1)
    )
    for t, actual in enumerate(env.history):
        np.testing.assert_array_equal(buf.obs[t], actual["obs"])
        np.testing.assert_array_equal(buf.state[t], actual["obs"][:, :, 0])
        np.testing.assert_array_equal(buf.action[t], actual["action"])
        np.testing.assert_array_equal(buf.reward_team[t], actual["reward"])
        np.testing.assert_array_equal(buf.done[t], actual["done"])
    assert buf.reward_team[0].min() > 0
    np.testing.assert_array_equal(result.last_value, result.batch["state"])
    assert env.reset_calls == 1
    assert [ep["len"] for ep in result.episodes] == [2, 3, 2]
    assert actor.channels[0]["drop_p"].tolist() == [0.0, 1.0]


def test_hidden_resets_only_for_the_environment_that_just_ended():
    env = ScriptedVec()
    buf = make_buffer(5)
    result = collect_rollout(
        env, ScriptedActor(), ScriptedCritic(), buf, env.reset(), torch.zeros(2, 1, 1)
    )
    np.testing.assert_array_equal(
        buf.h_prev[:, :, 0, 0], [[0, 0], [1, 1], [0, 2], [1, 0], [0, 1]]
    )
    np.testing.assert_array_equal(result.hidden[:, 0, 0].numpy(), [1, 2])


def test_rollout_segments_preserve_long_episodes_and_hidden_state():
    env, actor = ScriptedVec(lengths=(5,)), ScriptedActor()
    initial = env.reset()
    first_buf = make_buffer(2, b=1)
    first = collect_rollout(
        env, actor, ScriptedCritic(), first_buf, initial, torch.zeros(1, 1, 1)
    )
    assert not first.episodes and env.ticks.tolist() == [2]
    second_buf = make_buffer(4, b=1)
    second = collect_rollout(
        env, actor, ScriptedCritic(), second_buf, first.batch, first.hidden
    )
    np.testing.assert_array_equal(second_buf.obs[0], first.batch["obs"])
    assert second_buf.h_prev[0, 0, 0, 0] == 2
    assert second.episodes == [{"len": 5}]
    assert second_buf.done[:, 0].tolist() == [False, False, True, False]
    assert env.reset_calls == 1


def test_real_vector_autoreset_initial_obs_begins_the_next_transition():
    cfg = EnvConfig(backend="mock", n_agents=3, max_steps=2)
    with VecFormationEnv(1, cfg) as env:
        outputs = []
        original_step = env.step

        def traced_step(action):
            nxt = original_step(action)
            outputs.append(nxt)
            return nxt

        env.step = traced_step
        initial = env.reset()
        buf = make_buffer(
            3, b=1, n=3, d=env.obs_dim, slots=cfg.n_obs, state_dim=env.state_dim
        )
        collect_rollout(
            env, MovingActor(), ScriptedCritic(), buf, initial, torch.zeros(1, 3, 1)
        )
        np.testing.assert_array_equal(buf.obs[0], initial["obs"])
        np.testing.assert_array_equal(buf.obs[1], outputs[0]["obs"])
        np.testing.assert_array_equal(buf.obs[2], outputs[1]["obs"])
        np.testing.assert_array_equal(
            buf.reward_team[1],
            outputs[1]["info_raw"][0].info["terminal_step"].reward[None],
        )
        assert buf.done[:, 0].tolist() == [False, True, False]
        assert not np.array_equal(
            buf.obs[1, 0], outputs[1]["info_raw"][0].info["terminal_step"].obs
        )
        assert not buf.h_prev[2].any()


def test_gae_hand_calculated_terminal_boundaries_do_not_cross_episodes():
    buf = make_buffer(4, b=1)
    buf.ptr = 4
    buf.reward_team[:, 0, 0] = [1, 2, 10, 20]
    buf.value[:, 0, 0] = [0.5, 0.7, 100, 200]
    buf.done[:, 0] = [False, True, False, True]
    last = np.array([[999]], dtype=np.float32)
    buf.compute_returns(buf.reward_team, last, gamma=0.9, lam=1.0)
    np.testing.assert_allclose(buf.ret[:, 0, 0], [2.8, 2, 28, 20], atol=1e-5)
    adv, ret = gae(buf.reward_team, buf.value, buf.done, 0.9, 1.0, last)
    np.testing.assert_allclose(buf.ret, ret, atol=1e-5)
    np.testing.assert_allclose(buf.adv, adv, atol=1e-5)


def test_gae_nonterminal_segment_uses_latest_next_value():
    buf = make_buffer(2, b=1)
    buf.ptr = 2
    buf.reward_team[:, 0, 0] = [1, 2]
    buf.done[:, 0] = [True, False]
    buf.compute_returns(
        buf.reward_team, np.array([[4]], dtype=np.float32), gamma=0.5, lam=1.0
    )
    np.testing.assert_allclose(buf.ret[:, 0, 0], [1, 4])


@pytest.mark.parametrize("name", ["bac", "ic3net", "tarmac"])
def test_collected_gate_hidden_and_noise_reproduce_old_policy_log_probability(name):
    torch.manual_seed(12)
    cfg = EnvConfig(backend="mock", n_agents=3, max_steps=3, comm_radius=10.0)
    with VecFormationEnv(2, cfg) as env:
        comm = build_comm(name, 8, {"msg_bits": 4, "msg_dim": 4})
        actor = Actor(env.obs_dim, cfg.n_obs, 5, 8, comm)
        critic = Critic("central", env.state_dim, env.obs_dim, cfg.n_agents, 8)
        buf = RolloutBuffer(
            4, 2, 3, env.obs_dim, cfg.n_obs, env.state_dim, 8, 4 if name == "bac" else 0
        )
        initial = env.reset()
        initial["channel"]["drop_p"] = np.array([0.0, 1.0], dtype=np.float32)
        collect_rollout(env, actor, critic, buf, initial, actor.init_hidden(2, 3))
        batch = {
            k: torch.as_tensor(v) for k, v in buf.batched(buf.flat_indices()).items()
        }
        with torch.no_grad():
            out = actor(
                batch["obs"],
                batch["obs_mask"],
                batch["comm_mask"],
                batch["h_prev"],
                {},
                gate=batch["gate"],
                noise=ChannelNoise(batch["noise_drop"], batch["noise_flip"]),
            )
        torch.testing.assert_close(
            actor.log_prob(out, batch["action"]), batch["logp_old"]
        )
        assert not buf.noise_drop[0, 0].any()
        assert buf.noise_drop[0, 1][initial["comm_mask"][1]].all()
