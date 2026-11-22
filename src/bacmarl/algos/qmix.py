"""QMIX：单调值函数分解（对比基线，负责人 D）。

开题报告 §2.4：QMIX 作为「值函数分解」范式的代表，与策略梯度类（IPPO/MAPPO/
IC3Net/TarMAC/BAC）比较。它**不使用**通信模块，也不消费比特预算，因此
`comm/bits`、`comm/speak_rate` 恒为 0，运行名 `L0/B0.0`。

为什么在仓库内实现而不是直接调 BenchMARL（`baselines/qmix_benchmarl/` 仍保留日志
转换脚本）：

* 环境一致性。BenchMARL 的 MPE2 任务包的是原版 `simple_formation`：没有局部观测
  半径、没有碰撞惩罚、没有团队奖励重算，也没有本项目的完成判定。拿它和
  IPPO/MAPPO 比，比的不是算法而是环境。这里直接复用 `FormationCommEnv` /
  `VecFormationEnv`，奖励、终止、评测与策略梯度方法逐项同源。
* 可评测。本项目要求 `success_rate / formation_err / collisions` 与 `robust.csv`，
  这些都来自环境 `info`，经 `runners/evaluate.evaluate_policy` 统一产出。

实现口径（与 PyMARL 的 QMIX 对齐，差异写在注释里）：

1. 每个无人机共享一套 `AgentQNet`（Encoder + GRU + Q 头），输入只有自己的局部观测，
   不需要智能体编号——与 Actor 同构，保证「同参数量级」。
2. 隐状态作为输入使用、不做跨步 BPTT：rollout 时把 `h_prev` 与 `h_t` 一并存进
   replay，TD 目标只用存储的隐状态。这与策略梯度侧「缓存 h_prev、更新时不反传历史」
   的既有约定（contracts §4）一致，也让 replay 可以按转移均匀采样。
3. 混合网络 `MixingNet` 由全局状态经超网络生成非负权重（`abs`），保证
   ∂Q_tot/∂Q_i ≥ 0，即个体的贪心动作在联合 Q 上仍单调不降。
4. 团队奖励直接取环境返回的 `reward`（各机相同）；不额外做奖励缩放，避免非平稳
   的缩放把 TD 目标改成时变函数。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.encoder import Encoder


class AgentQNet(nn.Module):
    """共享参数的个体 Q 网络：局部观测 → Q(o, ·)，形状 [B, N, n_actions]。

    `use_agent_id=True` 时给输入追加 `agent_id` one-hot（PyMARL QMIX 参考实现的做法）。
    没有这个 one-hot 时，同构无人机看到相同观测就得到相同的 Q，贪心动作全一样，
    整队会朝同一个方向漂——实测 20k 步后编队误差 8.8、碰撞 65。主方法 Actor 不使用
    编号这一条不受影响（这是 QMIX 自己的输入约定，报告里单独说明）。
    """

    def __init__(
        self,
        obs_dim: int,
        n_obs: int,
        n_actions: int,
        hidden: int,
        n_agents: int = 1,
        use_agent_id: bool = False,
    ):
        super().__init__()
        self.hidden = hidden
        self.n_agents = int(n_agents)
        self.use_agent_id = bool(use_agent_id)
        in_dim = obs_dim + (self.n_agents if self.use_agent_id else 0)
        self.encoder = Encoder(in_dim, n_obs, hidden)
        self.gru = nn.GRUCell(hidden, hidden)
        self.q_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_actions)
        )
        if self.use_agent_id:
            self.register_buffer("agent_ids", torch.eye(self.n_agents), persistent=False)

    def init_hidden(self, b: int, n: int, device=None) -> torch.Tensor:  # [B, N, H]
        device = device or next(self.parameters()).device
        return torch.zeros(b, n, self.hidden, device=device)

    def _augment(self, obs: torch.Tensor) -> torch.Tensor:
        if not self.use_agent_id:
            return obs
        b = obs.shape[0]
        ids = self.agent_ids.unsqueeze(0).expand(b, -1, -1).to(obs.dtype)
        return torch.cat([obs, ids], dim=-1)

    def forward(self, obs, obs_mask, h_prev):
        """obs [B,N,D]、obs_mask [B,N,n_obs]、h_prev [B,N,H] → (q [B,N,A], h [B,N,H])。"""
        b, n, _ = obs.shape
        x = self.encoder(self._augment(obs), obs_mask).reshape(b * n, self.hidden)
        h = self.gru(x, h_prev.reshape(b * n, self.hidden)).reshape(
            b, n, self.hidden
        )
        return self.q_head(h), h


class MixingNet(nn.Module):
    """单调混合网络：Q_tot = f(Q_1..Q_N; s)，超网络按全局状态生成非负权重。"""

    def __init__(
        self,
        n_agents: int,
        state_dim: int,
        embed: int = 32,
        hypernet_embed: int = 64,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.embed = embed
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed),
            nn.ReLU(),
            nn.Linear(hypernet_embed, n_agents * embed),
        )
        self.hyper_b1 = nn.Linear(state_dim, embed)
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed),
            nn.ReLU(),
            nn.Linear(hypernet_embed, embed),
        )
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed), nn.ReLU(), nn.Linear(hypernet_embed, 1)
        )

    def forward(self, agent_qs, state):
        """agent_qs [B,N]（已选动作的 Q）、state [B,S] → Q_tot [B]。"""
        b = agent_qs.shape[0]
        w1 = torch.abs(self.hyper_w1(state)).view(b, self.n_agents, self.embed)
        b1 = self.hyper_b1(state).view(b, 1, self.embed)
        hidden = F.elu(torch.bmm(agent_qs.view(b, 1, self.n_agents), w1) + b1)
        w2 = torch.abs(self.hyper_w2(state)).view(b, self.embed, 1)
        b2 = self.hyper_b2(state).view(b, 1, 1)
        return (torch.bmm(hidden, w2) + b2).view(b)


class QMixAgent(nn.Module):
    """个体 Q + 混合网络；`forward` 同时返回逐动作 Q 与所选动作的 Q_tot。"""

    def __init__(
        self,
        obs_dim: int,
        n_obs: int,
        n_actions: int,
        hidden: int,
        n_agents: int,
        state_dim: int,
        mixer_embed: int = 32,
        hypernet_embed: int = 64,
        use_agent_id: bool = False,
    ):
        super().__init__()
        self.n_actions = n_actions
        self.agent = AgentQNet(
            obs_dim, n_obs, n_actions, hidden, n_agents, use_agent_id
        )
        self.mixer = MixingNet(n_agents, state_dim, mixer_embed, hypernet_embed)

    def init_hidden(self, b: int, n: int, device=None) -> torch.Tensor:
        return self.agent.init_hidden(b, n, device)

    def q_values(self, obs, obs_mask, h_prev):
        """只算个体 Q，用于 ε-贪心选动作。"""
        return self.agent(obs, obs_mask, h_prev)

    def q_tot(self, agent_qs, state):
        return self.mixer(agent_qs, state)

    def forward(self, obs, obs_mask, h_prev, state, action):
        """返回 (q [B,N,A], q_tot [B], h [B,N,H])，q_tot 用 action 对应的个体 Q 混合。"""
        q, h = self.agent(obs, obs_mask, h_prev)
        q_taken = q.gather(-1, action.unsqueeze(-1)).squeeze(-1)  # [B, N]
        return q, self.mixer(q_taken, state), h

    def target_copy(self) -> "QMixAgent":
        """深拷贝出 target 网络（不建梯度，硬更新用 load_state_dict）。"""
        target = copy.deepcopy(self)
        target.eval()
        for p in target.parameters():
            p.requires_grad_(False)
        return target

    def sync_target(self, target: "QMixAgent") -> None:
        target.load_state_dict(self.state_dict())


def epsilon_at(step: int, start: float, end: float, anneal_steps: int) -> float:
    """线性退火：step=0 取 start，step≥anneal_steps 取 end。"""
    if anneal_steps <= 0:
        return float(end)
    frac = min(1.0, max(0.0, float(step) / float(anneal_steps)))
    return float(start + (end - start) * frac)


def td_target(reward, done, q_tot_next, gamma: float, clip: float | None = None):
    """y = r + γ(1-d)·Q_tot(s')；done 为 float [B]，终止转移只保留 r。

    `clip` 给定时把 y 夹到 [-clip, clip]：Q 的自举是**不动点迭代**，一次发散就会把
    目标推到几千、再用 MSE 把梯度放大到百万级（2026-11-21 的 qmix 日志就是这个形状：
    `qmix/q_tot` 从 −5 涨到 5360、`loss/td` 峰值 9.4e6）。夹一刀等于给迭代加了界，
    不改变收敛点（合法 Q 量级见下），只拦掉数值逃逸。
    """
    y = reward + gamma * (1.0 - done) * q_tot_next
    if clip is not None and float(clip) > 0:
        y = y.clamp(-float(clip), float(clip))
    return y


@dataclass
class QMixLossStats:
    loss: float
    td_abs: float
    q_tot: float
    target: float
    clip_frac: float = 0.0


def qmix_td_loss(agent: QMixAgent, target: QMixAgent, batch: dict, gamma: float,
                 double_q: bool = True, huber: bool = True, huber_beta: float = 1.0,
                 target_clip: float | None = None) -> tuple[torch.Tensor, QMixLossStats]:
    """一次 TD 更新：Double-Q 目标 + 单调混合，返回标量 loss 与统计量。

    `huber=True` 用 smooth-L1（β 控制二次/线性切换点）：离群 TD 误差只贡献线性梯度，
    不会像 MSE 那样把一个坏样本放大成 1e6 级 loss。这是 QMIX 系实现里最常见的稳定化
    手段，和 PPO 侧 `value_norm` 针对的是同一类"价值目标量级失控"。

    batch 的键见 `bacmarl.algos.replay.TransitionReplay.sample`，形状均为 [K, ...]。
    """
    q, q_tot, _ = agent(
        batch["obs"], batch["obs_mask"], batch["h_prev"], batch["state"], batch["action"]
    )
    with torch.no_grad():
        q_online_next, _ = agent.agent(
            batch["next_obs"], batch["next_obs_mask"], batch["next_h"]
        )
        q_target_next, _ = target.agent(
            batch["next_obs"], batch["next_obs_mask"], batch["next_h"]
        )
        if double_q:
            idx = q_online_next.argmax(-1, keepdim=True)
            q_next = q_target_next.gather(-1, idx).squeeze(-1)
        else:
            q_next = q_target_next.max(-1).values
        q_tot_next = target.mixer(q_next, batch["next_state"])
        raw = td_target(batch["reward"], batch["done"], q_tot_next, gamma)
        y = td_target(
            batch["reward"], batch["done"], q_tot_next, gamma, clip=target_clip
        )

    if huber:
        loss = F.smooth_l1_loss(q_tot, y, beta=float(huber_beta))
    else:
        loss = F.mse_loss(q_tot, y)
    td = (q_tot - y).detach()
    clip_frac = float((raw != y).float().mean()) if target_clip else 0.0
    stats = QMixLossStats(
        loss=float(loss.detach()),
        td_abs=float(td.abs().mean()),
        q_tot=float(q_tot.mean().detach()),
        target=float(y.mean()),
        clip_frac=clip_frac,
    )
    return loss, stats
