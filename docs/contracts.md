# 接口与规范（唯一事实来源）

> 改这里之前先改文档，再改代码。第 1 周末冻结 `EnvStep` 与 `CommModule` 签名，之后只增字段。

## 1. 张量形状

- 批在前、智能体次之：单步 `[B, N, ...]`，序列 `[T, B, N, ...]`，`B` 为并行环境数。
- 观测与数值 `float32`，掩码 `bool`，动作与门控 `int64`。
- 所有函数 docstring 写形状注释，如 `# [B, N, H]`。

## 2. 环境接口（负责人 B）

```python
class FormationCommEnv:
    def __init__(self, cfg: EnvConfig): ...
    def reset(self, seed: int | None = None) -> EnvStep: ...
    def step(self, action) -> EnvStep: ...      # action: [N] int64
```

| 字段 | 形状 | 类型 | 含义 |
|---|---|---|---|
| obs | [N, 4 + 2*n_obs] | float32 | 自身速度(2) + 编队中心相对位置(2) + 最近 n_obs 个邻居相对位置（按距离升序，不足补零） |
| obs_mask | [N, n_obs] | bool | 邻居槽位是否有效 |
| state | [D_state] | float32 | 全体位置与速度 + 编队中心，仅供 Critic |
| comm_mask | [N, N] | bool | j 在 i 的 R_comm 内且属最近 K 个；对角为 False；不含门控与丢包 |
| reward | [N] | float32 | 团队项 r_t（各机相同）；通信代价不在环境内计算 |
| done | 标量 | bool | 时限到达，或完成成立且配置提前结束 |
| info | dict | — | formation_err, collisions, success, hold_steps, channel |

约定：
- 环境只负责物理、观测、团队奖励；门控、消息、信道噪声在模型侧，环境不感知消息内容。
- `reward = -(formation_err + λ_c·collisions)`；槽位匹配用 `scipy.optimize.linear_sum_assignment`，无 scipy 时退化为贪心。
- 信道参数由环境每回合 reset 时采样（训练从区间采，评估取固定值），经 `info["channel"]` 传给模型。
- `vec_env` 把 B 个环境堆成 `[B, N, ...]`，回合结束自动 reset，回合级统计留在 `info`。

## 3. 模型接口（负责人 C；通信模块 A、D）

四种通信方式实现同一个基类，Actor 只认这个接口：

```python
class CommModule(nn.Module):
    has_gate: bool          # 是否带发言门控
    bits_per_msg: int       # BAC 为 L；连续消息为 32 × 维度；无通信为 0

    def forward(self, h, comm_mask, gate, channel, noise=None) -> CommOut:
        # h [B, N, H]  comm_mask [B, N, N]  gate [B, N]  noise=None 表示新采样
        # 返回 CommOut(c=[B,N,C], bits=[B,N], attn=[B,N,N], noise=ChannelNoise)
```

```python
class Actor(nn.Module):
    def init_hidden(self, B: int, N: int) -> Tensor        # [B, N, H]
    def forward(self, obs, obs_mask, comm_mask, h_prev, channel, gate=None, noise=None) -> ActorOut
    # ActorOut: action_logits [B, N, 5], gate_logits [B, N] 或 None

class Critic(nn.Module):
    def forward(self, state=None, obs=None) -> Tensor     # [B, N]
    # central: 读 state [B, D_state]；local（IPPO）: 读 obs [B, N, D_obs]
```

单步计算顺序（所有方法一致，采样与更新两条路径共用）：
1. `h = GRU(encoder(obs, obs_mask), h_prev)`
2. 门控：有门控由 `gate_logits` 采样，更新时改用存储的 `gate`；无门控全 1
3. `comm = comm_module(h, comm_mask, gate, channel, noise)`，更新时传入存储的 `noise`
4. `action_logits = policy_head(h, comm.c)`
5. `log π = log π(a) + log π(g)`，无门控时只有前一项

信道：`channel.py` 提供 `sample(recv_mask, L, params)` 与 `apply(msgs, recv_mask, noise)`。二值消息同时做整条丢弃和比特翻转；连续消息（IC3Net/TarMAC）只做整条丢弃。

## 4. 训练接口（负责人 C）

| 缓存字段 | 形状 | 说明 |
|---|---|---|
| obs, obs_mask, state, comm_mask | [T, B, ...] | 来自环境 |
| h0 | [B, N, H] | 每段序列初始隐状态 |
| action | [T, B, N] | 动作样本 |
| gate | [T, B, N] | 门控样本 |
| noise | [T, B, ...] | 信道噪声样本，更新时复用 |
| logp_old | [T, B, N] | 采样时的 log π(a)+log π(g) |
| value | [T, B, N] | Critic 估计 |
| reward_team | [T, B, N] | 环境返回的团队项 |
| bits | [T, B, N] | 实际发送比特数 = gate × bits_per_msg |
| done | [T, B] | 回合结束标志 |

- 训练奖励在缓存中合成：`r_i = reward_team - β·bits_i`，β 取采样时的值。
- 每轮迭代顺序：采样 → 合成奖励 → GAE → PPO 多轮小批量更新 → 用本轮比特均值更新 β → 记录日志 → 定期评估与保存。
- 统一入口：`python -m bacmarl.runners.train env=formation algo=bac env.n_agents=6 algo.msg_bits=4 seed=0`

## 5. 配置规范（负责人 C；数值由 B 标定后填写）

| 配置键 | 含义 | 负责人 |
|---|---|---|
| env.n_agents | 无人机数 N | B |
| env.max_steps | 时限 T_max | B |
| env.formation_radius | 编队半径 r_f | B |
| env.obs_radius, env.n_obs | 感知半径 R_obs、观测邻居数 n_obs | B |
| env.comm_radius, env.recv_k | 通信半径 R_comm、接收容量 K | B |
| env.safe_dist, env.collision_coef | d_safe、λ_c | B |
| env.max_speed | v_max | B |
| env.success_dist, env.success_hold | 完成阈值 δ、保持步数 H | B |
| env.channel.drop_p, env.channel.flip_p | 训练采区间，评估取定值 | B |
| algo.comm | none / ic3net / tarmac / bac | A |
| algo.critic | central / local（local 即 IPPO） | C |
| algo.msg_bits | 消息长度 L | A |
| algo.budget_bits | 比特预算 b_max | A |
| algo.beta_init, algo.beta_lr | β 初值与对偶步长 | A |
| train.* | 总步数、并行环境数、采样长度、PPO 超参 | C |
| seed | 随机种子 | E |

- 所有方法共用 `train.*`，只允许改 `algo.*`。
- 任何默认值改动须在群里同步并写进提交说明。

## 6. 日志与结果规范（负责人 E）

- 运行名 `{env}_{N}n_{algo}_L{msg_bits}_K{recv_k}_B{budget_bits}_s{seed}`，输出到 `results/{实验组}/{运行名}/`，内含 `config.yaml`、`metrics.csv`、`tb/`、`ckpt/`。
- 横轴统一用 `env_steps`（所有并行环境累计步数，不乘智能体数）。
- 指标名 TensorBoard 与 `metrics.csv` 列名一致；QMIX 结果经 `baselines/qmix_benchmarl` 转成同一格式。

| 指标名 | 含义 |
|---|---|
| train/return, eval/return | 回合团队回报（不含通信代价） |
| train/success_rate, eval/success_rate | 完成率 |
| train/formation_err, eval/formation_err | 终止时刻编队误差 |
| train/collisions, eval/collisions | 每回合违反安全距离次数 |
| comm/bits | 每机每步平均发送比特数 |
| comm/speak_rate | 门控发言率 |
| comm/beta | 当前 β |
| loss/policy, loss/value, loss/entropy | 各项损失 |

- 评估：每隔固定步数用确定性策略跑固定数量回合；鲁棒性由 `evaluate.py` 读 checkpoint，按丢包率网格复测，输出 `robust.csv`（列：drop_p, success_rate, formation_err, speak_rate, bits）。
- 作图：只读 `metrics.csv` 与 `robust.csv`，曲线为多种子均值 ± 标准差。

## 7. 协作流程

- 分支：`main` 受保护；功能分支 `feat/<模块>-<内容>`，如 `feat/env-comm-mask`；合并需 1 人审阅并通过 `tests/`。
- 测试：每个接口至少一条形状测试（`test_env_api.py`、`test_comm_shapes.py`、`test_buffer.py`）；改接口必须同步改测试。
- 风格：black 格式化，函数写类型注解与形状注释。
- 接口冻结：第 1 周末冻结 `EnvStep` 与 `CommModule` 签名；之后只增字段，不改已有字段含义。
- 结果共享：实验配置入库，结果目录不入库；共享结果放统一网盘，按 `results/` 目录结构存放。
