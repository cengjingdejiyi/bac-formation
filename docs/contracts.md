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
- `reward = -(formation_err + λ_c·collisions)`；`collisions = sum_{i<j} 1[distance(i,j)<safe_dist]` 是不安全机对数，**不是**“本步存在碰撞”的 0/1 值。槽位匹配用 `scipy.optimize.linear_sum_assignment`，无 scipy 时退化为贪心。
- 信道参数由环境每回合 reset 时采样（训练从区间采，评估取固定值），经 `info["channel"]` 传给模型。
- `vec_env` 把 B 个环境堆成 `[B, N, ...]`，回合结束自动 reset，回合级统计留在 `info`。

### B 环境实现补充（2026-11-13）

- `backend=mpe2` 使用 `mpe2.simple_formation_v1.parallel_env` 的真实物理；`auto` 仅在未安装 `mpe2` 时告警退回 mock，安装损坏或接口不匹配须报错。mock 仅用于契约冒烟。
- 原生观测是 `[速度(2), 绝对位置(2), 地标相对位置(2)]`；包装层从 world 按固定 agent 顺序读取位置/速度，统一组装上述局部观测，不向 Actor 泄漏绝对位置。
- 中心取唯一 landmark。正式环境采用 `heading_mode=dynamic`：每个状态以相对中心极角（归一化到 `[0, 2π)`）最小的无人机锚定槽位朝向，与报告 §1.4 和 MPE2 一致；规则编队整体旋转后仍可完成。包装层按该朝向、可配 `formation_radius` 重算未裁剪的匹配误差与安全距离惩罚，奖励、完成判定和诊断使用同一朝向。`info.heading` 记录实际朝向，仅供诊断，不新增 Actor 观测。显式 `heading_mode=fixed` 时使用配置的 `heading`，保留历史固定朝向实验。`max_speed` 限制速度向量的模长。
- 完成要求所有匹配距离不超过 `success_dist` 且连续保持 `success_hold` 步；`info.success` 表示当前满足保持条件。回合统计 `ep.success` 记录本回合是否曾完成。
- 自动 reset 后 obs/state/comm_mask/channel 来自新回合，`reward`、`done`、`ep` 保留刚结束的转移；`info_raw[k].info["terminal_step"]` 保留终止帧（含终止指标和信道），便于评测。后续回合不重新播种，整段运行可复现且初始状态不重复。
- 单环境和并行环境提供 `close()` 及上下文管理器以释放 MPE2 资源。`scene=spread` 仍是选做扩展，本次不实现；须显式报错，避免误把编队结果当作覆盖实验。
- 标定脚本和报告见 `scripts/calibrate_env.py`、`docs/env-calibration.md`。正式编队配置统一为 `formation.yaml`（R_obs=R_comm=0.8）；旧半径只保留在明确标记的 `formation_legacy.yaml` 对照配置中。
- B 阈值验收默认采用 `ScriptedController`（标定策略名 `scripted`）：每架机仅使用自身速度、中心相对位置和掩码允许的邻居相对位置，通过径向收敛、局部角间距均衡与速度反馈排列槽位。不读取其他无人机的观测行、全局 state、`info.pos`、编号或匈牙利指派。`radial` 保留旧控制器作对照；`slot_oracle` 仍仅为全局可行性诊断。正式 `success_dist=0.09` 按局部控制器多种子标定，保留 `success_hold=10, max_steps=100`；0.05 仍可显式配置作严格阈值对照。脚本验证不代表学习策略收敛。

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

- 所有边张量统一为接收者在前、发送者在后：`comm_mask[b,i,j]` 表示 i 收 j，`edge[b,i,j,:]` 是 j 发给 i 的消息，`noise.drop[b,i,j]` 与 `noise.flip[b,i,j,:]` 作用于同一条 j→i 边。NumPy 与 torch 实现必须一致，不能只在聚合端转置已经掩码/加噪的边。
- 只有半径/容量掩码允许、发送者 gate=1 且未丢包的边参与归一化。无有效邻居时权重与通信上下文均为 0；注意力不能越过输入掩码。
- `channel.drop_p/flip_p` 支持标量或逐并行环境 `[B]` 参数；不得把不同环境的信道概率取均值后使用。PPO 更新复用存储的噪声。

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
- 缓存第 t 项是 `(obs_t, h_prev_t, action_t, gate_t, reward_{t+1}, done_{t+1})`：先从当前观测决策，再 step，最后把这次 step 的 reward/done 与**动作前**的观测一起写入缓存。终止帧只用于诊断，不能替代动作前的观测。
- rollout 段之间保留环境状态与循环隐状态；仅真正 done 的环境自动 reset，并在下一动作前清零对应隐状态。`done[t]` 同时切断 GAE 自举与跨回合优势传播；未终止段末使用最新 next state 的 value 自举。
- β 用 `bits.mean()`（所有 T、B、N 的均值）更新；一次广播不按接收边数重复收费，丢包也不退款。

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
| env.heading_mode, env.heading | dynamic 为正式状态锚定朝向；fixed 为显式固定朝向对照 | B |
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
- 本次修复统一正式参数：`backend=mpe2, obs_radius=comm_radius=0.8, collision_coef=1.0`；后者按每个不安全机对计价。`formation_calibrated` 仅是正式配置的兼容别名，`formation_legacy` 仅供旧半径对照，不参与默认正式实验。几何与可行性证据不等于惩罚系数最优或训练已收敛，报告数值以此配置表和 `docs/env-calibration.md` 为准。
- Issue #4 后正式朝向默认改为 `heading_mode=dynamic`、阈值改为 `success_dist=0.09`，`formation_legacy` 显式保留 `fixed`、0.05 阈值及旧半径。旧固定朝向的回报、成功率和标定结果不能与新口径混用；新标定清单保存完整配置与实际运行版本。

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
