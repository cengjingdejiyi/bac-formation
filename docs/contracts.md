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

### BAC 变体与 β 对偶（负责人 A，2026-11-20）

- `algo.aggregate`：`softmax`（默认，q/k 注意力）或 `mean`（消融 `mean_agg`，只在可见边上做均匀平均）。`mean` 模式下 `attn` 是「可见权重」而不是注意力，`w_q/w_k` 仍保留参数以保持两版同构，但不参与计算、梯度为 0。
- `algo.use_gate=false`（消融 `no_gate`）：`CommBAC.has_gate=False`，Actor 因此不建门控头、gate 恒为 1、`bits` 恒等于 `msg_bits`。此时 β 对偶只能推着常数预算走，结果只作对照，不能当成「学会按需发言」。
- 去信道噪声（消融 `no_noise`）是配置项：`env.channel.drop_p=0 env.channel.flip_p=0`。评测信道取「训练区间中点」，训练区间归零后必须显式写 `train.eval_drop_p` / `train.eval_flip_p`，否则评测也在零噪声下跑，鲁棒性无从比较。
- β 更新口径：`β ← clamp(β + η(b̂ − b_max), 0, β_max)`，η = `algo.beta_lr`，`β_max` = `algo.beta_max`（默认 5.0）；b̂ 为所有 T、B、N 的 `bits.mean()`，一轮更新一次；b̂ 非有限时跳过。训练奖励只合成 `r = reward_team - β·bits`，不再有单独的发言奖励系数（旧 `algo.r_gate` 已删除）。
- β 属对偶状态：`BetaDual.state_dict()` 写进 checkpoint 的 `meta["beta"]` 以便续训对齐；actor/critic 权重之外的对偶变量不能丢。

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
- 断点续训：`train.resume_from=<ckpt>` 恢复 actor/critic 权重、优化器动量、β（`meta["beta"]`）与奖励缩放统计（Welford 的 n/mean/m2），并从 `meta["steps"]` 接着计数；评测计时器按起始步对齐，回合级日志窗口（`recent_episodes`）不恢复。β 是唯一的对偶状态，丢了她等于换了预算口径，所以必须随 ckpt 存取。
- 统一入口：`python -m bacmarl.runners.train env=formation algo=bac env.n_agents=6 algo.msg_bits=4 seed=0`
- 缓存第 t 项是 `(obs_t, h_prev_t, action_t, gate_t, reward_{t+1}, done_{t+1})`：先从当前观测决策，再 step，最后把这次 step 的 reward/done 与**动作前**的观测一起写入缓存。终止帧只用于诊断，不能替代动作前的观测。
- rollout 段之间保留环境状态与循环隐状态；仅真正 done 的环境自动 reset，并在下一动作前清零对应隐状态。`done[t]` 同时切断 GAE 自举与跨回合优势传播；未终止段末使用最新 next state 的 value 自举。
- β 用 `bits.mean()`（所有 T、B、N 的均值）更新；一次广播不按接收边数重复收费，丢包也不退款。
- GAE 只有一份实现：`bacmarl.algos.gae.compute_gae`（numpy），`RolloutBuffer.compute_returns` 调用它；不允许再出现第二份 GAE。
- PPO 更新：优势在**整轮更新**上归一化（`train.norm_adv`），不逐小批量归一化。`train.reward_norm` 打开时，shaped reward 按 Welford 滑动标准差缩放（只缩小不放大），价值目标与自举同尺度，不改变最优策略；原始回合回报仍按 env 返回值写日志与评测。`train.value_clip_eps` 是可选的 PPO 式价值裁断（相对采样期 `v_old`），`null` 即关闭——奖励尺度未归一化时不要开，否则价值学习会被 0.2 的绝对窗宽卡死。日志补 `ppo/approx_kl`、`ppo/clip_frac`、`ppo/value_clip_frac`、`ppo/adv_std`、`train/reward_scale`，损失按小批量取平均而非取最后一批。
- 训练内评测：每 `train.eval_every` 个 `env_steps` 跑一次确定性评测（动作 argmax、门控取 p≥0.5、每回合零隐状态、回合内不重置），跑 `train.eval_episodes` 个回合。评测信道取**固定值**：默认是 `env.channel` 训练采样区间的中点，可用 `train.eval_drop_p` / `train.eval_flip_p` 覆盖；不得用训练期的逐环境采样区间评测。指标以 `eval/return`、`eval/success_rate`、`eval/formation_err`、`eval/collisions`、`eval/length`、`eval/bits`、`eval/speak_rate` 写入 `metrics.csv`，未评测的行写 `nan` 以保证表头稳定；同时落盘 `ckpt/step_{env_steps}.ckpt`。

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

- 运行名 `{env}_{N}n_{algo}_L{bits}_K{recv_k}_B{budget}_s{seed}`，输出到 `results/{实验组}/{运行名}/`，内含 `config.yaml`、`metrics.csv`、`tb/`、`ckpt/`。唯一实现是 `src/bacmarl/utils/naming.py`（train 与批量驱动共用，不再各写一份）。
- **L 段是单条消息的比特宽度 `bits_per_msg`**（Issue #11）：二值消息（BAC）取 `algo.msg_bits`；连续消息（IC3Net/TarMAC）取 `algo.bits_per_msg`，即 `32 × algo.msg_dim = 1024`，不得写 `L0`；无通信（MAPPO/IPPO/QMIX）为 `L0`。配置里 `msg_bits=0` 只是给信道采样占位，不代表零比特。
- **L 与 `comm/bits` 的关系**：`comm/bits` 是每机每步的平均发送量，`comm/bits = 发言率 × L`（门控方法不发言时为 0，丢包也照收不退款），所以 BAC 的 `L4` 在 `B2.0` 预算下实测约 2.0 bit/步，而 TarMAC 无门控，`L1024` 实测就是 1024。汇总表必须同时看 L 与 `comm/bits` 两列，只有 L 会掩盖门控的作用。
- **预算段**：`budget_bits` 有上限时沿用数字写法（`B2.0`）；无上限（`≥1e6`，如 IC3Net/TarMAC 的 `1e9` 占位）写 `Binf`，不允许出现 `B1000000000.0`。
- 汇总表/帕累托图必须按运行名里的 L 与 `comm/bits` 两列交叉核对：两者不一致即说明命名口径漂移，先修命名再出图。
- 消融变体在运行名末尾追加短后缀（默认值不加，既有目录不受影响）：`_agg{aggregate}`（非 `softmax`）、`_nogate`（`use_gate=false`）、`_nonoise`（`env.channel.drop_p=0` 且 `flip_p=0`），单独关一侧记 `_drop0` / `_flip0`，多项用 `-` 连接，例如 `..._s0_aggmean-nogate`。不加后缀会让 mean_agg 与 no_gate 落进同一目录并静默覆盖。
- 目标目录已有 `metrics.csv` 且本次不是续训时，训练入口打印 `[WARN]` 后覆盖；要并行保留多份请换 `experiment.name`。
- `configs/experiment/{ablation,sweep,robust}.yaml` 一律 `extends: main`：分组只声明实验矩阵，训练超参只有 `main.yaml` 一份真值；`experiment=<其它>` 不得清空 `train` 段。
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

- 评估：每隔固定步数用确定性策略跑固定数量回合；鲁棒性由 `evaluate.py` 读 checkpoint，按丢包率网格复测，输出 `robust.csv`（列：drop_p, success_rate, formation_err, speak_rate, bits；`--with-extra` 才追加 return/collisions/length）。网格来源优先级：`--drop-p` > `--config <实验配置>` > run 的 `config.yaml` > 仓库默认 `configs/experiment/robust.yaml`，实际来源必须打进日志（不许悄悄退化成 `[0.0]`）。
  - `--drop-p` 一旦显式给出就必须解析成功，解析失败直接 `[FAIL]` 返回 2；不许回退到仓库默认网格。
  - `--flip-p` 省略时取训练内评测口径（`train.eval_flip_p`，否则 `env.channel.flip_p` 训练区间中点），与 `train.py`/`qmix.py` 的 `eval/*` 保持同一信道；只有显式给值才覆盖。
  - `--ckpt` 是**相对 run 目录**的路径，默认 `ckpt/last.ckpt`；只看该 run 自己的目录，不跨 run 查找。
  - `--run` 目录不存在直接返回 2（不 `mkdir`）；缺 ckpt 返回 3；确实要做随机策略对照必须显式 `--allow-random`（日志会打 `[WARN] 退化为随机策略`，这种 `robust.csv` 不得进正式汇总）。
  - `--out` 只能配单个 `--run`；多 run 时各自写自己的 `robust.csv`。
- 作图：只读 `metrics.csv` 与 `robust.csv`，曲线为多种子均值 ± 标准差；`scripts/make_figures.py` 默认输出到 `results/figures/`（`train_eval_metrics.png`、`comm_metrics.png`、`robust_drop.png`、`curves_return.png`）。图例 label 取「相对 `--results` 的目录链 + run 名」（如 `main/formation_6n_bac_L4_K3_B2.0`），不同实验组里同名的 run 不得合并成多 seed。

## 7. 协作流程

- 分支：`main` 受保护；功能分支 `feat/<模块>-<内容>`，如 `feat/env-comm-mask`；合并需 1 人审阅并通过 `tests/`。
- 测试：每个接口至少一条形状测试（`test_env_api.py`、`test_comm_shapes.py`、`test_buffer.py`）；改接口必须同步改测试。
- 风格：black 格式化，函数写类型注解与形状注释。
- 接口冻结：第 1 周末冻结 `EnvStep` 与 `CommModule` 签名；之后只增字段，不改已有字段含义。
- 结果共享：实验配置入库，结果目录不入库；共享结果放统一网盘，按 `results/` 目录结构存放。
