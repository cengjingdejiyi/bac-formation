# 对比基线交付说明（负责人 D）

> 范围：开题报告 §2.4 的四条对比基线中属于 D 的部分——IC3Net 式、TarMAC 式通信模块，
> QMIX 值分解基线，以及「统一调参、日志统一格式」这层接口。
> 契约与命名口径见 [接口与规范](contracts.md) 第 3、6 节；本文只讲**基线怎么摆、怎么跑、怎么读**。

## 1. 交付清单

| 交付物 | 位置 | 状态 |
|---|---|---|
| IC3Net 式（门控 + 连续消息 + 均值聚合） | `src/bacmarl/comm/ic3net.py` | 已在 main（PR #1 修通信方向） |
| TarMAC 式（无门控全精度广播 + 注意力） | `src/bacmarl/comm/tarmac.py` | 已在 main |
| QMIX 值分解基线 | `src/bacmarl/algos/qmix.py`、`algos/replay.py`、`runners/qmix.py`、`configs/algo/qmix.yaml` | 本分支新增 |
| BenchMARL 日志 → `metrics.csv` 转换 | `baselines/qmix_benchmarl/convert.py` | 本分支新增（备用路径） |
| 运行名/比特口径修复（Issue #11） | `src/bacmarl/utils/naming.py`、`configs/algo/{ic3net,tarmac}.yaml`、`docs/contracts.md` §6 | 本分支新增 |
| 调参记录 | 本文 §4、§5 | — |

## 2. 三条基线在实验里各自的位置

| 方法 | 范式 | 通信方式 | 运行名里的 L | 检验什么 | 报告中的角色 |
|---|---|---|---|---|---|
| IC3Net 式 | CTDE + 门控 | 连续消息、均值聚合、无预算 | `L1024` / `Binf` | 比特效率是否来自注意力聚合而非门控本身 | 经典门控通信基线 |
| TarMAC 式 | CTDE，无门控 | 每步广播 32 维 fp32 = 1024 bit，注意力聚合 | `L1024` / `Binf` | 比特压缩的代价 | 通信性能上界 |
| QMIX | 值函数分解 | 不通信 | `L0` / `B0.0` | 策略梯度与值分解两类范式的差异 | 不同范式对照（不是「坏方法」） |

IPPO / MAPPO 属 C 的任务（`algo.comm=none` + `algo.critic=local/central`），这里只保证
与它们共用同一环境、同一奖励、同一评测与同一日志列。

**读表要点**：`L` 是单条消息的比特宽度，`comm/bits` 是每机每步的平均发送量
（`= 发言率 × L`）。IC3Net 有门控，实测 `comm/bits` 在 450–620 之间浮动；TarMAC 无门控，
恒等于 1024。把 IC3Net 的 `comm/bits` 直接和 BAC 的 `L4` 相比是错的——要比的是同一张
「完成率—比特数」图上的位置。

## 3. Issue #11：运行名与通信量口径（已修）

修复前 `algo=ic3net|tarmac` 的训练目录名是 `formation_6n_ic3net_L0_K3_B1e9_s0`：名字说
「零比特、预算 1e9」，实测 `comm/bits` 是 597~1024。按目录名汇总的表格会把两条基线读成
「不花通信预算就达到同等性能」，这是**结论级别**的误读。

修复后（真实 MPE2、32 步、seed=0、B=2 的实测）：

| 算法 | 修复前目录名 | 修复后目录名 | 实测 `comm/bits` | 实测 `speak_rate` |
|---|---|---|---|---|
| ic3net | `formation_6n_ic3net_L0_K3_B1e9_s0` | `formation_6n_ic3net_L1024_K3_Binf_s0` | 448–618 | 0.44–0.60 |
| tarmac | `formation_6n_tarmac_L0_K3_B1e9_s0` | `formation_6n_tarmac_L1024_K3_Binf_s0` | 1024.0（恒定） | 1.0 |
| qmix | — | `formation_6n_qmix_L0_K3_B0.0_s0` | 0.0 | 0.0 |
| bac（对照） | `formation_6n_bac_L4_K3_B2.0_s0` | 不变 | ≈2.0（预算生效时） | 按门控 |

实现要点：

- 运行名只有一份实现：`src/bacmarl/utils/naming.py`。`runners/train.py`、`runners/qmix.py`
  都从这里取，E 的批量驱动应 `from bacmarl.utils.naming import run_name`，不要再镜像一份
  （#13 里的镜像实现已经和这份对齐，合并后建议直接改 import，见 §7）。
- `L` 取 `algo.bits_per_msg`（连续消息缺省 `32 × msg_dim`），配置里显式写了 `bits_per_msg: 1024`，
  不是从 `msg_bits` 猜的。
- 预算无上限（`≥1e6`）写 `Binf`，不再出现 `B1000000000.0`。
- 防漂移测试：`tests/test_run_name.py::test_bit_width_matches_comm_module` 直接比对
  「名字里的 L」与通信模块上报的 `bits_per_msg`，两边不一致 CI 就红。

## 4. QMIX 实现说明

### 4.1 为什么在仓库内实现，而不是直接调 BenchMARL

开题报告写的是「QMIX 采用 BenchMARL 的现成实现」。落地时改成**仓库内实现 + 保留
BenchMARL 日志转换**，理由是按重要性排序：

1. **环境一致性**。BenchMARL 的 MPE2 任务包的是原版 `simple_formation`：没有局部观测半径、
   没有碰撞惩罚、没有本项目的团队奖励重算与完成判定。用它和 IPPO/MAPPO 比，比出来的是
   环境差异，不是算法差异。现在的 QMIX 直接跑 `FormationCommEnv` / `VecFormationEnv`。
2. **可比指标**。`success_rate / formation_err / collisions` 都来自环境 `info`，经
   `runners/evaluate.evaluate_policy` 统一产出；BenchMARL 侧拿不到这些量，报告里就只剩回报曲线。
3. **可复现与 CI**。仓库内实现能被 `pytest` 和 CI 的 32 步冒烟覆盖；BenchMARL + torchrl 的
   依赖（大版本敏感）进 CI 会让整条流水线变脆。

BenchMARL 这条路没有删：`baselines/qmix_benchmarl/convert.py` 把外部日志转成同一套列，
`--out results/<实验组>/<运行名>` 之后，作图脚本对两种来源一视同仁。

### 4.2 算法与工程口径

- **个体 Q**：`AgentQNet` = `Encoder`(与 Actor 同一个) + `GRUCell` + Q 头，所有无人机共享参数，
  输入是局部观测 + **`agent_id` one-hot**（PyMARL QMIX 参考实现的做法）。没有这个 one-hot 时，
  同构无人机看到相同观测就得到相同 Q，贪心时整队选同一个动作、编队直接漂走（实测 20k 步后
  编队误差 8.8、碰撞 65）。这是 QMIX 自己的输入约定，主方法 Actor「不用编号」的契约不受影响，
  报告里单独说明这一条差异。
- **混合网络**：`MixingNet` 由全局状态经超网络生成 `abs()` 后的权重，保证
  `∂Q_tot/∂Q_i ≥ 0`（有单测 `test_mixing_is_monotonic_in_each_agent_q`）。
- **目标**：Double-Q + 硬更新 target（默认每 500 个梯度步同步一次）+ n 步回报（默认 n=5）。
- **隐状态**：rollout 时把 `h_prev` 与 `h_t` 一起写进 replay，更新时不跨步反传——与 PPO 侧
  「缓存 h_prev、不做全轨迹 BPTT」的既有约定一致（contracts §4）。
- **奖励缩放**：学习用的每步奖励乘 `algo.reward_scale`（默认 0.1）。100 步回合、每步 ≈ −1，
  不缩放时价值量级到 −100，1 步 TD 在 200k 步预算内传不到那么远（实测 20k 步只到 −7 的量级）。
  缩放是 DQN 系的常规做法，环境返回的回合回报照旧记账，指标不受影响。
- **探索**：逐机 ε-贪心，`1.0 → 0.05`，线性退火 100k 步。
- **评测**：复用 `evaluate_policy`（确定性 argmax、每回合零隐状态），所以 `eval/*` 与
  策略梯度方法同口径；`--robust` 额外按丢包率网格写 `robust.csv`（契约 5 列）。

### 4.3 与策略梯度侧对齐的口径

| 项目 | 策略梯度入口 `runners/train.py` | QMIX 入口 `runners/qmix.py` |
|---|---|---|
| 环境/奖励/终止 | `FormationCommEnv` + `VecFormationEnv` | 同左 |
| 总步数、并行环境数、hidden、gamma | `train.*` | 同左（只多 `algo.lr` 等值分解专用项） |
| 评测入口 | `evaluate_policy` | 同左（`GreedyQPolicy` 适配） |
| 评测信道 | 训练区间中点 / `train.eval_drop_p/flip_p` | 同左（共用 `utils/runtime.eval_channel`） |
| 运行名 | `utils/naming.run_name` | 同左 |
| 日志列 | `RunLogger` 全列 | 同列；`comm/*` 恒 0，另有 `qmix/*` 诊断列 |

### 4.4 两条容易读错的边界

**① resume 不等于严格续训。** `runners/qmix.py` 读 `train.resume_from` 时只恢复 agent / target /
optimizer 与 `meta["steps"]`：回放不落盘，所以续训后 replay 是空的，还要重新预热
`algo.train_start` 步才开始更新；ε 按 `steps` 重算（会接着退火，这点没问题）。要真正续跑，
要么一次把 `train.total_env_steps` 设够，要么别把 QMIX 的 resume 和 PPO 的 `resume_from`
（它恢复 β、优化器动量与奖励缩放统计，见 contracts §4）写成同一件事。

**② `robust.csv` 对 QMIX 是整行平的，不只是 bits/speak。** `env.channel` 只喂给模型侧的通信信道，
环境物理不读它，而 QMIX 没有通信模块。实测同一 checkpoint 在 `drop_p=0.0` 与 `0.5` 下评测
五列逐位相同：`return` −265.9562、`formation_err` 4.9634、`collisions` 1.5、`bits`/`speak_rate` 0。
所以：

- 别把 QMIX 的 robust 曲线和 BAC 的「丢包下比特节省」画在一张图里比——一个是结构性零通信，
  一个是预算约束在起作用；
- 报告里若保留丢包鲁棒性图，QMIX 只能当「无通信基线的水平参考线」，并注明它的 `bits` 恒 0、
  与丢包率无关。


## 5. 调参记录

（下表是实际跑过的配置；「依据」一列写的是观察到的问题，不是事后编的理由。）

### 5.1 QMIX 超参

| 参数 | 取值 | 依据 |
|---|---|---|
| `algo.lr` | 5e-4 | 1e-3 时 TD loss 抖动大、Q 量级发散；5e-4 稳定 |
| `algo.batch_size` | 32 | 64 在本机 CPU 上 25 ms/步，200k 步预算下太贵；32 约 15 ms |
| `algo.n_step` | 5 | n=1 时 20k 步后 Q 只到 −7（视野约 7 步），n=5 后同一预算到 −30~−48 |
| `algo.reward_scale` | 0.1 | 不缩放时价值量级 −100 量级，TD 传播慢且易过冲 |
| `algo.updates_per_train` | 4（≈1 次更新 / 2 env_step） | 1 次/8 步时 200k 步只有 25k 次更新，明显欠拟合 |
| `algo.target_update` | 500 梯度步 | 200 步时 target 追得太紧、Q 抖动；1000 步时学习变慢 |
| `algo.eps_anneal_steps` | 1e5 | 与正式实验 2e5 步匹配：前一半充分探索，后一半利用 |
| `algo.double_q` | true | 关掉后 `qmix/q_tot` 系统性高于 `qmix/target`（过估计），默认开 |
| `algo.use_agent_id` | true | 关掉后同构智能体贪心动作完全一致，整队朝同一方向漂（实测编队误差 8.8、碰撞 65） |
| `algo.hidden` | 沿用 `train.hidden=64` | 与 Actor 同容量，不做「加大基线网络」的不公平比较 |

### 5.2 实际跑过的对照（真实 MPE2，N=6，B=8，20k 步，seed=0）

| 配置 | 最终 `train/return` | `eval/return`（贪心，4 回合） | 观察 |
|---|---|---|---|
| 随机策略（参照线） | — | −115 | 判断「有没有学到东西」的现实基准 |
| 1 步 TD，64/批，1 次更新/8 步 | −114 | −189 | 价值传不到 100 步（Q 只到 −7 量级） |
| 5 步 TD，32/批，1 次更新/2 步，无缩放 | −151 | −446 | 视角变长但策略崩塌，指标反而更差 |
| 5 步 TD，缩放 0.1，1 次更新/8 步 | （跑到 5k 步中止） | −411 | 更新太少，20k 内学不动 |
| 5 步 TD，缩放 0.1，`agent_id`，1 次更新/2 步 | −133 | −312 | 当前默认配置；仍未达到随机线 |
| 上一条直接按正式配置跑（`eval_every=25k`，跑到 50k 步） | −171 | 25k: −361 / 50k: −425 | 不是「还没学会」，是**在变差**：`eval/formation_err` 6.6 → 7.8 |

**结论**：实现层面是对的（见 5.3），但这条奖励下 QMIX 的贪心策略会滑向「散开避碰」：
`λ_c=1.0` 按每个不安全机对计，6 机最多 15 对，而编队误差量级只有 1~8；散开能把碰撞项清零，
代价只是编队误差变大，于是**局部最优是「飞散」而不是「成形」**。观察到的现象与之自洽：
`eval/formation_err` 单调涨到 7.8、`eval/collisions` 掉到个位数、回报从 −115（随机）掉到 −425。

机制上还有一个更硬的理由：QMIX 的单调分解要求 `∂Q_tot/∂Q_i ≥ 0`，它能表达「某架机多往前走更好」，
但表达不了「两架机同时朝中间走会撞」这种**非单调的联合作用**。碰撞惩罚恰好是后者。
这不是实现 bug，是值分解范式的结构性限制（QTRAN/QPLEX 这类工作就是为松绑这条约束）。

结论：这条编队任务上，QMIX 作为「值分解范式代表」是有效的对照，但**不能当成同预算下的
性能对手**；报告里应把它写成范式差异与局限，而不是「QMIX 分数低=基线没做好」。

### 5.3 玩具任务验证（把「实现错」和「学不出来」分开）

`tests/test_qmix_learning.py` 构造了一个可判定的玩具环境：每架机选动作 1 得 0 分、其余 −1
（奖励按个体可分离，不考协同探索——协同探索是 §5.2 的实测结论，不该混进单测）。600 步、
ε 退火到 0.05 后，贪心策略必须稳定选中动作 1。这条测试在 CI 里秒级完成，证明 TD 目标、
n 步展开、混合网络、回放采样、贪心执行与 checkpoint 这条链路本身是通的。


### 5.4 IC3Net/TarMAC 为什么不单独调参

开题报告的交付物写的是「基线代码 + 调参记录」，这里把两条通信基线的调参决定写清楚，免得被问
「你的调参记录在哪」：

| 项目 | 取值 | 说明 |
|---|---|---|
| 学习率、PPO 超参、GAMMA、hidden | 完全沿用 `train.*` / `configs/experiment/main.yaml` | 与 BAC、IPPO、MAPPO 同一套；只换通信模块，这是 §2.4「公平比较」的前提 |
| `algo.msg_dim` | 32 | 开题报告里 TarMAC 的定义就是「每步广播 32 维连续向量」 |
| `algo.budget_bits` | 无上限（`1e9` → 运行名 `Binf`） | 连续消息按实数计，本来就没有二值预算可卡；β 对偶对它们是空操作（`beta_lr=0`） |
| `algo.msg_bits` | 0（占位） | 只给信道采样用；真实宽度由 `algo.bits_per_msg=1024` 表达（见 §3） |
| IC3Net 门控 | 保留（`has_gate=True`） | 这是 IC3Net 的定义特征，去掉它就不是 IC3Net 了 |

也就是说：**两条基线的调参结论是「不单独调参，直接继承共享超参」**，唯一动的是各自算法的定义性字段
（消息维度、有无门控、无预算）。这与 QMIX 相反——QMIX 是另一套学习范式，`algo.*` 下才有一整套专用超参。
报告里把这两句写清楚，比列一堆「调了哪些 lr」更有说服力。

### 5.5 结论与风险

- QMIX 在这条任务上显著依赖 **n 步回报 + 奖励缩放**，缺一个就会退化成「价值传不到位」或
  「贪心策略崩塌」。这两项都写在配置里，可复现，不藏在代码里。
- QMIX 不通信，因此 `--robust` 的丢包网格对它是**平线**——这是结构性结论（策略对信道
  免疫），不是 bug，报告里要说明它和通信类方法不在同一条鲁棒性坐标上；§4.4 ② 给了逐位相同的实测。


## 6. 复现命令

```bash
# 训练（与策略梯度入口同一套配置）
python -m bacmarl.runners.qmix env=formation algo=qmix seed=0
python -m bacmarl.runners.qmix env=formation algo=qmix seed=0 --robust   # 另写 robust.csv

# 三条基线各跑 32 步（CI 用同一组参数，验证链路而不是性能）
python -m bacmarl.runners.train env=formation algo=ic3net train.total_env_steps=32 \
  train.n_envs=2 train.rollout_len=4 train.ppo_epochs=1 train.minibatch=8 \
  train.hidden=16 env.max_steps=6 experiment.name=ci seed=0
python -m bacmarl.runners.train env=formation algo=tarmac train.total_env_steps=32 \
  train.n_envs=2 train.rollout_len=4 train.ppo_epochs=1 train.minibatch=8 \
  train.hidden=16 env.max_steps=6 experiment.name=ci seed=0
python -m bacmarl.runners.qmix env=formation algo=qmix train.total_env_steps=32 \
  train.n_envs=2 train.hidden=16 env.max_steps=6 experiment.name=ci seed=0

# BenchMARL 外部日志并入统一格式
python baselines/qmix_benchmarl/convert.py --in benchmarl_qmix.csv \
  --out results/main/formation_6n_qmix_L0_K3_B0.0_s0
```

## 7. 与 E 的接轨

**命名已经收口（本次已改）**：`scripts/run_experiments.py` 现在直接
`from bacmarl.utils.naming import run_name`，删掉了驱动里的镜像实现与 `_variant_suffix` 副本；
`tests/test_experiments_plan.py` 的口径对照从只覆盖 `algo=bac` 扩到
`bac/mappo/ippo/ic3net/tarmac/qmix`，并断言连续消息算出 `L1024_K3_Binf`。
所以排班目录名与训练实际写盘的名字不可能再分叉（#17 合并后这里曾经分叉过：驱动算
`L0_B1000000000.0`、训练写 `L1024_Binf`）。

**还差 E 补一个算法轴**：驱动目前只展开 `experiment` 组，算法固定 `algo=bac`；
「6 方法 × 3 seed」需要：

1. 加 `--algos bac ippo mappo ic3net tarmac qmix`（默认 `["bac"]`，保持既有行为不变）。
2. `_job()` 里把 `algo=<name>` 塞进 tokens（运行名会跟着变，命名实现已经共用）。
3. 执行阶段按 `job["algo"] == "qmix"` 选模块：`bacmarl.runners.qmix` 而不是 `bacmarl.runners.train`。
   两条入口的 `k=v` 参数完全一致，命令拼接逻辑不需要第二套。
4. QMIX 的 `train.total_env_steps`、`seed`、`experiment.name` 与其它方法一致，18 个 run 的
   横轴才对得齐。

在那之前，可以先用 shell 循环跑 QMIX 的 3 个 seed：

```bash
for s in 0 1 2; do python -m bacmarl.runners.qmix env=formation algo=qmix seed=$s; done
```


## 8. 已知限制

- **QMIX 在正式预算内不会收敛到「成形」策略**（见 §5.2）。它检验的是「策略梯度 vs 值分解」
  的范式差异，不该被读成「这个基线写坏了」：玩具任务（§5.3）证明链路是通的；观察到的发散
  与「奖励以碰撞项为主 + 单调分解表达不了碰撞的联合作用」自洽。
  三条可选路线（需要 A 拍板）：
  1. **如实写**：把 QMIX 放进「范式差异 / 局限」一节，主对比表用 IPPO/MAPPO/IC3Net/TarMAC/BAC。
  2. **单独给预算 + 奖励塑形**：给 QMIX 600k~1M 步，并把 `λ_c` 从 1.0 降到 0.25（或给编队误差
     加权重），报告里注明「QMIX 用了不同的奖励系数」。
  3. **换更弱的对照任务**：在 N=3、`safe_dist` 更大的设置上跑 QMIX，只用于说明范式差异。
  都不改代码，改配置即可；`docs/baselines.md` 与 PR 里的数就是这个默认配置跑出来的。
- QMIX 没有通信，`comm/bits`、`speak_rate` 恒为 0，`robust.csv` 整行平（见 §4.4 ②，有实测）；
  它检验的是范式差异，不参与「比特压缩」比较。
- QMIX 的 resume 只恢复网络/优化器/步数，不恢复 replay 与 ε 进度（见 §4.4 ①）。
- 本机 CPU 上 QMIX 约 9 ms / env_step（20k 步 3 分钟），200k 步单 run 约 30 分钟，
  与 PPO 的 200k run 同量级，可以进同一条队列。
- BenchMARL 路径只提供转换脚本，没有在本仓库 CI 里跑（依赖太重）；若组里决定以 BenchMARL
  为准，用它导出的 csv 走 `convert.py` 即可，列与作图脚本不变。
- 报告里「每组 5 个 seed」与仓库 `configs/experiment/main.yaml` 的 3 seed 不一致，以 E 的
  实验矩阵为准，本文不擅自改 `train.*`。

