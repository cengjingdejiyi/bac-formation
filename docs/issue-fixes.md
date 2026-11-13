# 问题清单核查与修复（2026-11-13）

核查依据：交接清单、三个复现脚本、实际代码与 `docs/contracts.md`。清单中的结论和修法均逐条验证，不能把复现脚本中手写的旧逻辑当成当前训练器。

## 核查结论

| 编号 | 核查结果与处理 |
|---|---|
| P0-1 | 确认。修复前 BAC/IC3Net/TarMAC 的邻居影响均为 0；修复后原复现脚本三者均非 0。根治信道 sender/receiver 轴，NumPy 与 torch 一致；定向噪声、静默/丢包及全掩码行一起修复。 |
| P0-2 | 确认。新增真实训练共用的 `collect_rollout`，缓存动作前 obs 与当前 step 的 reward/done，首步不再缓存 reset 零奖励，末步奖励保留。 |
| P1-3 | 消费端问题确认，修法需纠正。自动 reset 返回新回合 obs 与终止 reward 的接口保持；缓存使用 obs_t/action_t/reward_{t+1}/done_{t+1}，不能拿终止 obs 替换 obs_t。终止帧仍保留供诊断。 |
| P1-4 | “代码与报告分别为机对数/0-1”不成立。本地报告 §1.4 与契约均是 `sum_{i<j} indicator`，即不安全机对数。保留每对 λ_c=1.0，在契约中明确计数公式。未声称该系数最优。 |
| P2-5 | 两份完整参数的风险确认。正式 `formation.yaml` 与 `EnvConfig` 统一真实后端、两个半径 0.8；`formation_calibrated` 只继承正式配置，旧值移至明确标记的 `formation_legacy`。报告 §3.1 实际写的是“实现阶段标定并写入配置”，未写死 0.4，因此没有该数字需要直接改写 PDF。 |
| P2-6 | 确认。CI 先安装 CPU torch，再可编辑安装 `.[dev,env]`，然后完整测试和 32 步真实环境模块入口训练。 |
| P2-7 | 确认。训练只在开始 reset；rollout 段间保持 batch/h，仅真正结束的环境自动 reset 并清零 h。回合长度可以超过单段采样长度。 |
| P3-8 | 确认原 README 混用了不同机器记录。开发基线与 CI 统一说明为 Python 3.11；本轮在独立 3.11 环境验证，3.14 结果单列为兼容检查。运行依赖版本按实际环境记录，不虚构未运行版本已验证。 |
| P3-9 | 确认原接口测试不能证明消息有效。新增有向边手算、邻居影响、静默、掩码/Top-K、丢包归一化、NumPy/torch 一致性、STE 邻居梯度、广播预算，以及转移、GAE、RNN 与噪声重放语义测试。 |

## 两个关键修法

信道统一 `edge[b, receiver_i, sender_j, :] = msgs[b, j, :]`。
不在聚合后简单 transpose：因为掩码与噪声必须一开始就作用在同一条 j→i 有向边上。
仅允许、发言且未丢包的边参与归一化；没有有效边时通信上下文为 0，线性层 bias 不能造成伪消息。
每个并行环境保留自己的信道概率，预算按 `bits.mean()` 统计，一次广播不按边数重复收费，丢包也不退款。

采样顺序为当前 obs 决策 → step → 缓存当前 obs/action 与这次 step 的 reward/done → 续用 next obs。
done 同时切断 RNN 状态与 GAE 跨回合传播；未结束的段末以最新 next state 的 value 自举。
训练日志 `train/return` 改为最近 100 个已完成回合的真实团队回报，`train/step_reward` 单独记录平均单步奖励。
尚无回合完成时回合类指标为空语义（CSV 记为 NaN），不能以 0 冒充评测结果。

## 验证

```bash
pip install -e ".[dev,env]"
python scripts/smoke.py
python -m pytest -q tests
python -m compileall -q src scripts tests
python scripts/verify_transitions.py --backend mock
python scripts/verify_transitions.py --backend mpe2
python -m bacmarl.runners.train env=formation algo=bac train.total_env_steps=32 train.n_envs=2 train.rollout_len=4 train.ppo_epochs=1 train.minibatch=8 train.hidden=16 env.max_steps=6 experiment.name=review_check seed=0
```

`tests/test_comm_semantics.py` 的消息权重与有向边可手算，不依赖随机初始化“碰巧有变化”。
`tests/test_rollout.py` 让奖励每步变化、两个环境在不同步结束，手算跨回合 GAE，并验证存储的 gate/h/noise 精确复现旧策略 log probability。
共 102 项测试；真实 MPE2 与 torch 相关语义测试不跳过。
Python 3.11.14 的干净环境（torch 2.15.0+cpu、numpy 2.4.6、scipy 1.17.1、MPE2 1.1.1）可编辑安装成功，102 项测试无跳过，冒烟、编译与依赖一致性检查通过。
另在 Python 3.14.2 / torch 2.11.0+cpu 上通过同样 102 项测试，作为额外兼容检查，未混称为同一套运行环境。
Python 3.11 基线下 BAC/IPPO/IC3Net/TarMAC 各完成 32 个真实 MPE2 环境步，损失与回报有限；每种方法记录 4 个完整回合，回合长度均为 6（大于 rollout 长度 4），验证了回合没有被采样段截断。
`scripts/verify_transitions.py` mock 首行缓存/真实奖励均为 -0.713176，MPE2 首行均为 -0.880190；两次终止的末步奖励也都保留。

交接包 `02_reward_shift.py` 自己写死了“先取旧 batch reward 再 step”，没有调用训练器；修复代码后直接重跑它仍会演示旧错误。
交接包 `03_autoreset_pairing.py` 只观测返回字段来源，不能单独证明转移缓存错误。
因此保留原复现文件作历史证据，验收以实际训练共用采样器的测试和 `verify_transitions.py` 为准。

## 实验与报告边界

历史训练 checkpoint/曲线受到两个 P0 的影响，不可作为正式性能依据。环境脚本的几何/动力学标定不依赖通信学习，这部分仍只说明任务可行性，不能证明分布式策略、避碰或抗噪性能。
正式半径修改已写进提交与 PR，团队正式实验应统一使用 `env=formation`。

2026-11-14 按 Issue #4 完成该差异的后续处理：先更新环境契约，再将正式朝向改为报告 §1.4 的动态最小极角锚定；显式 `heading_mode=fixed` 保留历史对照。默认脚本加入局部角间距均衡与速度反馈，并以 N=4/6/8、20 个种子重标完成阈值为 0.09，保持 10 步和 100 步时限。60 个局部脚本回合全部完成；0.05 严格阈值的结果另列，不能混用。原始“保留固定朝向、等待报告对齐”结论已被此次修复替代。详见 `docs/env-calibration.md`，未修改外部 PDF 或旧版 `main.tex`。
`simple_spread` 和 QMIX 仍属于既有未实现范围，本轮没有宣称完成。
