# B 任务交付：真实环境与参数标定

更新：2026-11-14，处理 [Issue #4](https://github.com/cengjingdejiyi/bac-formation/issues/4)。依据为 `docs/contracts.md` 第 2、5 节和报告 §1.4、§3.1。

正式配置统一为 `formation.yaml`：真实 MPE2、R_obs=R_comm=0.8、动态朝向、完成阈值 0.09、连续保持 10 步、时限 100 步。`formation_calibrated` 继承正式配置；`formation_legacy` 显式保留旧半径、固定朝向及 0.05 阈值，避免混用历史结果。

## 实现范围

- `FormationCommEnv` 对接 `mpe2.simple_formation_v1.parallel_env`，使用真实阻尼质点与碰撞物理；`auto` 在 MPE2 可用时实际运行该后端。
- 观测仅含自身速度、中心相对位置和按距离排序的局部邻居相对位置；全局 Critic 状态和有向通信掩码保持原接口。
- 每步按报告 §1.4，以相对中心极角 `[0, 2π)` 最小的无人机锚定槽位朝向。奖励、完成判定、`info.heading` 和标定的最大槽位距离使用同一朝向。规则编队整体旋转后仍能完成。显式 `heading_mode=fixed` 使用配置 `heading`，用于历史对照。
- 团队奖励为 `-(未裁剪的槽位平均匹配距离 + collision_coef * 不安全机对数)`，不混入通信成本。
- 速度上限限制向量模长。完成要求所有机体与匹配槽位的距离均不超过阈值，并连续保持 H 步。
- 自动 reset 保留末步 reward、done、ep 和 terminal_step，后续回合延续 RNG；环境支持 close 和上下文管理器。
- `scene=spread` 是选做扩展，显式报错，不能拿编队结果冒充覆盖实验。

## 默认脚本控制器

`ScriptedController`（标定策略名 `scripted`）逐行处理观测：径向收敛到圆环，再均衡可见邻居的顺/逆时针角间距，并用自身速度反馈制动。没有可见邻居的一侧以 `2π/N` 为参考。使用 0.02 控制死区、0.2 速度反馈和最大 0.5 弧度的单次目标角修正。

每个动作仅依赖该机的观测行和有效邻居掩码，不读取其他观测行、`info.pos`、全局 state、无人机编号或全局匈牙利指派。目标槽位不作为新增 Actor 观测。

- `radial`：旧 `RadialController`，只做径向收敛，保留作失败对照。
- `slot_oracle`：从全局位置一次性做匈牙利指派，采用初始 `info.heading`，以 0.2 速度反馈和 0.03 死区控制。仅作物理可行性诊断，不能作为分布式执行或学习模型性能的证据。
- `hover`、`random`：继续提供悬停和固定种子随机动作诊断。

脚本针对正式 MPE2 动力学标定；mock 仅作契约冒烟，不承诺相同控制效果。

## 可复现标定

Python 3.11 安装项目后，在仓库根目录运行：

```bash
pip install -e ".[dev,env]"
python scripts/calibrate_env.py --env-config formation --agents 4 6 8 --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 --policies scripted radial slot_oracle --out results/calibration/b-followup-final
python scripts/calibrate_env.py --env-config formation --agents 4 6 8 --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 --policies scripted --success-dist 0.05 --out results/calibration/b-followup-strict
```

正式标定为 N=4/6/8、seed=0–19、3 种控制方式，共 180 回合，每回合 100 步；另跑同样 60 个局部脚本回合的严格阈值对照。合计 240 回合、24,000 个环境步。

输出 `episodes.csv`（逐回合）、`summary.csv`（均值与总体标准差）、`config.json`（完整参数、种子与实际版本）。按团队约定，原始结果不入库，可用上述命令重现。

本次实际环境：Python 3.11.14、NumPy 2.4.6、SciPy 1.17.1、MPE2 1.1.1、PettingZoo 1.27.0。

## 结果与阈值选择

下表全部使用动态朝向、0.09 完成阈值、连续 10 步要求。完成率指时限内曾完成；误差是最后一步的匹配平均距离。

| N | 控制方式 | 完成率 | 终止误差（均值 ± std） | 每回合不安全机对累计数 |
|---|---|---|---|---|
| 4 | scripted | 20/20 | 0.0267 ± 0.0059 | 0.0 |
| 4 | radial | 0/20 | 0.3151 ± 0.0965 | 25.1 |
| 4 | slot_oracle | 20/20 | 0.0274 ± 0.0075 | 0.35 |
| 6 | scripted | 20/20 | 0.0309 ± 0.0060 | 1.55 |
| 6 | radial | 0/20 | 0.2788 ± 0.0657 | 67.7 |
| 6 | slot_oracle | 20/20 | 0.0253 ± 0.0051 | 2.05 |
| 8 | scripted | 20/20 | 0.0283 ± 0.0057 | 12.35 |
| 8 | radial | 0/20 | 0.2330 ± 0.0473 | 136.85 |
| 8 | slot_oracle | 20/20 | 0.0267 ± 0.0041 | 6.0 |

同一局部控制器在 0.05 阈值下仅 11/20、5/20、2/20 回合完成。平均误差小不等于所有机体都能持续满足最大距离阈值。0.075 与 0.08 在扩展种子中仍有未完成回合，因此最终采用 0.09，没有降低保持步数或延长时限。这是所列 60 个种子/规模组合的实测结果，不是任意初态的收敛保证。

参数选择的几何边界：

- `r_f=0.5` 时，相邻槽位距离为 `2*r_f*sin(pi/N)`。正式 R_obs=R_comm=0.8 覆盖 N=4/6/8 的理想相邻槽位；每机理想通信邻居数分别为 2/2/3（K=3）。
- 对最大的 N=8，相邻距离为 0.382683。两架机各偏离槽位最多 0.09 时，距离仍至少为 `0.382683 - 2*0.09 = 0.202683 > d_safe=0.2`，N=4/6 余量更大。这个完成状态的边界不代表过渡过程没有碰撞，也不自动适用于更大的 N。
- 保持 `max_speed=1.0, max_steps=100, success_hold=10, collision_coef=1.0`；碰撞按每个不安全机对计价。没有声称惩罚系数已最优。

## 验收与实验边界

```bash
python scripts/smoke.py
python -m pytest -q tests
python -m compileall -q src scripts tests
python scripts/verify_transitions.py --backend mock
python scripts/verify_transitions.py --backend mpe2
python -m bacmarl.runners.train env=formation algo=bac train.total_env_steps=32 train.n_envs=2 train.rollout_len=4 train.ppo_epochs=1 train.minibatch=8 train.hidden=16 env.max_steps=6 experiment.name=review_check seed=0
```

覆盖旋转规则编队、显式固定朝向对照、局部脚本的信息边界及多种子完成判定；原有后端、速度、终止转移和 RNG 测试继续执行。

朝向和阈值默认值变更已写入契约、配置及提交说明。所有学习方法必须采用相同指标配置；旧固定朝向、0.05 阈值的结果不能与新结果混用，旧模型需要按明确配置重新评测。旧训练结果受通信/奖励错位影响的边界继续见 `docs/issue-fixes.md`。

局部脚本在 N=6/8 仍有过渡碰撞，脚本完成率不能证明学习策略收敛、安全避碰或通信抗噪能力。选做的 simple_spread 仍未交付。
