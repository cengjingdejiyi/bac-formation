# B 任务交付：真实环境与参数标定

日期：2026-11-13。实现与验收依据为 `docs/contracts.md` 第 2、5 节。

修复后正式配置统一为 `formation.yaml`（感知/通信半径均为 0.8，真实 MPE2）。
`formation_calibrated` 仅继承它，保留旧命令兼容；`formation_legacy` 才是原半径对照。
本页脚本控制器数据只验证几何与动力学可行性；旧版通信学习模型与训练曲线因奖励错位/通信方向错误不能作为正式实验依据，详见 [问题核查与修复](issue-fixes.md)。

## 实现范围

- `FormationCommEnv` 对接 `mpe2.simple_formation_v1.parallel_env`，使用真实阻尼质点与碰撞物理；`auto` 在 MPE2 可用时实际运行该后端。
- 统一生成自身速度、中心相对位置、按距离排序的局部邻居观测、全局 Critic 状态及有半径/容量约束的有向通信掩码。
- 奖励严格为 `-(固定朝向槽位平均匹配距离 + collision_coef * 不安全机对数)`，各机相同。不使用原生 MPE2 的动态朝向、裁剪奖励，不混入通信成本。
- 速度上限限制向量模长；成功要求每架机都在匹配槽位阈值内连续保持 H 步，支持成功后提前终止。
- 批量环境自动 reset 时保留末步 reward、done、ep 和 `terminal_step`；新回合延续 RNG，既可复现，也避免每回合重复相同初始状态。
- 单/批环境支持 `close()` 和 `with`。非法配置、动作、安装损坏及未实现的 `scene=spread` 显式报错。

## 可复现标定

安装项目后，在仓库根目录运行：

```bash
pip install -e ".[dev,env]"
python scripts/calibrate_env.py --env-config formation_legacy --out results/calibration/legacy
python scripts/calibrate_env.py --env-config formation --out results/calibration/formal
```

每个配置运行 N=4/6/8、seed=0/1/2/3/4、4 种控制方式，共 60 回合，每回合 100 步。
两份配置合计 120 回合、12,000 个环境步。
输出 `episodes.csv`（逐回合）、`summary.csv`（均值与总体标准差）、`config.json`（完整参数、种子与版本）。
按团队约定，结果文件不入库。

本次实测：Python 3.14.2、numpy 2.5.4、scipy 1.18.1、mpe2 1.1.1、pettingzoo 1.27.0。

控制方式：

- `hover`：始终动作 0；有物理重叠时仍可能受到碰撞力。
- `random`：固定种子的均匀随机动作。
- `radial`：现有局部观测脚本控制器，只把机体引向圆环，不做角向槽位分配。
- `slot_oracle`：从全局位置一次性计算匈牙利分配目标，再以离散位置/速度反馈控制；系数 0.2，死区 0.03。**只验证任务可行性，使用全局信息，不能作为分布式执行或训练性能的证据。**

## 结果与参数选择

两份配置仅感知/通信半径及后端选择策略不同。这些脚本不依赖通信，且 radial 不使用邻居槽位，
因此两份配置的物理轨迹和下表任务指标一致；变化的是邻居可达性。

| N | 控制方式 | 完成率（曾连续保持 10 步） | 终止编队误差（均值 ± std） | 每回合不安全机对累计数 |
|---|---|---|---|---|
| 4 | hover | 0/5 | 0.6990 ± 0.1012 | 0.0 |
| 4 | random | 0/5 | 1.3871 ± 0.3270 | 2.4 |
| 4 | radial | 0/5 | 0.3127 ± 0.1392 | 22.0 |
| 4 | slot_oracle | 5/5 | 0.0262 ± 0.0040 | 0.8 |
| 6 | hover | 0/5 | 0.6443 ± 0.1792 | 60.0 |
| 6 | random | 0/5 | 1.2521 ± 0.2937 | 13.4 |
| 6 | radial | 0/5 | 0.2921 ± 0.0704 | 52.6 |
| 6 | slot_oracle | 5/5 | 0.0237 ± 0.0022 | 3.6 |
| 8 | hover | 0/5 | 0.5831 ± 0.0956 | 100.8 |
| 8 | random | 0/5 | 1.1347 ± 0.2255 | 27.6 |
| 8 | radial | 0/5 | 0.2703 ± 0.0330 | 168.4 |
| 8 | slot_oracle | 5/5 | 0.0234 ± 0.0021 | 8.8 |

`r_f=0.5` 时相邻槽位距离为 `2*r_f*sin(pi/N)`：

| N | 相邻槽位距离 | 历史 R_comm=0.4：理想接收邻居数 | 正式 R_comm=0.8、K=3：理想接收邻居数 | 正式 R_obs=0.8、n_obs=4：理想观测邻居数 |
|---|---|---|---|---|
| 4 | 0.7071 | 0 | 2 | 2 |
| 6 | 0.5000 | 0 | 2 | 2 |
| 8 | 0.3827 | 2 | 3 | 4 |

正式实验统一用 `env=formation`：

- `backend=mpe2`：缺依赖时立即失败，避免把 mock 结果混入正式实验。
- `obs_radius=comm_radius=0.8`：覆盖三种 N 的相邻槽位；仍保持半径限制及 K=3 容量限制。
- 保持 `r_f=0.5, d_safe=0.2, v_max=1.0, T_max=100, δ=0.05, H=10`。槽位距离与安全距离有余量，且可行性控制器在全部 15 个种子/规模组合中满足完成条件。
- 保持 `collision_coef=1.0` 和原信道区间，每个不安全机对各计一次惩罚；控制器存在过渡碰撞，尚不能声称 λ_c 已最优或策略安全，也不能据此证明通信收益/抗噪能力。

本次将 `formation.yaml` 与 `EnvConfig` 默认值统一为真实 MPE2、R_obs=R_comm=0.8；默认变更在修复提交与 PR 说明中列明，团队正式实验须采用同一配置。
局部径向控制器不能维持角向间隔，失败不能推导任务不可解；需要学习到的协作策略补足角向控制与避碰。
`simple_spread` 通用性实验属于选做扩展，尚未实现，禁止将其入口当作编队场景运行。

## 验收

```bash
python scripts/smoke.py
python -m pytest -q tests
python -m compileall -q src scripts tests
python scripts/verify_transitions.py
python -m bacmarl.runners.train env=formation algo=bac train.total_env_steps=32 train.n_envs=2 train.rollout_len=4 train.ppo_epochs=1 train.minibatch=8 train.hidden=16 env.max_steps=6 experiment.name=review_check seed=0
```

真实 MPE2 的回归测试覆盖 N=4/6/8，逐步对照原生物理、位置/速度顺序、局部信息与补零、
固定半径/朝向奖励、速度模长、连续保持/提前终止、终止奖励/回合统计和独立可复现的自动 reset。
原 67 项环境/接口测试已扩展为通信信息流与真实转移语义测试；当前验收结果见 [issue-fixes.md](issue-fixes.md)。
CI 安装项目及 MPE2/scipy，运行测试和真实环境的 32 步训练入口；rollout 长 4、回合长 6，显式覆盖跨段及自动 reset。
所有短训练只验证采样与更新，不代表训练收敛。
