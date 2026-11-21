# bac-formation

带宽受限下多无人机分布式编队通信学习（BAC-MAPPO）——课程设计代码骨架。

上层文档：`docs/contracts.md`（接口与规范，唯一事实来源）、开题报告附录 A.1–A.8。

推荐开发基线为 **Python 3.11**，与 CI 一致；其他版本作为兼容性验证单独记录，不把不同机器的 Python/torch 版本写成同一套环境。依赖约束统一放在 `pyproject.toml` / `requirements.txt`，标定输出同时保存实际运行版本。

## 快速开始

```bash
pip install -r requirements.txt      # 或 pip install -e ".[dev,env]"
pip install -e . --no-deps           # 使用 requirements.txt 安装后注册 src 包
python scripts/smoke.py              # 契约冒烟：不装 torch 也能过
python -m pytest -q tests            # 形状与缓冲测试
python scripts/verify_transitions.py # 实测当前训练采样器的奖励/终止对齐
python -m bacmarl.runners.train env=formation algo=bac env.n_agents=6 algo.msg_bits=4 seed=0
python -m bacmarl.runners.qmix env=formation algo=qmix seed=0   # QMIX 基线（值分解）
```

基线（IC3Net 式 / TarMAC 式 / QMIX）的摆位、调参记录与复现命令见 [对比基线交付说明](docs/baselines.md)。
运行名与通信量口径（Issue #11）由 `src/bacmarl/utils/naming.py` 统一实现，别再各写一份。

Windows 没有 make，直接敲上面的命令即可；macOS/Linux 可用 `make smoke` 等目标。

正式实验与出图（一次排 6 个方法 × N 个 seed，QMIX 自动派发到 `runners/qmix.py`）：

```bash
python scripts/run_experiments.py --groups main --algos bac ippo mappo ic3net tarmac qmix --seeds 0 1 2 --dry-run   # 只排班，先核对名字
python scripts/run_experiments.py --groups main --algos bac ippo mappo ic3net tarmac qmix --seeds 0 1 2 --execute
python scripts/make_figures.py           # 读 results/ 出 4 张图
```

## 目录（与开题报告 A.1 一致）

```
configs/            Hydra 风格配置（C 维护）
src/bacmarl/
  envs/             环境封装 FormationCommEnv、并行封装、指标（B）
  comm/             通信模块基类与 none/ic3net/tarmac/bac（A、D）
  models/           encoder / actor / critic（C）
  algos/            PPO、rollout 缓存、β 对偶（C、A）；QMIX 与回放（D）
  runners/          统一训练与评测入口：train / evaluate（C、E）、qmix（D）
  utils/            日志、随机、checkpoint（C、E）；命名口径 naming（D）
baselines/          QMIX（BenchMARL）接入说明与日志转换（D）
scripts/            冒烟、作图、批量脚本（E、F）
tests/              每个接口一条形状测试
results/            实验输出（不入库）
```

## 所有权（改之前先看 docs/contracts.md）

备份关系与开题报告第四节一致：A↔C、B↔E、D↔F。

| 模块 | 负责人（角色） | 备份（角色） |
|---|---|---|
| BAC 模块、β 对偶、统筹 | A | C |
| 环境与参数标定 | B | E |
| 训练框架与 PPO | C | A |
| 对比基线（IC3Net/TarMAC/QMIX） | D | F |
| 实验与评测 | E | B |
| 可视化与文档 | F | D |

## 阶段

W1 环境+框架+接口冻结 → W2–3 方法跑通 → W4–5 正式实验 → W6 结题。

## 修复与验证状态（2026-11-21）

| 检查 | 结果 |
|---|---|
| 环境、通信信息流、缓存与 GAE 测试 | **257 项测试**：原 124 项覆盖旋转编队、局部脚本多种子完成、有向边、静默/丢包、STE、发送预算及跨回合边界；后续补入 6 算法命名口径、QMIX 学习、评测失败路径与出图 label 分组 |
| `python scripts/verify_transitions.py` | 首步与末步缓存奖励均与实际 step 返回值一致 |
| rollout 分段 | 环境与隐状态跨段延续；仅 done 环境重置 |
| CI | Python 3.11，验证可编辑安装、完整测试与真实 MPE2 的标准模块训练入口（QMIX 冒烟与 ic3net 命名检查也已纳入） |
| 正式参数 | `env=formation`：真实 MPE2，R_obs=R_comm=0.8；旧半径移至 `formation_legacy` |

旧版“训练跑通 / bits 接近预算”的结果受通信方向和奖励错位影响，不能证明通信有效或作为正式实验数据。修复依据、逐项核查和可复现命令见 [问题核查与修复](docs/issue-fixes.md)。短训练仅验证代码运行，不代表策略收敛。

## B 环境与标定交付（2026-11-13）

- **B 已完成**：真实 MPE2 编队后端、局部邻居/通信掩码、契约奖励与成功判定、批量自动 reset 修复及可复现标定已落地，详见 [B 交付与标定报告](docs/env-calibration.md)。`formation_calibrated` 现在仅是正式配置的兼容别名，避免维护第二份数值。选做的 `simple_spread` 尚未实现，入口会显式报错。
- **Issue #4 后续修复（2026-11-14）**：正式环境对齐报告的动态朝向，升级仅使用逐机局部观测的脚本控制器；完成阈值重标为 0.09，保持连续 10 步和 100 步时限。N=4/6/8 各 20 个种子全部完成；原径向控制器 0/60。固定朝向、0.05 阈值及旧半径保留在 `formation_legacy`，旧实验指标须与新口径区分。

## D 基线交付（2026-11-21）

- **QMIX 已落地**：`algos/qmix.py`（个体 Q + 单调混合）、`algos/replay.py`（回放 + n 步回报）、`runners/qmix.py`（训练/评测入口，与 PPO 入口共用配置、命名、评测信道与日志列）。它跑同一套 `FormationCommEnv`，不再走 BenchMARL 的外部环境；BenchMARL 路径保留为日志转换（`baselines/qmix_benchmarl/convert.py`）。
- **Issue #11 已修**：运行名按通信模块真实的 `bits_per_msg` 计算，IC3Net/TarMAC 从 `L0/B1e9` 变为 `L1024/Binf`，无预算写 `Binf`。命名只有一份实现（`utils/naming.py`），并有「名字里的 L == 模块上报 bits」的防漂移测试。详见 [对比基线交付说明](docs/baselines.md)。

## 已知缺口（按负责人认领）

- **C**：PPO 目前按单步分段更新（缓存的 `h_prev` 逐帧传入，不做全轨迹 BPTT）；需要真实梯度截断时才扩展为分段 BPTT。
- **D**：QMIX 已落地并能在玩具任务上收敛（`tests/test_qmix_learning.py`），但在本奖励、本预算下贪心策略会滑向「散开避碰」：编队误差涨到 ~8、`eval/return` −425，而随机策略是 −115。原因是碰撞项按机对计价、且单调分解表达不了碰撞的联合作用（QTRAN/QPLEX 就是在松绑这条约束）。取舍与三条可选路线见 [对比基线交付说明](docs/baselines.md) §5.2/§8，需要 A 拍板。
- **E**：正式实验的排班、评测与出图链路已打通（`--algos` 算法轴 + QMIX 派发、评测目录/ckpt 校验、`--drop-p`/`--flip-p` 口径、出图 label 带实验组），当前缺的是**跑数**：`results/` 里只有 ≤2000 步的验证跑，需按

  ```bash
  python scripts/run_experiments.py --groups main --algos bac ippo mappo ic3net tarmac qmix --seeds 0 1 2 --execute
  ```

  跑完再 `python scripts/make_figures.py` 出 4 张图。seed 数与算力预算待定（单机 CPU 实测 200k 步约 25 min/run，18 run 约 7–8 h）。


