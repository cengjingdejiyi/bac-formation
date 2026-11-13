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
```

Windows 没有 make，直接敲上面的命令即可；macOS/Linux 可用 `make smoke` 等目标。

## 目录（与开题报告 A.1 一致）

```
configs/            Hydra 风格配置（C 维护）
src/bacmarl/
  envs/             环境封装 FormationCommEnv、并行封装、指标（B）
  comm/             通信模块基类与 none/ic3net/tarmac/bac（A、D）
  models/           encoder / actor / critic（C）
  algos/            PPO、rollout 缓存、β 对偶（C、A）
  runners/          统一训练与评测入口（C、E）
  utils/            日志、随机、checkpoint（C、E）
baselines/          QMIX（BenchMARL）接入位（D）
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

## 修复与验证状态（2026-11-13）

| 检查 | 结果 |
|---|---|
| 环境、通信信息流、缓存与 GAE 测试 | 102 项测试；含有向边、静默/丢包、STE、发送预算及跨回合边界 |
| `python scripts/verify_transitions.py` | 首步与末步缓存奖励均与实际 step 返回值一致 |
| rollout 分段 | 环境与隐状态跨段延续；仅 done 环境重置 |
| CI | Python 3.11，验证可编辑安装、完整测试与真实 MPE2 的标准模块训练入口 |
| 正式参数 | `env=formation`：真实 MPE2，R_obs=R_comm=0.8；旧半径移至 `formation_legacy` |

旧版“训练跑通 / bits 接近预算”的结果受通信方向和奖励错位影响，不能证明通信有效或作为正式实验数据。修复依据、逐项核查和可复现命令见 [问题核查与修复](docs/issue-fixes.md)。短训练仅验证代码运行，不代表策略收敛。

## B 环境与标定交付（2026-11-13）

- **B 已完成**：真实 MPE2 编队后端、局部邻居/通信掩码、契约奖励与成功判定、批量自动 reset 修复及可复现标定已落地，详见 [B 交付与标定报告](docs/env-calibration.md)。`formation_calibrated` 现在仅是正式配置的兼容别名，避免维护第二份数值。选做的 `simple_spread` 尚未实现，入口会显式报错。

## 已知缺口（按负责人认领）

- **C**：PPO 目前按单步分段更新（缓存的 `h_prev` 逐帧传入，不做全轨迹 BPTT）；需要真实梯度截断时才扩展为分段 BPTT。
- **D**：`baselines/qmix_benchmarl/` 只有接入说明，QMIX 尚未落地。
- **E**：`scripts/make_figures.py` 读的是 `results/*/metrics.csv`，正式实验要在 `configs/experiment/main.yaml` 里补齐多 seed 矩阵后才能出统计图。

