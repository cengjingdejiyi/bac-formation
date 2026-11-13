# bac-formation

带宽受限下多无人机分布式编队通信学习（BAC-MAPPO）——课程设计代码骨架。

上层文档：`docs/contracts.md`（接口与规范，唯一事实来源）、开题报告附录 A.1–A.8。

## 快速开始

```bash
pip install -r requirements.txt      # 或 pip install -e ".[dev,env]"
python scripts/smoke.py              # 契约冒烟：不装 torch 也能过
python -m pytest -q tests            # 形状与缓冲测试
python -m bacmarl.runners.train env=formation algo=bac env.n_agents=6 algo.msg_bits=4 seed=0
```

Windows 没有 make，直接敲上面四条命令即可；macOS/Linux 可用 `make smoke` 等目标。

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

## 当前验证状态（2026-11-13，本机 Python 3.13 + torch 2.15.0+cpu）

| 检查 | 结果 |
|---|---|
| `python scripts/smoke.py` | 18 项 PASS，`SMOKE OK` |
| `python -m pytest -q tests` | 16 passed（无 skip） |
| `python -m compileall src scripts tests` | 退出码 0 |
| 四种方法 32 步小规模训练 | `bac` / `ippo` / `tarmac` / `ic3net` 全部 `[DONE]` |
| BAC 128 步训练 | `comm/bits ≈ 1.92–2.13`，与 `budget_bits=2.0` 吻合；β 在 0.096–0.102 间摆动（对偶在起作用） |
| `runners/evaluate.py` | 生成 `robust.csv` |

## 已知缺口（按负责人认领）

- **B**：`backend="mpe2"` 仅留适配位并显式抛 `NotImplementedError`。入口与实测形状已写在 `docs/mpe2-notes.md`（`mpe2.simple_formation_v1`：obs 单机 6 维、无队友观测、`Discrete(5)`、共享奖励、`info` 为空、地标即编队中心），照着对齐即可。
- **C**：PPO 目前按单步分段更新（缓存的 `h_prev` 逐帧传入，不做全轨迹 BPTT）；需要真实梯度截断时才扩展为分段 BPTT。
- **D**：`baselines/qmix_benchmarl/` 只有接入说明，QMIX 尚未落地。
- **E**：`scripts/make_figures.py` 读的是 `results/*/metrics.csv`，正式实验要在 `configs/experiment/main.yaml` 里补齐多 seed 矩阵后才能出统计图。

