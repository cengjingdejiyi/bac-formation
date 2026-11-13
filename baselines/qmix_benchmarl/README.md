# QMIX 基线接入位（负责人 D）

用 BenchMARL(TorchRL) 跑 QMIX，把日志转成本仓库 `metrics.csv` 格式。

## 待做

1. `python -m pip install benchmarl`（或在独立 conda 环境装，避免污染主环境）。
2. 用同一套环境参数（`configs/env/formation.yaml`）跑 QMIX。
3. 导出列：`env_steps, train/return, train/success_rate, train/collisions, comm/bits`（QMIX 无通信，`comm/bits=0`）。
4. 写 `convert.py`：BenchMARL csv → `results/main/qmix_.../metrics.csv`，列名与本仓库一致。

> 注意：QMIX 的动作空间需为离散（MPE2 默认离散 5 动作，正好匹配）。
