# 正式实验进度看板（2026-11-22）

> 目标：`main` 组 6 方法 × 3 seed = **18 个 run**，每个 200k env_steps，单 run 15–20 分钟。
> 结果不入库，交接靠 zip → `scripts/merge_results.py` → `scripts/check_results.py`。

## 1. 逐 run 状态

| 方法 | 负责人 | seed 0 | seed 1 | seed 2 | 状态 |
|---|---|---|---|---|---|
| ippo | E | ✅ | ✅ | ✅ | **有效**，已交（issue #27） |
| qmix | E | ⚠️ | ⚠️ | ⚠️ | **作废**：旧配置价值发散，待按新配置重跑 |
| bac | C | — | — | — | 未交 |
| mappo | C | — | — | — | 未交 |
| ic3net | D | — | — | — | 未交 |
| tarmac | D | — | — | — | 未交 |

**有效进度 3/18**（ippo 三个 seed）。QMIX 三个 run 需要重跑约 1 小时：<br>
`python scripts/run_experiments.py --groups main --algos qmix --seeds 0 1 2 --execute --keep-going`

## 2. 我方本地已有的验证 run（不进 main 汇总）

| run | 位置 | 用途 |
|---|---|---|
| BAC 200k value_norm A/B | `results/val_vnorm/` | 证明 `value_norm` 修复（详见 `results/bac_diagnosis.md`） |
| BAC 200k 旧口径基线 | `results/_legacy_scale_bug/main_1seed_scale_bug/` | 问题对照，**不得进图** |
| QMIX 200k 修复验证 | `results/qmix_fix200k/` | 证明发散修复（`loss/td` 2.18、`q_tot` 9.67） |

## 3. 交接与验收（每人交之前先自检）

```bash
# 1) 自己那台机器上
python scripts/check_results.py --results results/main        # 全 OK 再打包
# 2) 打包（只打 run 目录，不要带 _logs / figures）
# 3) 汇总机（A 这边）
python scripts/merge_results.py --dest results/main --src <zip 或目录> [--overwrite]
python scripts/check_results.py --results results/main
python scripts/make_figures.py --results results/main --out results/figures
```

`check_results` 现在会拦四类问题：缺文件/缺列（含 `train/return_all`、`train/formation_err_mean`）、
`loss/value` 均值 >50（value_norm 没生效）、QMIX 配置缺 `td_huber`/`target_clip`（旧发散配置）、
QMIX 的 `qmix/q_tot` >30 或 `loss/td` >1000（价值发散）。

## 4. 剩余工作量估算

| 事项 | 负责人 | 预计 |
|---|---|---|
| qmix × 3 seed 重跑 | E | 1 h |
| bac × 3、mappo × 3 | C | 2 h |
| ic3net × 3、tarmac × 3 | D | 1.5 h |
| 合并 18 run + 出图 + 汇总表 | A/E | 0.5 h |
| 结题报告 + 答辩 PPT | 全员 | 待排 |

## 5. 已知待决（需要拍板，不是 bug）

1. 是否加跑 `collision_coef ∈ {0.2, 0.5}` 对照（B 标定层）；
2. 是否把 `eval_gate_mode` 默认从 `threshold` 翻成 `sample`（影响 bits 读法，报告要注明）；
3. encoder 是否加 `LayerNorm`（会改 ckpt 形状，需单独 A/B）；
4. 是否加跑 500k–1M 冲 `success_rate`（当前最好 ferr 0.686，阈值 0.09，差约 7 倍）。
