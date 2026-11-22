# 正式实验进度看板（2026-11-22）

> 目标：`main` 组 6 方法 × 3 seed = **18 个 run**，每个 200k env_steps，单 run 15–20 分钟。
> 结果不入库，交接靠 zip → `scripts/merge_results.py` → `scripts/check_results.py`。

## 0. 开跑前必做（C / D / E 都一样）

```bash
git pull                      # 必须到 d51987e5 之后
python scripts/preflight.py   # 全 PASS 才开跑
```

`preflight` 会核对代码版本（`value_norm`、QMIX 稳定化参数）、主配置（`value_norm=true`、
`reward_norm=false`、正式环境参数、200k 步）、QMIX 配置（`td_huber`/`target_clip`/`lr`）、
真实后端（`backend=mpe2`）和 `results/` 可写。**任何 FAIL 都不要开跑**，按提示处置后重跑；
否则跑出来的数据在汇总阶段会被 `check_results` 直接判废，白烧 1.5–2 小时。

废数据的典型症状（出现即停）：`backend=mock`、`loss/value` 均值 >50、`loss/td` 破千、
缺 `train/return_all`、IC3Net/TarMAC 运行名写成 `L0/B0.0`。完整表见 `docs/RUNBOOK.md` §0.1。

## 1. 逐 run 状态

| 方法 | 负责人 | seed 0 | seed 1 | seed 2 | 状态 |
|---|---|---|---|---|---|
| ippo | E | ✅ | ✅ | ✅ | **有效**，已入库（E 的 Release `results-2026-11-22-e-ippo-qmix`） |
| qmix | E | ✅ | ✅ | ✅ | **有效**（新配置重跑）：ferr 1.276±0.625，全表最弱 |
| bac | C | ✅ | ✅ | ✅ | **有效**，已交（Release `results-2026-11-22-c-bac-mappo`，SHA256 校验一致） |
| mappo | C | ✅ | ✅ | ✅ | **有效**，同上 |
| ic3net | D | ✅ | ✅ | ✅ | **有效**，已交（D 的 fork Release `results-2026-11-22-d-ic3net-tarmac`，SHA256 校验一致） |
| tarmac | D | ✅ | ✅ | ✅ | **有效**，同上 |

**有效进度 18/18**（六方法各 3 seed 全部入库）。`check_results` 对 `results/main` 的 18 个 run：**18/18 通过**。

六方法汇总数（末次评测 @194560，3 seed 聚合）：

| 方法 | eval/return | eval/formation_err | eval/collisions | comm/bits | comm/speak_rate |
|---|---|---|---|---|---|
| BAC（L4/B2.0） | −38.4 ± 2.9 | **0.342 ± 0.022** | **2.08 ± 0.51** | 1.81 ± 0.04 | 0.45 |
| TarMAC（L1024/B∞） | −38.4 ± 2.7 | 0.317 ± 0.047 | 3.12 ± 1.27 | **1024**（恒发） | 1.00 |
| MAPPO（无通信） | −50.8 ± 26.3 | 0.365 ± 0.149 | 12.67 ± 12.54 | 0 | 0 |
| IPPO（无通信） | −43.0 ± 6.7 | 0.385 ± 0.073 | 3.96 ± 1.44 | 0 | 0 |
| QMIX（无通信，新配置） | −115.6 ± 45.8 | 1.276 ± 0.625 | 20.38 ± 16.22 | 0 | 0 |
| IC3Net（L1024/B∞） | −45.9 ± 10.0 | 0.436 ± 0.133 | 2.83 ± 0.95 | 511.4 ± 60.9 | 0.48 |

**口径提醒（汇总出图必读）**

- `speak_rate` = “这一步到底发没发消息”：有门控头的（BAC/IC3Net）取门控概率；
  无门控但恒发的（TarMAC，1024 bit）记 **1.0**；完全不发的（IPPO/MAPPO）记 **0**。
  这条口径在 PR #33 首版修过硬：当时按 `gate_logits is None → 0` 一刀切，会把 TarMAC
  写成 `bits=1024` 却 `speak_rate=0`，属于新的自相矛盾；已在 #34 改成按“发没发消息”判。
- **D 交的 tarmac `robust.csv` 里 `speak_rate` 是 0.0**，那是 #33 首版口径留下的痕迹（训练日志
  里同一 run 是 1.0）。数据本身没问题，出图/写表时 TarMAC 的 speak 统一按 1.0 处理，
  不必重跑；要刷新这一列重跑一次 `evaluate --drop-p` 即可（实测 tarmac s0 = 1.0000）。
- 成本对比只能横向看 `bits`，且 BAC 是 2.0 bit 预算内的结果，IC3Net/TarMAC 是 B=∞ 上界，
  报告里必须写明“不同预算下的上界参照”，不能直接当同预算对手比。

- BAC 三个 seed 的 ferr 极稳（0.318 / 0.346 / 0.361），且 `bits=1.81 < 2.0` 预算内；
- MAPPO 的 `speak_rate=1.0` 是**显示口径 bug**（Actor 无门控头时 `gate=ones`，旧逻辑照 `gate` 统计），
  已修（`evaluate.py` 改为看 `gate_logits`、`train.py` 按 `algo.comm=none` 归零），数据本身有效，表里按 0 填；
- MAPPO 的 s2 是离群 seed（ferr 0.537、collisions 27、return −81），报告里要写方差而不是只取均值；
- BAC 的 `robust.csv` 在 drop_p 0→0.5 上基本平的（ferr 0.42–0.45），说明丢包没打崩策略，
  但**不能据此宣称"鲁棒"**——训练时本来就带丢包，报告里按"无显著退化"写。

QMIX 已按新配置重跑并入库（E 的 Release `results-2026-11-22-e-ippo-qmix`，SHA256 `be845c13…` 校验一致），
训练不再发散，但末次 ferr 1.276±0.625、collisions 20.38，仍是六方法中最弱的一个；写报告时按实测值写，
不要把"修复发散"等同于"性能恢复"。

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
| qmix × 3 seed 重跑 | E | ✅ 已交并入库 |
| bac × 3、mappo × 3 | C | ✅ 已交并入库 |
| ic3net × 3、tarmac × 3 | D | ✅ 已交并入库 |
| E 的 ippo/qmix 包落进 `results/main` | A/E | ✅ 完成（18/18 通过） |
| 合并 18 run + 出图 + 汇总表 | A/E | ✅ 完成（18/18，报告与 PPT 已按此更新） |
| 结题报告 + 答辩 PPT | 全员 | ✅ 已出 PDF（20 页）与 9 页 PPT，待填组号/姓名 |

`scripts/make_figures.py` 已按 comm 类型重算 `speak_rate`（不再直接采信历史 `robust.csv` 那一列，
见 `GATE_COMMS`），18 run 的图在 `results/figures/` 与 `results/_logs/deck_figs/`（报告/PPT 用的是后者）。

## 5. 已知待决（需要拍板，不是 bug）

1. 是否加跑 `collision_coef ∈ {0.2, 0.5}` 对照（B 标定层）；
2. 是否把 `eval_gate_mode` 默认从 `threshold` 翻成 `sample`（影响 bits 读法，报告要注明）；
3. encoder 是否加 `LayerNorm`（会改 ckpt 形状，需单独 A/B）；
4. 是否加跑 500k–1M 冲 `success_rate`（当前最好 ferr 0.686，阈值 0.09，差约 7 倍）。
