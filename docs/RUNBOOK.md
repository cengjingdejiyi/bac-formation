# 运行手册（多 seed 正式实验）

本文件是"另一个人在自己的机器上跑多 seed"的唯一操作说明。接口与指标口径见
`docs/contracts.md`；本文只讲**装环境、跑数、交结果、验收**。

## 0. 环境口径（必须对齐）

| 项目 | 取值 | 说明 |
|---|---|---|
| Python | **3.11**（CI 基线）或 3.13（本机已验证） | 两者都能跑；同一批 seed 必须用同一个版本，不要把 3.11 和 3.13 的结果混在一张表里 |
| torch | `2.15.0+cpu`（或任意 `>=2.2` 的 CPU 版） | 无 GPU 时不要装 CUDA 版，装 CPU 版省 2 GB |
| mpe2 | `1.1.1` | 真实物理后端；缺了会自动退回 mock，**正式数据必须是 mpe2** |
| numpy / scipy / matplotlib | 见 `requirements-lock.txt` | 出图用同一份 matplotlib，避免字体/线宽不一致 |

环境清单：

- `requirements.txt`：**约束**（版本区间），日常安装用这个；
- `requirements-lock.txt`：**冻结**（本机实测可用的精确版本，50 个包），复现用这个。

### 一键安装

Windows（PowerShell）：

```powershell
pwsh -File scripts/setup_env.ps1            # 默认用清华镜像 + torch CPU 源
pwsh -File scripts/setup_env.ps1 -Lock      # 严格按 requirements-lock.txt 复现
```

macOS / Linux：

```bash
bash scripts/setup_env.sh            # 约束安装
bash scripts/setup_env.sh --lock     # 冻结复现
```

装完必须自检（三条都要过）：

```bash
python scripts/smoke.py                      # 契约冒烟，不依赖 torch
python -m pytest -q                          # 期望 265+ passed
python scripts/verify_transitions.py         # 缓存奖励/终止与真实 step 对齐
python -c "import torch, mpe2; print(torch.__version__, mpe2.__version__)"
```

> 输出里 `backend=mpe2` 才算数。看到 `backend=mock` 说明 mpe2 没装上，**这一批数据作废**。

> Windows 上 `setup_env.ps1` 已存为 **UTF-8 with BOM**，Windows PowerShell 5.1 和 `pwsh` 7 都能直接跑。
> 若你把它复制/另存成无 BOM 的 UTF-8，PS 5.1 会按 ANSI 解码中文注释并报 `Unexpected token`，此时改用 `pwsh -File ...` 即可。
> 仓库已加 `.gitattributes`：`*.sh` 强制 LF、`*.ps1` 强制 CRLF，避免 Git Bash 下出现 `\r` 解析错误。

## 1. 正式实验：6 方法 × 3 seed

先排班、核对运行名，再执行：

```bash
# 1) 只排班（不训练），确认名字带 _s0/_s1/_s2 且 L/B 段正确
python scripts/run_experiments.py --groups main \
  --algos bac ippo mappo ic3net tarmac qmix --seeds 0 1 2 --dry-run

# 2) 真跑（单进程串行；失败继续）
python scripts/run_experiments.py --groups main \
  --algos bac ippo mappo ic3net tarmac qmix --seeds 0 1 2 \
  --execute --keep-going
```

- 单 run 200k 步实测 **约 15–20 分钟**（4 分钟/5 万步，8 并行环境，CPU）；18 个 run 串行约 **5–6 小时**。
- 想跑更长（500k–1M，冲成功率）加 `--total-env-steps 500000`。
- 想分机器跑，用 `--only` 或 `--algos` 拆开，但**不要改运行名**，否则汇总时对不上。
- **QMIX 必须用 2026-11-22 之后的配置**（`td_huber: true`、`target_clip: 50.0`、`lr 2.5e-4`、
  `updates_per_train: 2`、`target_update: 200`）。旧的 3 个 seed（`results/main/...qmix...s0/s1/s2`）
  在 43k 步起价值发散（`loss/td` 到 1e6 级），**曲线作废**，需按新配置重跑后再汇总。
  自检只需看 `qmix/q_tot` 是否保持在 ±30 内、`loss/td` 是否稳定在个位数。

### 分工建议（3 台机器，每人 2 个方法 × 3 seed）

| 人 | 命令片段 | 预计 |
|---|---|---|
| C | `--algos bac mappo --seeds 0 1 2` | ~2 h |
| D | `--algos ic3net tarmac --seeds 0 1 2` | ~1.5 h |
| E | `--algos ippo qmix --seeds 0 1 2` | ~1.5 h |

每台机器跑完，把整个 `results/main/` 目录打包给对方（`results/` 不入库，见 `.gitignore`）。

## 2. 结果交付与汇总

目录结构固定：`results/<实验组>/<env>_<N>n_<algo>_L<bits>_K<k>_B<budget>_s<seed>/`

必须包含：`config.yaml`、`metrics.csv`、`ckpt/last.ckpt`、`robust.csv`；
`tb/` 可选。缺 `metrics.csv` 的目录直接退回。

### 2.1 合并多人的包（汇总机执行）

```bash
python scripts/merge_results.py --dest results/main \
  --src D:/交接/E_ippo_qmix.zip --src D:/交接/C_bac_mappo
```

- 源可以是 zip 或目录，会递归找**含 `metrics.csv`** 的目录当作 run；run 名必须带 `_s<seed>`，否则拒收（防止把 `_logs`、`figures` 搬进来）；
- 同名 run 默认**跳过并报告**，`--overwrite` 才覆盖；合并清单写到 `results/main/_merge_manifest.tsv`（谁交的、多大、什么时候并的）；
- `--dry-run` 先看清单不动文件；
- 合并完必须接着跑第 2.2 节自检。

### 2.2 汇总前自检（E 负责，Windows/macOS/Linux 通用）

```bash
python scripts/check_results.py --results results/main
```

它逐个 run 检查：`config.yaml`/`metrics.csv`/`ckpt/last.ckpt` 是否齐全、必需列是否缺、
行数与 `env_steps` 是否匹配、`loss/value` 是否仍在 O(1) 量级（`value_norm` 没生效会直接报出来）；
QMIX 还会多查两条：配置里有没有 `td_huber`/`target_clip`（缺=2026-11-22 之前的发散配置），
以及 `qmix/q_tot` 是否 >30、`loss/td` 是否 >1000（价值发散）。
有问题退出码非 0，可以直接串进批处理。

出图（**务必指定正式目录**，默认 `results/` 会把历史验证目录混进图例）：

```bash
python scripts/make_figures.py --results results/main --out results/figures
```

## 3. 指标怎么读（别读错）

| 指标 | 口径 | 用途 |
|---|---|---|
| `train/return` | 最近 100 回合滑动均值 | 看趋势 |
| `train/return_all` | 全程累计均值（续训继承） | 报告里写"平均回合回报"用它 |
| `train/formation_err` | **回合末**编队误差 | 终局精度 |
| `train/formation_err_mean` | **回合内每步**均值 | 与 `train/step_reward` 同口径 |
| `eval/formation_err` | 确定性策略 + 固定丢包（训练区间中点） | 正式对比列 |
| `eval/bits` | 受 `eval_gate_mode` 影响：`threshold` 会高估发言率 | 预算合规请用 `evaluate.py --gate-mode sample` 复测 |
| `comm/beta` | 对偶变量；`0.000` 占多数说明预算约束松弛 | 判"带宽受限是否真的起作用" |

## 4. 复测（鲁棒性）

```bash
# 训练内评测口径（threshold，历史可比）
python -m bacmarl.runners.evaluate --run results/main/<run_name> --episodes 8

# 与训练分布同口径（预算比较用这个）
python -m bacmarl.runners.evaluate --run results/main/<run_name> --episodes 8 --gate-mode sample

# 丢包网格
python -m bacmarl.runners.evaluate --run results/main/<run_name> --drop-p 0.0,0.15,0.3 --gate-mode sample
```

失败码：`2` 参数/路径错、`3` 缺 ckpt、**`4` ckpt 存在但加载失败**（维度/hidden/通信模块不符）。
只有显式 `--allow-random` 才会用随机策略出表，那种 `robust.csv` **不得进汇总**。

## 5. 提交前检查清单

- [ ] 每台机器都打印过 `backend=mpe2`
- [ ] `metrics.csv` 行数 ≈ `total_env_steps / 512`（200k ≈ 391 行）
- [ ] 运行名里的 `_s<seed>` 与 `config.yaml` 的 `seed` 一致
- [ ] `loss/value` 在 O(1) 量级（`value_norm=true` 生效；若均值 > 50 说明配置没带上）
- [ ] `python -m pytest -q` 全绿；`python scripts/smoke.py` 通过
- [ ] 出图只用 `--results results/main`
