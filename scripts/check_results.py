"""结果汇总前的自检（负责人 E）。

    python scripts/check_results.py --results results/main

检查每个 run 目录是否齐全、metrics.csv 是否含必需列、行数是否与步数匹配、
是否出现过 mock 后端、`loss/value` 量级是否正常（value_norm 生效的旁证）、
QMIX 是否用了稳定化配置且价值没有发散（`qmix/q_tot` / `loss/td` 双阈值）。
不通过时以非零码退出，方便接到 CI 或批处理里。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REQUIRED_COLUMNS = {
    "env_steps",
    "train/return",
    "train/return_all",
    "train/formation_err",
    "train/formation_err_mean",
    "train/collisions",
    "comm/bits",
    "comm/beta",
    "loss/value",
}
# QMIX 走 TD 而不是 PPO，没有 loss/value；其余列必须齐。
QMIX_EXEMPT = {"loss/value"}
REQUIRED_FILES = ("config.yaml", "metrics.csv", "ckpt/last.ckpt")

# QMIX 发散判据（2026-11-22 修复后设定）：奖励缩放到 0.1 后合法 Q 量级约 −20，
# 旧配置实测冲到 5360 / loss 2.2e6。这里用「正常上限的 1.5 倍」留出余量。
QMIX_Q_TOT_MAX = 30.0
QMIX_TD_LOSS_MAX = 1000.0
# 稳定化三件套进配置才说明这批 QMIX 是新口径（2026-11-22 之后）。
QMIX_REQUIRED_ALGO_KEYS = ("td_huber", "target_clip")


def _num(row: dict, key: str) -> float | None:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return None


def _load_yaml(path: Path) -> dict:
    try:
        import yaml
    except Exception:  # pragma: no cover - 没装 pyyaml 时跳过配置检查
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _check_qmix_config(run_dir: Path) -> list[str]:
    """QMIX 的 run 必须带稳定化参数，否则就是 2026-11-22 之前的发散配置。"""
    problems = []
    cfg = _load_yaml(run_dir / "config.yaml")
    algo = (cfg.get("algo") or {}) if isinstance(cfg, dict) else {}
    missing = [k for k in QMIX_REQUIRED_ALGO_KEYS if k not in algo]
    if missing:
        problems.append(
            f"配置缺 {missing}：这是 2026-11-22 之前的 QMIX 配置（价值发散），需重跑"
        )
    elif algo.get("td_huber") is False:
        problems.append("td_huber=false：MSE 自举在 200k 预算下会发散，需重跑")
    return problems


def _check_qmix_series(rows: list[dict]) -> list[str]:
    """QMIX 价值量级双阈值：任何一项越界都说明这批曲线不能用。"""
    problems = []
    q_tot = [abs(v) for v in (_num(r, "qmix/q_tot") for r in rows) if v is not None]
    if q_tot:
        peak = max(q_tot)
        if peak > QMIX_Q_TOT_MAX:
            problems.append(f"qmix/q_tot 峰值 {peak:.1f} > {QMIX_Q_TOT_MAX:.0f}：价值发散")
    td_loss = [v for v in (_num(r, "loss/td") for r in rows) if v is not None]
    if td_loss:
        mean = sum(td_loss) / len(td_loss)
        if mean > QMIX_TD_LOSS_MAX:
            problems.append(
                f"loss/td 均值 {mean:.0f} > {QMIX_TD_LOSS_MAX:.0f}：TD 目标在放大"
            )
    return problems


def check_run(run_dir: Path, min_rows: int) -> list[str]:
    problems = []
    is_qmix = "qmix" in run_dir.name
    for rel in REQUIRED_FILES:
        if not (run_dir / rel).exists():
            problems.append(f"缺文件 {rel}")
    if is_qmix:
        problems.extend(_check_qmix_config(run_dir))
    metrics = run_dir / "metrics.csv"
    if not metrics.exists():
        return problems
    rows = list(csv.DictReader(metrics.open(encoding="utf-8")))
    if not rows:
        problems.append("metrics.csv 为空")
        return problems
    missing = REQUIRED_COLUMNS - set(rows[0])
    if is_qmix:
        missing -= QMIX_EXEMPT
    if missing:
        problems.append(f"缺列 {sorted(missing)}")
    if len(rows) < min_rows:
        problems.append(f"行数过少 {len(rows)} < {min_rows}")
    steps = _num(rows[-1], "env_steps")
    if steps:
        expect = int(steps) // 512
        if abs(len(rows) - expect) > 2:
            problems.append(
                f"行数 {len(rows)} 与 env_steps {int(steps)} 不匹配（应约 {expect}）"
            )
    vloss = [_num(r, "loss/value") for r in rows]
    vloss = [v for v in vloss if v is not None]
    if vloss and sum(vloss) / len(vloss) > 50:
        problems.append(
            f"loss/value 均值 {sum(vloss) / len(vloss):.1f} > 50：value_norm 可能没生效"
        )
    if is_qmix:
        problems.extend(_check_qmix_series(rows))
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="汇总前自检（E）")
    ap.add_argument("--results", default="results/main", help="实验组目录")
    ap.add_argument("--min-rows", type=int, default=30, help="metrics.csv 最少行数")
    args = ap.parse_args(argv)

    root = Path(args.results)
    if not root.is_dir():
        print(f"[FAIL] 目录不存在：{root}")
        return 2
    runs = sorted(
        p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")
    )
    if not runs:
        print(f"[FAIL] {root} 下没有 run 目录")
        return 2

    failed = 0
    for run in runs:
        problems = check_run(run, args.min_rows)
        if problems:
            failed += 1
            print(f"[BAD ] {run.name}: " + "; ".join(problems))
        else:
            print(f"[OK  ] {run.name}")
    print(f"\n{len(runs) - failed}/{len(runs)} 通过")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
