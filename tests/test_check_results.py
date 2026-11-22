"""汇总前自检与结果合并的回归测试（2026-11-22 补）。

`check_results.py` 之前只检文件和列；QMIX 发散那批曲线照样能"通过"，所以补两条：
旧配置（缺 td_huber/target_clip）判 BAD、价值量级越界判 BAD。
`merge_results.py` 用来合并多人交付，必须能识别 run 名、拒绝非 run 目录、不覆盖。
"""

from __future__ import annotations

import csv
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for extra in ("scripts", "src"):
    if str(ROOT / extra) not in sys.path:
        sys.path.insert(0, str(ROOT / extra))

import check_results  # noqa: E402
import merge_results  # noqa: E402

COLS_PPO = [
    "env_steps", "train/return", "train/return_all", "train/formation_err",
    "train/formation_err_mean", "train/collisions", "comm/bits", "comm/beta",
    "loss/value", "eval/formation_err",
]
COLS_QMIX = [c for c in COLS_PPO if c != "loss/value"] + [
    "loss/td", "qmix/q_tot", "qmix/target", "qmix/td_abs", "qmix/target_clip",
    "qmix/grad_norm",
]


def _write_run(
    root: Path,
    name: str,
    cols: list[str],
    steps: int = 5120,
    q_tot: float = -3.0,
    td: float = 0.2,
    ret: str = "-40",
    yaml_text: str = "algo:\n  name: ippo\n",
    with_ckpt: bool = True,
) -> Path:
    run = root / name
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.yaml").write_text(yaml_text, encoding="utf-8")
    if with_ckpt:
        (run / "ckpt").mkdir(exist_ok=True)
        (run / "ckpt" / "last.ckpt").write_bytes(b"x")
    rows = []
    n = max(1, steps // 512)
    for i in range(n):
        row = {c: "0" for c in cols}
        row["env_steps"] = str((i + 1) * 512)
        row["train/return"] = ret
        row["train/return_all"] = "-45"
        row["train/formation_err"] = "1.0"
        row["train/formation_err_mean"] = "1.1"
        row["train/collisions"] = "2"
        if "loss/value" in cols:
            row["loss/value"] = "0.5"
        if "loss/td" in cols:
            row["loss/td"] = str(td)
            row["qmix/q_tot"] = str(q_tot)
            row["qmix/target"] = str(q_tot)
            row["qmix/td_abs"] = "0.3"
            row["qmix/target_clip"] = "0"
            row["qmix/grad_norm"] = "3"
        rows.append(row)
    with (run / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return run


QMIX_YAML_OK = "algo:\n  name: qmix\n  td_huber: true\n  target_clip: 50.0\n  lr: 0.00025\n"
QMIX_YAML_OLD = "algo:\n  name: qmix\n  lr: 0.0005\n  double_q: true\n"


def test_ppo_run_passes(tmp_path):
    _write_run(tmp_path, "formation_6n_ippo_L0_K3_B0.0_s0", COLS_PPO)
    assert check_results.check_run(tmp_path / "formation_6n_ippo_L0_K3_B0.0_s0", 5) == []


def test_qmix_new_config_passes(tmp_path):
    _write_run(tmp_path, "formation_6n_qmix_L0_K3_B0.0_s0", COLS_QMIX, yaml_text=QMIX_YAML_OK)
    assert check_results.check_run(tmp_path / "formation_6n_qmix_L0_K3_B0.0_s0", 5) == []


def test_qmix_old_config_is_rejected(tmp_path):
    """缺 td_huber/target_clip 的 run 就是发散那批，必须在汇总前拦下来。"""
    _write_run(tmp_path, "formation_6n_qmix_L0_K3_B0.0_s1", COLS_QMIX, yaml_text=QMIX_YAML_OLD)
    problems = check_results.check_run(tmp_path / "formation_6n_qmix_L0_K3_B0.0_s1", 5)
    assert any("td_huber" in p for p in problems)


def test_qmix_diverged_values_are_rejected(tmp_path):
    """配置对但价值量级越界（Q 到 5000 / loss 到 2e6）同样判 BAD。"""
    _write_run(
        tmp_path,
        "formation_6n_qmix_L0_K3_B0.0_s2",
        COLS_QMIX,
        q_tot=5000.0,
        td=2.2e6,
        yaml_text=QMIX_YAML_OK,
    )
    problems = check_results.check_run(tmp_path / "formation_6n_qmix_L0_K3_B0.0_s2", 5)
    assert any("qmix/q_tot" in p for p in problems)
    assert any("loss/td" in p for p in problems)


def test_merge_copies_runs_and_skips_non_runs(tmp_path):
    dest = tmp_path / "main"
    src = tmp_path / "交接_1"
    _write_run(src, "formation_6n_bac_L4_K3_B2.0_s0", COLS_PPO)
    (src / "_logs").mkdir()
    (src / "_logs" / "run.log").write_text("x", encoding="utf-8")

    rc = merge_results.main(["--dest", str(dest), "--src", str(src)])
    assert rc == 0
    assert (dest / "formation_6n_bac_L4_K3_B2.0_s0" / "metrics.csv").exists()
    assert not (dest / "_logs").exists()
    assert (dest / merge_results.MANIFEST).exists()


def test_merge_refuses_collision_without_overwrite(tmp_path):
    dest = tmp_path / "main"
    src1, src2 = tmp_path / "a", tmp_path / "b"
    _write_run(src1, "formation_6n_bac_L4_K3_B2.0_s0", COLS_PPO, ret="-40")
    _write_run(src2, "formation_6n_bac_L4_K3_B2.0_s0", COLS_PPO, ret="-99")
    assert merge_results.main(["--dest", str(dest), "--src", str(src1)]) == 0
    assert merge_results.main(["--dest", str(dest), "--src", str(src2)]) == 0
    # 第二次被跳过 → 目标仍是第一份
    rows = list(csv.DictReader((dest / "formation_6n_bac_L4_K3_B2.0_s0" / "metrics.csv").open()))
    assert rows[0]["train/return"] == "-40"


def test_merge_handles_zip_source(tmp_path):
    dest = tmp_path / "main"
    src = tmp_path / "src"
    run = _write_run(src, "formation_6n_tarmac_L1024_K3_Binf_s0", COLS_PPO)
    zip_path = tmp_path / "pack.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for f in run.rglob("*"):
            if f.is_file():
                zf.write(f, Path(run.name) / f.relative_to(run))
    rc = merge_results.main(["--dest", str(dest), "--src", str(zip_path)])
    assert rc == 0
    assert (dest / "formation_6n_tarmac_L1024_K3_Binf_s0" / "metrics.csv").exists()


def test_merge_dry_run_writes_nothing(tmp_path):
    dest = tmp_path / "main"
    src = tmp_path / "src"
    _write_run(src, "formation_6n_mappo_L0_K3_B0.0_s1", COLS_PPO)
    rc = merge_results.main(["--dest", str(dest), "--src", str(src), "--dry-run"])
    assert rc == 0
    assert not (dest / "formation_6n_mappo_L0_K3_B0.0_s1").exists()


def test_merge_bad_path_returns_2(tmp_path):
    assert merge_results.main(["--dest", str(tmp_path / "m"), "--src", str(tmp_path / "nope")]) == 2
