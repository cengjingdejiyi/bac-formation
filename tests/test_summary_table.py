"""`scripts/make_summary_table.py` 的聚合口径测试（不依赖 torch）。"""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_summary_table.py"


def _load():
    spec = importlib.util.spec_from_file_location("make_summary_table", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_run(root: Path, group: str, name: str, rows: list[dict]) -> Path:
    d = root / group / name
    d.mkdir(parents=True, exist_ok=True)
    cols = sorted({k for r in rows for k in r})
    with open(d / "metrics.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["env_steps", *cols])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return d


def test_aggregates_two_seeds_and_takes_last_eval(tmp_path):
    mod = _load()
    root = tmp_path / "results"
    for seed, ret in ((0, -10.0), (1, -20.0)):
        _write_run(root, "main", f"formation_6n_bac_L4_K3_B2.0_s{seed}", [
            {"env_steps": 10000, "eval/return": -99.0, "eval/formation_err": 9.0,
             "eval/collisions": 9.0, "comm/bits": 9.0, "comm/speak_rate": 9.0},
            {"env_steps": 20000, "eval/return": ret, "eval/formation_err": 0.3 + seed,
             "eval/collisions": 2.0, "comm/bits": 1.8, "comm/speak_rate": 0.45},
        ])
    md, recs = mod.render(root)
    assert len(recs) == 1
    r = recs[0]
    assert r["run"] == "formation_6n_bac_L4_K3_B2.0"
    assert r["seeds"] == 2
    # 取最后一次评测（-10 / -20），不是第一行 -99
    assert r["eval/return"] == pytest.approx(-15.0)
    assert r["eval/return_std"] == pytest.approx(7.0710678, rel=1e-4)
    assert "B2.0" in md and "L4" in md and "BAC" in md


def test_no_comm_speak_rate_zero_and_name_parsing(tmp_path):
    mod = _load()
    root = tmp_path / "results"
    for seed in (0, 1, 2):
        _write_run(root, "main", f"formation_6n_ippo_L0_K3_B0.0_s{seed}", [
            {"env_steps": 20000, "eval/return": -40.0, "eval/formation_err": 0.4,
             "eval/collisions": 3.0, "comm/bits": 0.0, "comm/speak_rate": 0.0},
        ])
    md, recs = mod.render(root)
    assert recs[0]["L"] == "0" and recs[0]["B"] == "0.0"
    assert recs[0]["seeds"] == 3
    assert "IPPO" in md
    assert "!旧占位口径" not in md
    assert "⚠n=" not in md


def test_no_comm_placeholder_normalized_to_zero(tmp_path):
    """无通信 run 的 metrics.csv 里 speak_rate 若为历史占位 1.0，汇总表按 0 计。"""
    mod = _load()
    root = tmp_path / "results"
    _write_run(root, "main", "formation_6n_mappo_L0_K3_B0.0_s0", [
        {"env_steps": 20000, "eval/return": -50.0, "eval/formation_err": 0.36,
         "eval/collisions": 4.0, "comm/bits": 0.0, "comm/speak_rate": 1.0},
    ])
    md, recs = mod.render(root)
    assert recs[0]["comm/speak_rate"] == pytest.approx(0.0)
    assert "!旧占位口径" not in md   # 不再输出误导性的 1.00
    assert "| 0.00 |" in md          # speak_rate 列已归零
    assert "⚠n=1" in md              # 单 seed 不能当结论


def test_ignores_dirs_without_metrics_or_bad_names(tmp_path):
    mod = _load()
    root = tmp_path / "results"
    (root / "main" / "_legacy_qmix" / "formation_6n_qmix_L0_K3_B0.0_s0").mkdir(parents=True)
    (root / "main" / "not_a_run").mkdir(parents=True)
    _write_run(root, "main", "formation_6n_tarmac_L1024_K3_Binf_s0", [
        {"env_steps": 20000, "eval/return": -38.0, "eval/formation_err": 0.32,
         "eval/collisions": 3.0, "comm/bits": 1024.0, "comm/speak_rate": 1.0},
    ])
    md, recs = mod.render(root)
    assert len(recs) == 1
    assert "B∞" in md
    assert "TarMAC" in md
