"""开跑前自检脚本的回归测试。

重点不是"当前仓库全绿"，而是**旧 clone 必须被拦下**：少 `td_huber` 的 QMIX 配置、
少 `value_target_scale` 的训练入口、少 QMIX 判据的 check_results，都要判 FAIL。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import preflight  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_fails(monkeypatch):
    monkeypatch.setattr(preflight, "FAILS", [])
    yield


def test_current_repo_passes_without_env(tmp_path, monkeypatch):
    """当前仓库的配置层必须全绿（跳真实环境，CI 里也能跑）。"""
    assert preflight.main(["--skip-env"]) == 0
    assert not preflight.FAILS


def test_old_clone_qmix_config_is_caught(tmp_path, monkeypatch):
    """旧 clone：qmix.yaml 没有 td_huber/target_clip、train.py 没有 value_target_scale。"""
    fake_root = tmp_path
    (fake_root / "configs" / "algo").mkdir(parents=True)
    (fake_root / "configs" / "algo" / "qmix.yaml").write_text(
        "algo:\n  name: qmix\n  lr: 0.0005\n", encoding="utf-8"
    )
    (fake_root / "scripts").mkdir()
    (fake_root / "scripts" / "check_results.py").write_text("# 旧版\n", encoding="utf-8")
    (fake_root / "src" / "bacmarl" / "runners").mkdir(parents=True)
    (fake_root / "src" / "bacmarl" / "runners" / "train.py").write_text(
        "def _restore_run(): pass\n", encoding="utf-8"
    )
    monkeypatch.setattr(preflight, "ROOT", fake_root)
    preflight.FAILS = []
    preflight.check_code_version()
    assert len(preflight.FAILS) == 3
    assert "代码版本：QMIX 稳定化参数已入库" in preflight.FAILS


def test_qmix_old_defaults_are_caught(monkeypatch):
    """即使代码是新的，配置被改回旧口径（lr 5e-4 / updates 4 / 不夹逼）也要 FAIL。"""
    old = {
        "algo": {
            "name": "qmix",
            "td_huber": False,
            "target_clip": None,
            "lr": 5e-4,
            "updates_per_train": 4,
        }
    }
    monkeypatch.setattr(preflight, "_load_config", lambda tokens: old)
    preflight.FAILS = []
    preflight.check_qmix_config()
    assert len(preflight.FAILS) == 4


def test_value_norm_off_is_caught(monkeypatch):
    bad = {
        "train": {
            "value_norm": False,
            "reward_norm": False,
            "eval_gate_mode": "threshold",
            "total_env_steps": 200000,
        },
        "env": {
            "heading_mode": "dynamic",
            "success_dist": 0.09,
            "formation_radius": 0.5,
            "collision_coef": 1.0,
        },
    }
    monkeypatch.setattr(preflight, "_load_config", lambda tokens: bad)
    preflight.FAILS = []
    preflight.check_main_config()
    assert preflight.FAILS == ["train.value_norm = true"]


def test_backend_mock_is_caught():
    """mock 后端必须记 FAIL（这是"整批数据作废"的最常见原因）。"""
    preflight.FAILS = []
    preflight.check("真实后端 backend=mpe2", False, "实际 mock：这一批数据作废")
    assert preflight.FAILS == ["真实后端 backend=mpe2"]
