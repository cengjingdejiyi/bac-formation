"""评测网格解析与 robust.csv 列测试（E，contracts 第 6 节）。

不需要 torch：不调用真实的 `evaluate_policy`，只验证网格来源优先级与产物列。
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bacmarl.runners import evaluate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _write_yaml(path: Path, grid) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "experiment:\n  robustness:\n    drop_p: [" + ", ".join(str(g) for g in grid) + "]\n"
    path.write_text(text, encoding="utf-8")
    return path


def test_columns_default_matches_contract():
    assert evaluate.columns() == [
        "drop_p",
        "success_rate",
        "formation_err",
        "speak_rate",
        "bits",
    ]


def test_columns_with_extra_appends_tail():
    cols = evaluate.columns(with_extra=True)
    assert cols[:5] == evaluate.columns()
    assert cols[5:] == ["return", "collisions", "length"]


def test_grid_from_config_accepts_scalar_and_list():
    assert evaluate.grid_from_config({"experiment": {"robustness": {"drop_p": 0.0}}}) == [0.0]
    assert evaluate.grid_from_config(
        {"experiment": {"robustness": {"drop_p": [0.0, 0.1, 0.3]}}}
    ) == [0.0, 0.1, 0.3]


def test_grid_from_config_missing_returns_none():
    assert evaluate.grid_from_config({}) is None
    assert evaluate.grid_from_config(None) is None
    assert evaluate.grid_from_config({"experiment": {"robustness": {}}}) is None


def test_parse_grid_supports_comma_and_bracket():
    assert evaluate.parse_grid("0.0,0.15,0.3") == [0.0, 0.15, 0.3]
    assert evaluate.parse_grid("[0.0,0.3]") == [0.0, 0.3]
    assert evaluate.parse_grid("0.2") == [0.2]
    assert evaluate.parse_grid(None) is None


def test_resolve_grid_cli_wins(tmp_path):
    run_cfg = {"experiment": {"robustness": {"drop_p": [0.0, 0.15]}}}
    ext = _write_yaml(tmp_path / "ext.yaml", [0.0, 0.1])
    grid, src = evaluate.resolve_grid(
        run_cfg, cli_grid=[0.4], config_path=ext, repo_grid=tmp_path / "none.yaml"
    )
    assert grid == [0.4] and src == "--drop-p"


def test_resolve_grid_config_beats_run_config(tmp_path):
    run_cfg = {"experiment": {"robustness": {"drop_p": [0.0, 0.15]}}}
    ext = _write_yaml(tmp_path / "ext.yaml", [0.0, 0.1, 0.2, 0.3])
    grid, src = evaluate.resolve_grid(run_cfg, config_path=ext, repo_grid=tmp_path / "none.yaml")
    assert grid == [0.0, 0.1, 0.2, 0.3]
    assert "ext.yaml" in src


def test_resolve_grid_run_config_beats_repo_default(tmp_path):
    run_cfg = {"experiment": {"robustness": {"drop_p": [0.0, 0.15]}}}
    grid, src = evaluate.resolve_grid(run_cfg, repo_grid=tmp_path / "none.yaml")
    assert grid == [0.0, 0.15]
    assert "config.yaml" in src


def test_resolve_grid_falls_back_to_repo_robust_yaml(tmp_path):
    # experiment=main 训练出来的 run 里没有 experiment.robustness
    grid, src = evaluate.resolve_grid({"experiment": {"name": "main"}})
    assert grid == [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    assert "robust.yaml" in src


def test_resolve_grid_last_resort_is_zero_and_says_so(tmp_path, capsys):
    grid, src = evaluate.resolve_grid({}, repo_grid=tmp_path / "none.yaml")
    assert grid == [0.0] and src == "fallback"
    assert "找不到丢包网格" in capsys.readouterr().out


def test_list_grid_runs_without_torch(tmp_path, capsys):
    run = tmp_path / "formation_6n_bac_L4_K3_B2.0_s0"
    run.mkdir()
    _write_yaml(run / "config.yaml", [0.0, 0.2])
    assert evaluate.main(["--run", str(run), "--list-grid"]) == 0
    assert "丢包网格" in capsys.readouterr().out


def test_robust_csv_columns_follow_contract(tmp_path, monkeypatch):
    run = tmp_path / "formation_6n_bac_L4_K3_B2.0_s0"
    run.mkdir()
    _write_yaml(run / "config.yaml", [0.0, 0.3])
    metrics = {k: 0.5 for k in evaluate._METRIC_KEYS}
    monkeypatch.setattr(evaluate, "evaluate_policy", lambda *a, **k: dict(metrics))
    monkeypatch.setattr(evaluate, "_load_actor", lambda *a, **k: None)

    assert evaluate.main(["--run", str(run), "--episodes", "1", "--allow-random"]) == 0
    with open(run / "robust.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0].keys()) == evaluate.columns()
    assert [float(r["drop_p"]) for r in rows] == [0.0, 0.3]


def test_robust_csv_extra_columns(tmp_path, monkeypatch):
    run = tmp_path / "formation_6n_bac_L4_K3_B2.0_s0"
    run.mkdir()
    _write_yaml(run / "config.yaml", [0.0])
    metrics = {k: 0.25 for k in evaluate._METRIC_KEYS}
    monkeypatch.setattr(evaluate, "evaluate_policy", lambda *a, **k: dict(metrics))
    monkeypatch.setattr(evaluate, "_load_actor", lambda *a, **k: None)

    assert evaluate.main(
        ["--run", str(run), "--episodes", "1", "--with-extra", "--allow-random"]
    ) == 0
    with open(run / "robust.csv", encoding="utf-8") as f:
        header = next(csv.reader(f))
    assert header == evaluate.columns(with_extra=True)


def test_multiple_runs_write_own_csv(tmp_path, monkeypatch):
    runs = []
    for seed in (0, 1):
        run = tmp_path / f"formation_6n_bac_L4_K3_B2.0_s{seed}"
        run.mkdir()
        _write_yaml(run / "config.yaml", [0.0])
        runs.append(str(run))
    metrics = {k: 0.1 for k in evaluate._METRIC_KEYS}
    monkeypatch.setattr(evaluate, "evaluate_policy", lambda *a, **k: dict(metrics))
    monkeypatch.setattr(evaluate, "_load_actor", lambda *a, **k: None)

    assert evaluate.main(["--run", *runs, "--episodes", "1", "--allow-random"]) == 0
    for run in runs:
        assert (Path(run) / "robust.csv").exists()


def test_missing_run_dir_is_an_error_and_writes_nothing(tmp_path, capsys):
    """路径打错不能"成功"：以前会 mkdir 并写一份随机策略的 robust.csv。"""
    ghost = tmp_path / "typo_run"
    assert evaluate.main(["--run", str(ghost), "--drop-p", "0.0", "--episodes", "1"]) == 2
    assert "不存在" in capsys.readouterr().out
    assert not ghost.exists()


def _write_run_cfg(path: Path, grid, flip=None) -> Path:
    """带 env.channel 的 run 配置，用来验证评测信道口径。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["experiment:", "  robustness:", "    drop_p: [" + ", ".join(str(g) for g in grid) + "]"]
    if flip is not None:
        lines += ["env:", "  channel:", f"    flip_p: [{flip[0]}, {flip[1]}]"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_flip_p_defaults_to_training_eval_channel(tmp_path, monkeypatch, capsys):
    """省略 --flip-p 时取训练内评测口径（区间中点 0.05），不写死 0。"""
    run = tmp_path / "formation_6n_bac_L4_K3_B2.0_s0"
    run.mkdir()
    _write_run_cfg(run / "config.yaml", [0.0], flip=(0.0, 0.1))
    seen = {}

    def fake_eval(actor, env_cfg, episodes=8, seed=0):
        seen["flip_p"] = float(env_cfg.channel["flip_p"])
        return {k: 0.5 for k in evaluate._METRIC_KEYS}

    monkeypatch.setattr(evaluate, "evaluate_policy", fake_eval)
    monkeypatch.setattr(evaluate, "_load_actor", lambda *a, **k: None)
    assert evaluate.main(["--run", str(run), "--episodes", "1", "--allow-random"]) == 0
    assert seen["flip_p"] == 0.05
    assert "flip_p=0.05" in capsys.readouterr().out


def test_flip_p_cli_overrides_training_channel(tmp_path, monkeypatch):
    run = tmp_path / "formation_6n_bac_L4_K3_B2.0_s0"
    run.mkdir()
    _write_run_cfg(run / "config.yaml", [0.0], flip=(0.0, 0.1))
    seen = {}

    def fake_eval(actor, env_cfg, episodes=8, seed=0):
        seen["flip_p"] = float(env_cfg.channel["flip_p"])
        return {k: 0.5 for k in evaluate._METRIC_KEYS}

    monkeypatch.setattr(evaluate, "evaluate_policy", fake_eval)
    monkeypatch.setattr(evaluate, "_load_actor", lambda *a, **k: None)
    assert evaluate.main(
        ["--run", str(run), "--episodes", "1", "--allow-random", "--flip-p", "0.2"]
    ) == 0
    assert seen["flip_p"] == 0.2


def test_invalid_drop_p_is_an_error_not_a_fallback(tmp_path, capsys):
    """显式给了 --drop-p 却解析不出来，不能悄悄退回仓库默认网格。"""
    run = tmp_path / "formation_6n_bac_L4_K3_B2.0_s0"
    run.mkdir()
    _write_yaml(run / "config.yaml", [0.0, 0.3])
    assert evaluate.main(
        ["--run", str(run), "--drop-p", "0.0,abc", "--episodes", "1", "--allow-random"]
    ) == 2
    assert "非法" in capsys.readouterr().out
    assert not (run / "robust.csv").exists()


def test_out_with_multiple_runs_is_an_error(tmp_path, capsys):
    runs = []
    for seed in (0, 1):
        run = tmp_path / f"formation_6n_bac_L4_K3_B2.0_s{seed}"
        run.mkdir()
        _write_yaml(run / "config.yaml", [0.0])
        runs.append(str(run))
    out = tmp_path / "one.csv"
    assert evaluate.main(["--run", *runs, "--out", str(out), "--episodes", "1"]) == 2
    assert "--out" in capsys.readouterr().out
