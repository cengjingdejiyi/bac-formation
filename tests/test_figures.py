"""出图脚本测试（E）：多种子聚合、nan 过滤、缺数据降级。

不需要 torch：只造 metrics.csv / robust.csv，验证聚合与产物。
"""

import csv
import sys
from pathlib import Path

import numpy as np  # noqa: F401  (aggregate 返回 ndarray，测试里用它做比较)
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import make_figures as mf  # noqa: E402

METRIC_HEADER = [
    "env_steps",
    "train/return",
    "train/success_rate",
    "train/formation_err",
    "train/collisions",
    "eval/return",
    "eval/success_rate",
    "eval/formation_err",
    "eval/collisions",
    "eval/bits",
    "eval/speak_rate",
]


def _write_csv(path: Path, header: list[str], rows: list[list]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return path


def _metrics_rows(train_return: float, eval_return: float) -> list[list]:
    return [
        [1000, train_return, 0.1, 0.9, 2, "nan", "", "", "", "", ""],
        [2000, train_return + 1, 0.2, 0.8, 1, eval_return, 0.3, 0.7, 1, 3.2, 0.4],
    ]


def _build_tree(root: Path) -> Path:
    for seed, bump in ((0, 0.0), (1, 1.0)):
        run = root / "main" / f"formation_6n_bac_L4_K3_B2.0_s{seed}"
        _write_csv(run / "metrics.csv", METRIC_HEADER, _metrics_rows(1.0 + bump, 2.0 + bump))
        _write_csv(
            run / "robust.csv",
            ["drop_p", "success_rate", "formation_err", "speak_rate", "bits"],
            [[0.0, 0.9 + 0.1 * bump, 0.5, 0.2, 3.0], [0.3, 0.6 + 0.1 * bump, 0.8, 0.3, 2.5]],
        )
    run = root / "ablation" / "formation_6n_bac_L4_K3_B2.0_s0_nogate"
    _write_csv(run / "metrics.csv", METRIC_HEADER, _metrics_rows(0.5, 1.0))
    return root


def test_parse_run_name_seed_and_variant():
    info = mf.parse_run_name("formation_6n_bac_L4_K3_B2.0_s0")
    assert info == {"label": "formation_6n_bac_L4_K3_B2.0", "seed": 0}

    info = mf.parse_run_name("formation_6n_bac_L4_K3_B2.0_s3_nogate-nonoise")
    assert info["label"] == "formation_6n_bac_L4_K3_B2.0_nogate-nonoise"
    assert info["seed"] == 3


def test_parse_run_name_without_seed():
    assert mf.parse_run_name("whatever") == {"label": "whatever", "seed": None}


def test_load_metrics_groups_seeds_by_config(tmp_path):
    _build_tree(tmp_path)
    data = mf.load_metrics(tmp_path)
    label = "main/formation_6n_bac_L4_K3_B2.0"
    assert label in data["series"]
    assert data["seeds"][label] == {0, 1}
    pts = data["series"][label]["train/return"][1000.0]
    assert sorted(pts) == [1.0, 2.0]
    assert "ablation/formation_6n_bac_L4_K3_B2.0_nogate" in data["series"]


def test_labels_carry_the_experiment_group(tmp_path):
    """同一 run 名出现在不同实验组时必须分开，否则会被当成多 seed 合并。"""
    _build_tree(tmp_path)
    # 与 main 里同名、同 seed、且没有变体后缀的消融跑
    run = tmp_path / "ablation" / "formation_6n_bac_L4_K3_B2.0_s0"
    _write_csv(run / "metrics.csv", METRIC_HEADER, _metrics_rows(9.0, 9.0))
    data = mf.load_metrics(tmp_path)
    main_lb = "main/formation_6n_bac_L4_K3_B2.0"
    abl_lb = "ablation/formation_6n_bac_L4_K3_B2.0"
    assert data["seeds"][main_lb] == {0, 1}
    assert data["seeds"][abl_lb] == {0}
    assert max(data["series"][abl_lb]["train/return"][1000.0]) == 9.0
    assert 9.0 not in data["series"][main_lb]["train/return"][1000.0]


def test_run_outside_group_dirs_keeps_plain_label(tmp_path):
    """`--results` 下一层直接放 run（没有实验组目录）时不加前缀。"""
    run = tmp_path / "formation_6n_bac_L4_K3_B2.0_s0"
    _write_csv(run / "metrics.csv", METRIC_HEADER, _metrics_rows(1.0, 2.0))
    data = mf.load_metrics(tmp_path)
    assert list(data["series"]) == ["formation_6n_bac_L4_K3_B2.0"]


def test_load_metrics_skips_nan_and_empty(tmp_path):
    _build_tree(tmp_path)
    data = mf.load_metrics(tmp_path)
    col = data["series"]["main/formation_6n_bac_L4_K3_B2.0"]["eval/return"]
    assert sorted(col) == [2000.0]  # 1000 步那行是 nan / 空
    assert col[2000.0] == [2.0, 3.0]


def test_aggregate_mean_and_std():
    xs, mean, std, n = mf.aggregate({0.0: [1.0, 3.0], 0.3: [2.0]})
    assert list(xs) == [0.0, 0.3]
    assert list(mean) == [2.0, 2.0]
    assert std[0] == pytest.approx(1.4142135, rel=1e-6)
    assert std[1] == 0.0
    assert list(n) == [2, 1]


def test_load_robust_groups_seeds(tmp_path):
    _build_tree(tmp_path)
    data = mf.load_robust(tmp_path)
    label = "main/formation_6n_bac_L4_K3_B2.0"
    pts = data["series"][label]["success_rate"][0.3]
    assert sorted(pts) == [0.6, 0.7]


def test_main_returns_one_without_data(tmp_path):
    assert mf.main(["--results", str(tmp_path)]) == 1


def test_main_list_only_does_not_write(tmp_path, capsys):
    _build_tree(tmp_path)
    out = tmp_path / "figs"
    assert mf.main(["--results", str(tmp_path), "--out", str(out), "--list"]) == 0
    assert "main/formation_6n_bac_L4_K3_B2.0" in capsys.readouterr().out
    assert not out.exists()


def test_main_writes_figures(tmp_path, capsys):
    pytest.importorskip("matplotlib")
    _build_tree(tmp_path)
    out = tmp_path / "figs"
    assert mf.main(["--results", str(tmp_path), "--out", str(out)]) == 0
    names = sorted(p.name for p in out.glob("*.png"))
    assert names == [
        "comm_metrics.png",
        "curves_return.png",
        "robust_drop.png",
        "train_eval_metrics.png",
    ]
    assert "共 4 张图" in capsys.readouterr().out


def test_main_skip_robust(tmp_path):
    pytest.importorskip("matplotlib")
    _build_tree(tmp_path)
    out = tmp_path / "figs"
    assert mf.main(["--results", str(tmp_path), "--out", str(out), "--skip-robust"]) == 0
    assert not (out / "robust_drop.png").exists()
