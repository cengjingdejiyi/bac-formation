"""批量实验驱动的排班测试（负责人 E）。

只测「计划」这一层：展开、撞名、评测信道对齐、与 train.run_name 口径一致。
不涉及训练本身，缺 torch 也能跑（口径对照那一条会在 CI 里真正执行）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for extra in ("scripts", "src"):
    p = str(ROOT / extra)
    if p not in sys.path:
        sys.path.insert(0, p)

import run_experiments as rx  # noqa: E402


def test_ablation_and_main_never_share_a_directory():
    jobs = rx.plan(["main", "ablation"], seeds=[0, 1], total_env_steps=2000)
    keys = [(j["exp_name"], j["run_name"]) for j in jobs]
    assert len(keys) == len(set(keys)), "消融与主实验不能落在同一个 results 目录"
    assert not rx.find_collisions(jobs)


def test_ablation_variants_get_the_suffix_from_contract_section_6():
    jobs = rx.plan(["ablation"], seeds=[0], total_env_steps=1000)
    assert {j["exp_name"] for j in jobs} == {"ablation"}
    suffixes = {j["variant"]: j["run_name"].split("_s0", 1)[1] for j in jobs}
    assert suffixes == {"no_gate": "_nogate", "mean_agg": "_aggmean", "no_noise": "_nonoise"}


def test_no_gate_pins_dual_step():
    job = next(
        j for j in rx.plan(["ablation"], seeds=[0], total_env_steps=1000) if j["variant"] == "no_gate"
    )
    assert job["cfg"]["algo"]["use_gate"] is False
    assert job["cfg"]["algo"]["beta_lr"] == 0.0


def test_mean_agg_switches_aggregate_only():
    job = next(
        j for j in rx.plan(["ablation"], seeds=[0], total_env_steps=1000) if j["variant"] == "mean_agg"
    )
    assert job["cfg"]["algo"]["aggregate"] == "mean"
    assert job["cfg"]["algo"]["use_gate"] is True


def test_no_noise_keeps_eval_channel_at_training_midpoint():
    job = next(
        j for j in rx.plan(["ablation"], seeds=[0], total_env_steps=1000) if j["variant"] == "no_noise"
    )
    env_channel = job["cfg"]["env"]["channel"]
    assert env_channel["drop_p"] == 0.0 and env_channel["flip_p"] == 0.0
    # 训练区间 [0, 0.3] / [0, 0.1] 的中点，否则评测也在零噪声下跑
    assert job["cfg"]["train"]["eval_drop_p"] == pytest.approx(0.15)
    assert job["cfg"]["train"]["eval_flip_p"] == pytest.approx(0.05)


def test_sweep_is_single_factor_and_skips_baseline():
    jobs = rx.plan(["sweep"], seeds=[0], total_env_steps=1000)
    # 每轴只扫非基线的点：msg_bits 2/8/16 + recv_k 1/2/4（默认 K=3 不在列表）
    # + budget_bits 1.0/4.0 = 8，不是 4×3×3=36 的笛卡尔积
    assert len(jobs) == 8
    assert {j["exp_name"] for j in jobs} == {"sweep"}
    assert not rx.find_collisions(jobs)


def test_include_baseline_merges_identical_configs():
    jobs = rx.plan(["sweep"], seeds=[0], total_env_steps=1000, include_baseline=True)
    # 10 个候选里 msg_bits=4 与 budget_bits=2.0 是同一份配置，合并成 1 条
    assert len(jobs) == 9
    assert not rx.find_collisions(jobs)


def test_seeds_expand_every_variant():
    jobs = rx.plan(["ablation"], seeds=[0, 1, 2], total_env_steps=1000)
    assert len(jobs) == 9
    assert sorted({j["seed"] for j in jobs}) == [0, 1, 2]


def test_total_env_steps_override_reaches_config():
    job = rx.plan(["main"], seeds=[0], total_env_steps=2000)[0]
    assert job["cfg"]["train"]["total_env_steps"] == 2000
    assert "train.total_env_steps=2000" in job["tokens"]


def test_unknown_group_is_rejected():
    with pytest.raises(ValueError):
        rx.plan(["nope"], seeds=[0])


def test_collision_guard_flags_same_experiment_and_run_name():
    jobs = [
        {"exp_name": "main", "run_name": "formation_6n_bac_L4_K3_B2.0_s0", "variant": "main"},
        {"exp_name": "main", "run_name": "formation_6n_bac_L4_K3_B2.0_s0", "variant": "mean_agg"},
    ]
    problems = rx.find_collisions(jobs)
    assert problems and "撞名" in problems[0]


@pytest.mark.parametrize(
    "overrides",
    [
        [],
        ["algo.aggregate=mean"],
        ["algo.use_gate=false"],
        ["env.channel.drop_p=0.0", "env.channel.flip_p=0.0"],
    ],
)
def test_run_name_matches_train_module(overrides):
    """口径必须与 train.run_name 完全一致，含 contracts 第 6 节的变体后缀。"""
    tr = pytest.importorskip("bacmarl.runners.train")
    from bacmarl.utils.config import load_config

    tokens = ["env=formation", "algo=bac", "seed=7", *overrides]
    cfg = load_config(tokens)
    assert rx.run_name(cfg) == tr.run_name(cfg)


@pytest.mark.parametrize("algo", ["bac", "mappo", "ippo", "ic3net", "tarmac", "qmix"])
def test_run_name_matches_training_entry_for_every_algo(algo):
    """#11 之后每个算法都必须对得上；连续消息的 L/B 段是这次修的重点。"""
    pytest.importorskip("torch")
    from bacmarl.utils.config import load_config

    from bacmarl.runners import qmix as qmix_runner
    from bacmarl.runners import train as train_runner

    cfg = load_config([f"algo={algo}", "seed=3"])
    expected = qmix_runner.run_name(cfg) if algo == "qmix" else train_runner.run_name(cfg)
    assert rx.run_name(cfg) == expected
    # 名字里的 L 必须是通信模块口径：连续消息 1024，无通信 0
    if algo in ("ic3net", "tarmac"):
        assert "_L1024_" in expected and "L0_" not in expected
    elif algo in ("mappo", "ippo", "qmix"):
        assert "_L0_" in expected


def test_plan_directories_match_training_run_name_for_continuous_baselines():
    """排班算出的 run_dir 必须与训练实际写盘的名字一致（否则结果会落进两处）。"""
    jobs = rx.plan(["main"], seeds=[0])
    assert jobs
    from bacmarl.utils.config import load_config

    for algo in ("ic3net", "tarmac"):
        cfg = load_config(["algo=" + algo, "seed=0"])
        assert rx.run_name(cfg).endswith("_L1024_K3_Binf_s0")
        assert "B1000000000" not in rx.run_name(cfg)


def test_algos_axis_expands_main_and_routes_qmix():
    """主实验的算法轴：算法 × seed，QMIX 必须派发到独立入口。"""
    jobs = rx.plan(["main"], seeds=[0], total_env_steps=1000, algos=["bac", "ic3net", "qmix"])
    names = {j["algo"]: j["run_name"] for j in jobs}
    assert set(names) == {"bac", "ic3net", "qmix"}
    assert names["bac"].endswith("L4_K3_B2.0_s0")
    assert names["ic3net"].endswith("L1024_K3_Binf_s0")
    assert names["qmix"].endswith("L0_K3_B0.0_s0")

    runners = {j["algo"]: j["runner"] for j in jobs}
    assert runners["qmix"] == "bacmarl.runners.qmix"
    assert runners["bac"] == "bacmarl.runners.train"
    assert "bacmarl.runners.qmix" in next(j["command"] for j in jobs if j["algo"] == "qmix")
    assert not rx.find_collisions(jobs)


def test_algos_axis_multiplies_seeds():
    jobs = rx.plan(["main"], seeds=[0, 1], total_env_steps=1000, algos=["bac", "qmix"])
    assert len(jobs) == 4
    assert {(j["algo"], j["seed"]) for j in jobs} == {
        ("bac", 0),
        ("bac", 1),
        ("qmix", 0),
        ("qmix", 1),
    }


def test_algos_only_affects_main_group(capsys):
    """消融/扫描是 BAC 自己的轴，--algos 不该把它们展开成别的算法。"""
    jobs = rx.plan(["ablation"], seeds=[0], total_env_steps=1000, algos=["ippo"])
    assert {j["algo"] for j in jobs} == {"bac"}
    assert "只作用于 main 组" in capsys.readouterr().out


def test_main_group_without_algos_still_plans_bac():
    """不传 --algos 时保持旧行为，避免已有脚本的命令行语义变化。"""
    jobs = rx.plan(["main"], seeds=[0], total_env_steps=1000)
    assert [j["variant"] for j in jobs] == ["main"]
    assert [j["algo"] for j in jobs] == ["bac"]

