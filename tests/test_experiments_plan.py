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
    "algo,overrides",
    [
        ("bac", []),
        ("bac", ["algo.aggregate=mean"]),
        ("bac", ["algo.use_gate=false"]),
        ("bac", ["env.channel.drop_p=0.0", "env.channel.flip_p=0.0"]),
        # 连续消息基线的 L/B 段（Issue #11）：驱动必须与训练算出同一个名字，
        # 否则排班写 `L0_B1e9`、训练写 `L1024_Binf`，同名判断和撞名检查全失效。
        ("ic3net", []),
        ("tarmac", []),
        # 值分解基线（L0/B0.0）。
        ("qmix", []),
    ],
)
def test_run_name_matches_train_module(algo, overrides):
    """口径必须与 train.run_name 完全一致，含 contracts 第 6 节的变体后缀。"""
    tr = pytest.importorskip("bacmarl.runners.train")
    from bacmarl.utils.config import load_config

    tokens = ["env=formation", f"algo={algo}", "seed=7", *overrides]
    cfg = load_config(tokens)
    assert rx.run_name(cfg) == tr.run_name(cfg)
