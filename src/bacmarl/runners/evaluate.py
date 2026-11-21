"""评测：确定性回合评测（供 train 定期调用）与按丢包率网格复测。

评测入口（负责人 E）与训练内评测（负责人 C）共用 `evaluate_policy`，避免两套策略语义。

用法：
    # 单个 run：网格取该 run 的 config.yaml，缺失时回落到 configs/experiment/robust.yaml
    python -m bacmarl.runners.evaluate --run results/main/formation_6n_bac_L4_K3_B2.0_s0
    # 多个 seed 一次跑完（各自写自己的 robust.csv）
    python -m bacmarl.runners.evaluate --run results/main/xxx_s0 results/main/xxx_s1
    # 用 experiment=main 训练的 run 里没有 experiment.robustness，显式给网格来源
    python -m bacmarl.runners.evaluate --run results/main/xxx --config configs/experiment/robust.yaml
    # 直接给网格（也支持 "[0.0,0.15]" 写法）
    python -m bacmarl.runners.evaluate --run results/main/xxx --drop-p "0.0,0.15,0.3"
    # 只看网格、不加载模型（不需要 torch）
    python -m bacmarl.runners.evaluate --run results/main/xxx --list-grid
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from bacmarl.envs.formation_comm import EnvConfig, FormationCommEnv  # noqa: E402
from bacmarl.utils.runtime import eval_channel  # noqa: E402

_METRIC_KEYS = (
    "return",
    "success_rate",
    "formation_err",
    "collisions",
    "length",
    "bits",
    "speak_rate",
)

# contracts 第 6 节：robust.csv 列 = drop_p, success_rate, formation_err, speak_rate, bits
DEFAULT_COLUMNS = ("drop_p", "success_rate", "formation_err", "speak_rate", "bits")
EXTRA_COLUMNS = ("return", "collisions", "length")
REPO_GRID = ROOT / "configs" / "experiment" / "robust.yaml"


def columns(with_extra: bool = False) -> list[str]:
    """契约列在前；`--with-extra` 时把 return/collisions/length 追加在后。"""
    cols = list(DEFAULT_COLUMNS)
    if with_extra:
        cols.extend(EXTRA_COLUMNS)
    return cols


def grid_from_config(cfg: dict | None) -> list[float] | None:
    """从 `experiment.robustness.drop_p` 取网格；没有该段返回 None（不返回空列表）。"""
    rob = ((cfg or {}).get("experiment") or {}).get("robustness") or {}
    grid = rob.get("drop_p")
    if grid is None:
        return None
    if isinstance(grid, (int, float)):
        grid = [grid]
    vals: list[float] = []
    for g in grid:
        try:
            vals.append(float(g))
        except (TypeError, ValueError):
            continue
    return vals or None


def _load_config_file(path: Path) -> dict:
    """读外部实验配置（支持 `extends: main`，与 configs/experiment/*.yaml 同语义）。"""
    from bacmarl.utils.config import _read_group

    return _read_group(Path(path))


def resolve_grid(
    run_cfg: dict | None,
    cli_grid: list[float] | None = None,
    config_path: str | Path | None = None,
    repo_grid: str | Path | None = None,
) -> tuple[list[float], str]:
    """决定丢包网格，优先级：--drop-p > --config > run 的 config.yaml > 仓库默认 > [0.0]。

    返回 (网格, 来源说明)，来源会打到日志里，避免「网格悄悄退化成 [0.0]」。
    """
    if cli_grid:
        return list(cli_grid), "--drop-p"
    if config_path is not None:
        try:
            g = grid_from_config(_load_config_file(Path(config_path)))
        except Exception as exc:
            print(f"[WARN] 读 {config_path} 失败（{exc}）")
            g = None
        if g:
            return g, f"--config {config_path}"
        print(f"[WARN] {config_path} 里没有 experiment.robustness.drop_p，继续回退")
    g = grid_from_config(run_cfg)
    if g:
        return g, "run 的 config.yaml"
    fallback = Path(repo_grid) if repo_grid is not None else REPO_GRID
    if fallback.exists():
        try:
            g = grid_from_config(_load_config_file(fallback))
        except Exception as exc:
            print(f"[WARN] 读默认鲁棒网格失败（{exc}）")
            g = None
        if g:
            return g, f"默认 {fallback}"
    print("[WARN] 找不到丢包网格，只用 drop_p=0.0 复测")
    return [0.0], "fallback"


def parse_grid(text: str | None) -> list[float] | None:
    """把 `--drop-p` 文本解析成浮点列表，复用 config 的 `[a,b]` / 逗号分隔语义。"""
    if not text:
        return None
    from bacmarl.utils.config import _parse_scalar

    val = _parse_scalar(text)
    if isinstance(val, (int, float)):
        val = [val]
    if not isinstance(val, list):
        return None
    try:
        return [float(v) for v in val]
    except (TypeError, ValueError):
        # 混进 `abc` 这类脏值时整条网格视为非法，由调用方报错，不静默丢掉那一项
        return None


def evaluate_policy(
    actor, env_cfg: EnvConfig, episodes: int = 8, seed: int = 0
) -> dict:
    """用确定性策略跑固定回合，返回各指标均值。

    动作取 argmax；门控取 p>=0.5（`Actor.forward(deterministic=True)`）；
    每回合隐状态从零开始，回合内不重置。actor 为 None 时退化为均匀随机动作。
    """
    import torch

    n = env_cfg.n_agents
    acc = {k: [] for k in _METRIC_KEYS}
    for ep in range(episodes):
        env = FormationCommEnv(env_cfg)
        try:
            step = env.reset(seed=seed + ep)
            torch.manual_seed(seed + ep)
            hidden = (
                actor.init_hidden(1, n) if actor is not None else torch.zeros(1, n, 1)
            )
            rng = np.random.default_rng(seed + ep)
            total, length, bits, gates = 0.0, 0, [], []
            while True:
                if actor is not None:
                    with torch.no_grad():
                        out = actor(
                            torch.as_tensor(step.obs)[None],
                            torch.as_tensor(step.obs_mask)[None],
                            torch.as_tensor(step.comm_mask)[None],
                            hidden,
                            dict(step.info["channel"]),
                            deterministic=True,
                        )
                    hidden = out.h
                    action = out.action_logits.argmax(-1)[0].numpy()
                    bits.append(float(out.bits.mean()))
                    if out.gate is not None:
                        gates.append(float(out.gate.mean()))
                else:
                    action = rng.integers(0, 5, size=n)
                nxt = env.step(action)
                total += float(np.mean(nxt.reward))
                length += 1
                done, step = nxt.done, nxt
                if done:
                    break
            acc["return"].append(total)
            acc["length"].append(length)
            acc["success_rate"].append(float(step.info["success"]))
            acc["formation_err"].append(float(step.info["formation_err"]))
            acc["collisions"].append(float(step.info["collisions_ep"]))
            acc["bits"].append(float(np.mean(bits)) if bits else 0.0)
            acc["speak_rate"].append(float(np.mean(gates)) if gates else 0.0)
        finally:
            env.close()
    return {k: float(np.mean(v)) if v else 0.0 for k, v in acc.items()}


def _read_cfg(run_dir: Path) -> dict:
    """读 run 目录里的 config.yaml；缺失时回退默认配置但**必须显式告警**。

    run 目录没有 config.yaml 说明它不是本仓库训练产出的，回退默认配置会让
    hidden/n_agents/通信模块与真实 ckpt 不符，于是加载必然失败——这条路径
    以前被 `except Exception` 吞成了"随机策略"，现在至少先把原因打出来。
    """
    p = run_dir / "config.yaml"
    if not p.exists():
        from bacmarl.utils.config import load_config

        print(f"[WARN] {p} 不存在：回退默认配置，模型超参可能与 ckpt 不符")
        return load_config([])
    try:
        import yaml

        return yaml.safe_load(p.read_text(encoding="utf-8"))
    except Exception:
        import json

        return json.loads(p.read_text(encoding="utf-8"))


def _load_actor(run_dir: Path, cfg: dict, ckpt: str):
    """构造 actor 并载入权重；**只有 ckpt 文件不存在**时返回 None。

    ckpt 存在但加载失败（维度不符 / hidden 改变 / 通信模块换过 / 文件损坏）
    必须向上抛错：静默退化成随机策略会写出一行看起来完全正常的 robust.csv，
    比直接报错危险得多。是否允许随机策略由调用方的 `--allow-random` 决定。
    """
    from bacmarl.comm.base import build_comm
    from bacmarl.models.actor import Actor
    from bacmarl.utils.checkpoint import load_checkpoint

    path = run_dir / ckpt
    if not path.exists():
        print(f"[WARN] 未找到 {path}，退化为随机策略")
        return None

    hidden = int((cfg.get("train") or {}).get("hidden", 64))
    algo = cfg.get("algo") or {}
    if "comm" not in algo:
        raise ValueError(f"{run_dir / 'config.yaml'} 缺少 algo.comm，无法判定通信模块")
    env_cfg = EnvConfig.from_dict(cfg.get("env", {}))
    comm = build_comm(algo["comm"], hidden, algo)
    prot = FormationCommEnv(env_cfg)
    try:
        actor = Actor(prot.obs_dim, env_cfg.n_obs, 5, hidden, comm)
    finally:
        prot.close()
    load_checkpoint(path, actor=actor)
    actor.eval()
    return actor


def _resolve_run(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="+", required=True, help="一个或多个 run 目录")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--ckpt", default="ckpt/last.ckpt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--config", default=None, help="外部实验配置，读 experiment.robustness.drop_p"
    )
    ap.add_argument("--drop-p", default=None, help='显式丢包网格，如 "0.0,0.15,0.3"')
    ap.add_argument(
        "--flip-p",
        type=float,
        default=None,
        help="复测时固定翻转率；省略则与训练内评测同口径（train.eval_flip_p 或训练区间中点）",
    )
    ap.add_argument(
        "--with-extra",
        action="store_true",
        help="robust.csv 追加 return/collisions/length",
    )
    ap.add_argument("--out", default=None, help="输出 csv 路径（仅单 run 时有效）")
    ap.add_argument("--list-grid", action="store_true", help="只打印网格并退出")
    ap.add_argument(
        "--allow-random",
        action="store_true",
        help="显式允许缺 checkpoint 时用随机策略出对照行（默认缺 ckpt 直接失败）",
    )
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    cli_grid = parse_grid(args.drop_p)
    if args.drop_p is not None and cli_grid is None:
        # 静默回退到仓库默认会让"我指定了网格"这件事变成假话，网格来源在日志里也不对。
        print(
            f"[FAIL] --drop-p 取值非法：{args.drop_p!r}"
            '（示例："0.0,0.15,0.3" 或 "[0.0,0.3]"）'
        )
        return 2
    cols = columns(args.with_extra)

    # 路径打错时绝不能"成功"：mkdir 会把目录建出来、缺 ckpt 会静默退化成随机策略，
    # 最后在正式结果的目录里留一份随机策略的 robust.csv，事后很难辨认。
    runs = [_resolve_run(p) for p in args.run]
    missing = [p for p in runs if not p.is_dir()]
    if missing:
        for p in missing:
            print(f"[FAIL] run 目录不存在：{p}")
        return 2
    if args.out and len(runs) > 1:
        print(
            "[FAIL] --out 只能配单个 --run；多 run 时请让每个 run 写自己的 robust.csv"
        )
        return 2

    for run_dir in runs:
        cfg = _read_cfg(run_dir)
        grid, source = resolve_grid(cfg, cli_grid=cli_grid, config_path=args.config)
        print(f"[INFO] {run_dir.name or run_dir} 丢包网格 {grid}（来源：{source}）")
        if args.list_grid:
            continue

        # 翻转率不能写死 0：训练内评测用的是 train.eval_flip_p 或训练区间中点，
        # 复测若固定 0 就变成"训练时见过噪声、评测时没有"，两条曲线不可比。
        if args.flip_p is None:
            ec_train = cfg.get("train") or {}
            flip_p = float(eval_channel(cfg.get("env", {}), ec_train)["flip_p"])
            flip_src = "训练内评测口径（train.eval_flip_p 或训练区间中点）"
        else:
            flip_p = float(args.flip_p)
            flip_src = "--flip-p"
        if not 0.0 <= flip_p <= 1.0:
            print(f"[FAIL] flip_p={flip_p:g} 越界：概率必须在 [0,1]")
            return 2
        bad = [d for d in grid if not 0.0 <= float(d) <= 1.0]
        if bad:
            print(f"[FAIL] 丢包概率越界：{bad}（必须在 [0,1]）")
            return 2
        print(
            f"[INFO] {run_dir.name or run_dir} 固定 flip_p={flip_p:g}（来源：{flip_src}）"
        )

        ckpt_path = run_dir / args.ckpt
        if not ckpt_path.exists() and not args.allow_random:
            print(
                f"[FAIL] 缺少 {ckpt_path}：正式评测不能拿随机策略冒充策略结果；"
                "确实要做随机对照请显式加 --allow-random"
            )
            return 3

        actor = None
        if ckpt_path.exists():
            try:
                actor = _load_actor(run_dir, cfg, args.ckpt)
            except Exception as exc:
                if not args.allow_random:
                    print(
                        f"[FAIL] {ckpt_path} 加载失败：{exc}\n"
                        "       ckpt 存在却加载不了（维度/hidden/通信模块/文件损坏），"
                        "不能静默退化成随机策略；确认要做随机对照请显式加 --allow-random"
                    )
                    return 4
                print(
                    f"[WARN] {ckpt_path} 加载失败（{exc}），按 --allow-random 用随机策略"
                )
        else:
            actor = _load_actor(run_dir, cfg, args.ckpt)
        rows = []
        for drop_p in grid:
            ec = EnvConfig.from_dict(cfg.get("env", {}))
            ec.channel = {"drop_p": float(drop_p), "flip_p": flip_p}
            m = evaluate_policy(actor, ec, episodes=args.episodes, seed=args.seed)
            rows.append(
                {"drop_p": float(drop_p), **{k: m[k] for k in cols if k != "drop_p"}}
            )

        out_csv = Path(args.out) if args.out else run_dir / "robust.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"[DONE] {out_csv}")
        for r in rows:
            print("  " + ", ".join(f"{c}={r[c]:.4f}" for c in cols))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
