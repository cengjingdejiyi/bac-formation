"""批量实验驱动：把 experiment 配置展开成 train 命令，查撞名，按序执行（负责人 E）。

契约依据 `docs/contracts.md` 第 6 节（运行名与结果目录规范）。本脚本只做「排班」：
展开 seeds × 配置 → 生成命令 → 检查撞名 → 写清单 → 可选执行；
不改 train / evaluate 的任何语义，也不复制它们内部的逻辑。
运行名直接取 `bacmarl.utils.naming.run_name`（Issue #11 之后命名只有这一份实现），
排班算出的目录名与训练实际写盘的名字不可能再分叉。

四个必须挡住的坑（都在脚本里处理）：
1. `run_name` 不含 aggregate/use_gate/信道，消融变体彼此同名 → 依赖 #12 的变体后缀
   （`_aggmean` / `_nogate` / `_nonoise`）区分，本脚本复用同一份命名实现并再查一遍撞名。
2. `no_noise` 把训练信道归零后，评测信道会跟着变 0，鲁棒性失去对照
   → 自动补 `train.eval_drop_p/eval_flip_p` = 训练区间中点。
3. 单因素扫描的基线值与其它轴算出的名字相同 → 默认跳过基线点（由 main 组提供）。
4. 同一 (实验组, run_name) 出现两次 → 直接报错，不落任何目录。

用法：
    python scripts/run_experiments.py --list
    python scripts/run_experiments.py --groups main ablation --seeds 0 1 2 --dry-run
    python scripts/run_experiments.py --groups main --algos bac ippo ic3net tarmac qmix --seeds 0 --dry-run
    python scripts/run_experiments.py --groups main --seeds 0 --total-env-steps 2000 --execute

`--algos` 只作用于 `main` 组：主实验是「六种方法 × 多 seed」，算法名进 `run_name`，
QMIX 走独立入口 `bacmarl.runners.qmix`，其余走 `bacmarl.runners.train`。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from bacmarl.utils.config import load_config  # noqa: E402
from bacmarl.utils.naming import run_name  # noqa: E402  (Issue #11：命名只留一份实现)

PLAN_COLUMNS = (
    "group",
    "variant",
    "seed",
    "total_env_steps",
    "exp_name",
    "run_name",
    "run_dir",
    "command",
)

# 字符串形式的旧版 ablation 配置（`ablation: [no_gate, ...]`）也要能跑。
_LEGACY_OVERRIDES = {
    "no_gate": {"algo.use_gate": False, "algo.beta_lr": 0.0},
    "mean_agg": {"algo.aggregate": "mean"},
    "no_noise": {"env.channel.drop_p": 0.0, "env.channel.flip_p": 0.0},
}


def _fmt(value) -> str:
    """把配置值转成 CLI token 形式（bool 必须是 true/false，load_config 才认）。"""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _dig(cfg: dict, dotted: str):
    """按 `algo.msg_bits` 这种点路径取配置值，取不到返回 None。"""
    cur = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _channel_midpoint(base: dict) -> dict:
    """训练信道区间中点，与 `train._eval_channel` 的退化规则一致。"""
    channel = (base.get("env") or {}).get("channel") or {}
    out = {}
    for key in ("drop_p", "flip_p"):
        value = channel.get(key, 0.0)
        if isinstance(value, (list, tuple)):
            out[f"eval_{key}"] = 0.5 * (float(value[0]) + float(value[-1]))
        else:
            out[f"eval_{key}"] = float(value)
    return out


def _variant(entry) -> tuple[str, dict]:
    """消融条目 → (变体名, 覆盖项)。兼容字符串与 {name, overrides} 两种写法。"""
    if isinstance(entry, str):
        if entry not in _LEGACY_OVERRIDES:
            raise ValueError(f"未知消融变体 {entry!r}，且没有自带 overrides")
        return entry, dict(_LEGACY_OVERRIDES[entry])
    name = entry["name"]
    overrides = entry.get("overrides") or _LEGACY_OVERRIDES.get(name, {})
    return name, dict(overrides)


# QMIX 是 off-policy 值分解，有独立入口；其余都挂在统一 PPO 入口上。
_QMIX_RUNNER = "bacmarl.runners.qmix"
_TRAIN_RUNNER = "bacmarl.runners.train"


def _runner_module(algo_name: str) -> str:
    """按算法选入口：只有 qmix 走独立 runner，其余（含 BAC/IPPO/MAPPO/IC3Net/TarMAC）走 train。"""
    return _QMIX_RUNNER if str(algo_name) == "qmix" else _TRAIN_RUNNER


def _job(
    group: str,
    variant: str,
    seed: int,
    extra_tokens: list[str],
    total_env_steps: int | None,
    exp_name: str,
    algo: str | None = None,
) -> dict:
    tokens = [
        f"experiment={group}",
        f"experiment.name={exp_name}",
        f"seed={int(seed)}",
        *([f"algo={algo}"] if algo else []),
        *extra_tokens,
    ]
    if total_env_steps:
        tokens.append(f"train.total_env_steps={int(total_env_steps)}")
    cfg = load_config(tokens)
    name = run_name(cfg)
    algo_name = str(cfg["algo"].get("name") or cfg["algo"].get("comm"))
    module = _runner_module(algo_name)
    run_dir = ROOT / "results" / str(cfg["experiment"]["name"]) / name
    cmd = [sys.executable, "-m", module, *tokens]
    return {
        "group": group,
        "variant": variant,
        "seed": int(seed),
        "total_env_steps": int((cfg.get("train") or {}).get("total_env_steps", 0)),
        "exp_name": str(cfg["experiment"]["name"]),
        "run_name": name,
        "run_dir": str(run_dir),
        "command": " ".join(cmd),
        "runner": module,
        "algo": algo_name,
        "cfg": cfg,
        "tokens": tokens,
    }


def plan(
    groups: list[str],
    seeds: list[int],
    total_env_steps: int | None = None,
    only: list[str] | None = None,
    include_baseline: bool = False,
    algos: list[str] | None = None,
) -> list[dict]:
    """展开实验计划。groups 只接受会训练的分组；robust 是评测网格，不在此列。

    `algos` 只展开 `main` 组（主实验 = 六种方法 × 多 seed）；消融与扫描本来就是
    BAC 自己的轴，传进来会被忽略并提示。
    """
    jobs: list[dict] = []
    for group in groups:
        if group == "robust":
            print("[SKIP] robust 只是评测网格（evaluate.py 用），没有训练侧任务")
            continue
        base = load_config([f"experiment={group}"])

        if group == "main":
            for algo in algos or [None]:
                for seed in seeds:
                    jobs.append(
                        _job(group, algo or "main", seed, [], total_env_steps, "main", algo)
                    )

        elif group == "ablation":
            if algos:
                print("[SKIP] --algos 只作用于 main 组；消融是 BAC 自己的变体轴")
            variants = (base.get("experiment") or {}).get("ablation") or []
            for entry in variants:
                name, overrides = _variant(entry)
                if only and name not in only:
                    continue
                tokens = [f"{k}={_fmt(v)}" for k, v in overrides.items()]
                touches_channel = any(k.startswith("env.channel") for k in overrides)
                if name == "no_noise" or touches_channel:
                    # 训练区间被改掉时，评测信道必须显式钉在原来的中点
                    for key, value in _channel_midpoint(base).items():
                        tokens.append(f"train.{key}={_fmt(value)}")
                for seed in seeds:
                    jobs.append(
                        _job(group, name, seed, tokens, total_env_steps, "ablation")
                    )

        elif group == "sweep":
            if algos:
                print("[SKIP] --algos 只作用于 main 组；扫描是 BAC 自己的 L/K/B 轴")
            entries = (base.get("experiment") or {}).get("sweep") or []
            for entry in entries:
                key, values = entry["key"], entry["values"]
                for value in values:
                    label = f"{key.split('.')[-1]}={_fmt(value)}"
                    if only and label not in only:
                        continue
                    # 扫描点改的是 L/K/B，本身就在 run_name 里；但等于默认值的那一点
                    # 与其它轴的基线算出来同名，默认交给 main 组，不重复训练。
                    if not include_baseline and _dig(base, key) == value:
                        print(
                            f"[SKIP] {label} 等于当前默认值，基线点由 main 组提供"
                            "（--include-baseline 可强制跑）"
                        )
                        continue
                    for seed in seeds:
                        jobs.append(
                            _job(
                                group,
                                label,
                                seed,
                                [f"{key}={_fmt(value)}"],
                                total_env_steps,
                                "sweep",
                            )
                        )
        else:
            raise ValueError(f"未知实验分组 {group!r}（可选：main / ablation / sweep）")
    return _dedupe(jobs)


def _dedupe(jobs: list[dict]) -> list[dict]:
    """同一份**解析后的配置**被多个扫描轴展开时只留一条（`--include-baseline` 会遇到）。

    比较的是合并完成的 cfg（如 `algo.msg_bits=4` 与默认值 4 解析后完全相同），
    而不是 token 字面量；配置不同却算出同一个 run_name 属于真撞名，
    留给 `find_collisions` 报错，不会被这里悄悄吞掉。
    """
    seen: dict[tuple, dict] = {}
    out: list[dict] = []
    for job in jobs:
        key = (job["exp_name"], job["run_name"], job["seed"])
        twin = seen.get(key)
        if twin is not None and _canonical(twin["cfg"]) == _canonical(job["cfg"]):
            print(f"[DEDUPE] {job['variant']} 与 {twin['variant']} 配置相同，合并为一条")
            continue
        seen.setdefault(key, job)
        out.append(job)
    return out


def _canonical(cfg: dict) -> str:
    """把配置压成可比较的字符串；非 JSON 类型退化为 repr。"""
    return json.dumps(cfg, sort_keys=True, ensure_ascii=False, default=str)


def find_collisions(jobs: list[dict]) -> list[str]:
    """同一 (实验组, run_name) 出现多次就会覆盖目录，必须挡住。"""
    seen: dict[tuple[str, str], str] = {}
    problems: list[str] = []
    for job in jobs:
        key = (job["exp_name"], job["run_name"])
        if key in seen:
            problems.append(
                f"{key[0]}/{key[1]}：{seen[key]} 与 {job['variant']} 撞名"
            )
        else:
            seen[key] = job["variant"]
    return problems


def write_plan(jobs: list[dict], path: Path | None = None) -> Path:
    path = path or (ROOT / "results" / "_plan" / "plan.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(PLAN_COLUMNS))
        writer.writeheader()
        for job in jobs:
            writer.writerow({k: job[k] for k in PLAN_COLUMNS})
    return path


def _missing_runtime() -> list[str]:
    missing = []
    for mod in ("torch", "mpe2"):
        try:
            __import__(mod)
        except Exception:
            missing.append(mod)
    return missing


def _print_plan(jobs: list[dict]) -> None:
    header = f"{'group':<9} {'variant':<22} {'seed':>4} {'exp_name':<22} {'run_name':<40} {'steps':>7}"
    print(header)
    print("-" * len(header))
    for job in jobs:
        print(
            f"{job['group']:<9} {job['variant'][:22]:<22} {job['seed']:>4} "
            f"{job['exp_name'][:22]:<22} {job['run_name']:<40} {job['total_env_steps']:>7}"
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="展开并执行批量实验（负责人 E）")
    ap.add_argument("--list", action="store_true", help="列出可用分组与消融变体后退出")
    ap.add_argument("--groups", nargs="*", default=["main"], help="main / ablation / sweep")
    ap.add_argument(
        "--algos",
        nargs="*",
        default=None,
        help="main 组的算法轴（bac / ippo / mappo / ic3net / tarmac / qmix）；省略则用配置默认",
    )
    ap.add_argument("--seeds", nargs="*", type=int, default=None, help="种子列表，默认取 config.yaml 的 seed")
    ap.add_argument("--only", nargs="*", default=None, help="只跑这些变体/扫描点")
    ap.add_argument("--total-env-steps", type=int, default=None, help="覆盖 train.total_env_steps")
    ap.add_argument(
        "--include-baseline",
        action="store_true",
        help="扫描轴也跑默认值那一点（默认跳过，基线由 main 组提供）",
    )
    ap.add_argument("--execute", action="store_true", help="真的按顺序执行（默认只打印计划）")
    ap.add_argument("--keep-going", action="store_true", help="单个任务失败继续跑后面的")
    ap.add_argument("--show-commands", action="store_true", help="打印每条完整命令")
    ap.add_argument("--dry-run", action="store_true", help="等价于默认行为，显式表达用")
    args = ap.parse_args(argv)

    if args.list:
        for group in ("main", "ablation", "sweep", "robust"):
            cfg = load_config([f"experiment={group}"])
            exp = cfg.get("experiment") or {}
            if group == "ablation":
                names = [_variant(v)[0] for v in (exp.get("ablation") or [])]
                print(f"ablation : {', '.join(names)}")
            elif group == "sweep":
                for entry in exp.get("sweep") or []:
                    print(f"sweep    : {entry['key']} ∈ {entry['values']}")
            elif group == "robust":
                grid = (exp.get("robustness") or {}).get("drop_p")
                print(f"robust   : drop_p ∈ {grid}（评测网格，训练侧无任务）")
            else:
                print(f"{group:<8} : train.total_env_steps={((cfg.get('train') or {}).get('total_env_steps'))}")
        print("algos    : bac / ippo / mappo / ic3net / tarmac / qmix（--algos，仅 main 组）")
        return 0

    seeds = args.seeds if args.seeds else [int(load_config([])["seed"])]
    jobs = plan(
        args.groups, seeds, args.total_env_steps, args.only, args.include_baseline, args.algos
    )
    if not jobs:
        print("[WARN] 计划为空，检查 --groups / --only")
        return 1

    problems = find_collisions(jobs)
    if problems:
        print("[FAIL] 存在会互相覆盖的实验：")
        for line in problems:
            print("  " + line)
        return 2

    _print_plan(jobs)
    if args.show_commands or not args.execute:
        print()
        for job in jobs:
            print(job["command"])
    out = write_plan(jobs)
    print(f"\n[INFO] 计划 {len(jobs)} 个任务，清单已写 {out}")

    if not args.execute:
        print("[INFO] 未执行（默认只排班）。确认后加 --execute")
        return 0

    missing = _missing_runtime()
    if missing:
        print(f"[FAIL] 缺少训练依赖：{', '.join(missing)}。先按 requirements.txt 装好再 --execute")
        return 3

    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    failed = 0
    for index, job in enumerate(jobs, 1):
        print(f"\n[RUN {index}/{len(jobs)}] {job['exp_name']}/{job['run_name']} ({job['algo']})")
        print(job["command"])
        proc = subprocess.run(
            [sys.executable, "-m", job["runner"], *job["tokens"]],
            cwd=str(ROOT),
            env=env,
        )
        if proc.returncode != 0:
            failed += 1
            print(f"[FAIL] 退出码 {proc.returncode}")
            if not args.keep_going:
                print("[ABORT] 已中止；加 --keep-going 可跳过失败任务继续")
                return proc.returncode
    print(f"\n[DONE] {len(jobs) - failed}/{len(jobs)} 成功")
    return 0 if failed == 0 else 4


if __name__ == "__main__":
    raise SystemExit(main())
