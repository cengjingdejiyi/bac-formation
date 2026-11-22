"""开跑前自检（3 分钟）：确认你这台机器上的代码/配置是"新口径"，避免跑到废数据。

    python scripts/preflight.py
    python scripts/preflight.py --algo qmix        # 只查 QMIX 那一路
    python scripts/preflight.py --skip-env         # 不想起真实环境时

检查项（任何一项 [FAIL] 都不要开跑）：

1. **代码版本**：`configs/algo/qmix.yaml` 有 `td_huber`/`target_clip`、`scripts/check_results.py`
   有 QMIX 发散判据 —— 缺这些说明 clone 还停在 2026-11-21 及以前（value_norm 或 QMIX 修复没拿到）；
2. **主配置**：`experiment=main` 的 `train.value_norm=true`、`train.reward_norm=false`、
   `eval_gate_mode` 合法；`env=formation` 的 `heading_mode/backend/success_dist` 是正式值；
3. **QMIX 配置**：`td_huber=true`、`target_clip>0`、`lr<=2.5e-4`、`updates_per_train<=2`；
4. **依赖与真后端**：`torch`/`mpe2` 可导入、起一个真实 `FormationCommEnv` 走一步，
   `backend=mpe2`（`mock` 说明 mpe2 没装上，这一批数据作废）；
5. **落盘**：`results/` 可写。

退出码 0 = 全绿；1 = 有 FAIL（按提示 `git pull` 或装环境后重跑）。
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    """通过只打名字；失败带上处置提示（不然一屏 PASS 里全是噪音）。"""
    if ok:
        print("[PASS] " + name)
        return
    print("[FAIL] " + name + (f"  {detail}" if detail else ""))
    FAILS.append(name)


def _load_config(tokens: list[str]) -> dict:
    from bacmarl.utils.config import load_config

    return load_config(tokens)


def check_code_version() -> None:
    qmix_yaml = (ROOT / "configs" / "algo" / "qmix.yaml").read_text(encoding="utf-8")
    check(
        "代码版本：QMIX 稳定化参数已入库",
        "td_huber" in qmix_yaml and "target_clip" in qmix_yaml,
        "缺 td_huber/target_clip → 你的 clone 停在 2026-11-21 之前，先 git pull",
    )
    checker = (ROOT / "scripts" / "check_results.py").read_text(encoding="utf-8")
    check(
        "代码版本：汇总自检含 QMIX 发散判据",
        "QMIX_REQUIRED_ALGO_KEYS" in checker,
        "缺判据 → 旧包会被当成好数据，先 git pull",
    )
    train_py = (ROOT / "src" / "bacmarl" / "runners" / "train.py").read_text(encoding="utf-8")
    check(
        "代码版本：训练入口有 value_target_scale（value_norm 已生效）",
        "value_target_scale" in train_py,
        "缺 → clone 停在 value_norm 修复之前，跑出来的 BAC/MAPPO 曲线不能用",
    )


def check_main_config() -> None:
    cfg = _load_config(["env=formation", "algo=bac", "experiment=main"])
    tr, env = cfg["train"], cfg["env"]
    check(
        "train.value_norm = true",
        bool(tr.get("value_norm")) is True,
        f"实际 {tr.get('value_norm')!r}：false 时价值目标会回到 O(100)，曲线作废",
    )
    check(
        "train.reward_norm = false",
        bool(tr.get("reward_norm", False)) is False,
        f"实际 {tr.get('reward_norm')!r}：和 value_norm 同时开会直接报错",
    )
    check(
        "train.eval_gate_mode 合法",
        str(tr.get("eval_gate_mode", "threshold")) in {"threshold", "sample"},
        f"实际 {tr.get('eval_gate_mode')!r}",
    )
    check(
        "env：正式标定参数",
        env.get("heading_mode") == "dynamic"
        and abs(float(env.get("success_dist", 0)) - 0.09) < 1e-9
        and abs(float(env.get("formation_radius", 0)) - 0.5) < 1e-9
        and abs(float(env.get("collision_coef", 0)) - 1.0) < 1e-9,
        f"heading={env.get('heading_mode')} success_dist={env.get('success_dist')} "
        f"R={env.get('formation_radius')} coll={env.get('collision_coef')}",
    )
    check(
        "train.total_env_steps = 200000",
        int(tr.get("total_env_steps", 0)) == 200000,
        f"实际 {tr.get('total_env_steps')}（正式实验是 3 seed × 200k）",
    )


def check_qmix_config() -> None:
    cfg = _load_config(["env=formation", "algo=qmix", "experiment=main"])
    algo = cfg["algo"]
    check(
        "qmix.td_huber = true",
        algo.get("td_huber") is True,
        f"实际 {algo.get('td_huber')!r}：false 时 MSE 自举会发散到 1e6 级",
    )
    clip = algo.get("target_clip")
    check(
        "qmix.target_clip > 0",
        clip not in (None, 0, "none", "None") and float(clip) > 0,
        f"实际 {clip!r}：不夹逼会在 43k 步后开始放大",
    )
    check(
        "qmix.lr <= 2.5e-4",
        float(algo.get("lr", 0)) <= 2.5e-4 + 1e-12,
        f"实际 {algo.get('lr')}：5e-4 是旧发散配置",
    )
    check(
        "qmix.updates_per_train <= 2",
        int(algo.get("updates_per_train", 99)) <= 2,
        f"实际 {algo.get('updates_per_train')}：4 是旧发散配置",
    )


def check_env_backend() -> None:
    try:
        import torch
        import mpe2
    except Exception as exc:  # pragma: no cover - 缺依赖
        check("torch / mpe2 可导入", False, str(exc))
        return
    check("torch / mpe2 可导入", True, f"torch {torch.__version__}, mpe2 {mpe2.__version__}")

    import numpy as np

    from bacmarl.envs.formation_comm import EnvConfig, FormationCommEnv

    cfg = EnvConfig.from_dict(
        {**_load_config(["env=formation"])["env"], "max_steps": 20}
    )
    env = FormationCommEnv(cfg)
    try:
        step = env.reset(seed=0)
        backend = getattr(getattr(env, "world", None), "backend", None) or getattr(
            env, "backend", None
        )
        check(
            "真实后端 backend=mpe2",
            str(backend) == "mpe2",
            f"实际 {backend}：mock 说明 mpe2 没装上，这批数据作废",
        )
        n = cfg.n_agents
        rng = np.random.default_rng(0)
        for _ in range(5):
            out = env.step(rng.integers(0, 5, size=n))
        obs = np.asarray(out.obs)
        check(
            "环境步进形状正确",
            obs.shape[0] == n,
            f"obs={obs.shape}，期望 ({n}, …)",
        )
    finally:
        env.close()


def check_writable() -> None:
    try:
        with tempfile.NamedTemporaryFile(dir=ROOT / "results", prefix="_preflight_", delete=True):
            pass
        check("results/ 可写", True, str(ROOT / "results"))
    except Exception as exc:  # pragma: no cover
        check("results/ 可写", False, str(exc))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="开跑前自检（3 分钟）")
    ap.add_argument("--algo", default="all", help="all | qmix | bac | ippo …")
    ap.add_argument("--skip-env", action="store_true", help="跳过真实环境步进")
    args = ap.parse_args(argv)

    check_code_version()
    check_main_config()
    if args.algo in ("all", "qmix"):
        check_qmix_config()
    if not args.skip_env:
        check_env_backend()
    check_writable()

    print()
    if FAILS:
        print(f"[FAIL] {len(FAILS)} 项不通过，先别开跑：")
        for name in FAILS:
            print(f"  - {name}")
        print("\n常见处置：git pull（到 d51987e5 之后）→ pwsh -File scripts/setup_env.ps1 或 bash scripts/setup_env.sh → 重跑本脚本")
        return 1
    print("[PREFLIGHT OK] 代码/配置/环境都是新口径，可以开跑。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
