"""作图：只读 results/*/metrics.csv 与 results/*/robust.csv（负责人 E）。

- `metrics.csv`：训练内指标（`train/*`）与周期评测指标（`eval/*`）在同一张表里，
  未评测的行是 `nan` / 空，读入时跳过。
- `robust.csv`：`evaluate.py` 按丢包率网格复测产物（列见 contracts 第 6 节）。
- 同一配置（去掉 `_s{seed}` 后缀）的多个 seed 按步数 / 丢包率求均值 ± 标准差，
  单 seed 时标准差记为 0（用 `ddof=0`），多 seed 用样本标准差（`ddof=1`）。
- label 带实验组前缀：`results/main/xxx_s0` 的 label 是 `main/xxx`，避免 `main` 与
  `ablation` 里同名的 run 被当成同一次跑的多 seed 合并。
- 缺 matplotlib 时只打印汇总并返回 0，CI 不装绘图库也能跑。

用法：
    python scripts/make_figures.py                     # 读 results/，写 results/figures/
    python scripts/make_figures.py --list               # 只打印汇总，不画图
    python scripts/make_figures.py --results results --out results/figures --dpi 200
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIG_DIR = RESULTS / "figures"

# run 名形如 formation_6n_bac_L4_K3_B2.0_s0_nogate：seed 之后可能跟变体后缀
_RUN_RE = re.compile(r"^(?P<stem>.+?)_s(?P<seed>\d+)(?P<variant>(?:_[A-Za-z0-9.\-]+)?)$")

TRAIN_METRICS = ("return", "success_rate", "formation_err", "collisions")
COMM_METRICS = ("bits", "speak_rate")
ROBUST_METRICS = ("success_rate", "formation_err", "bits")


def parse_run_name(name: str) -> dict:
    """把 run 目录名拆成「跨 seed 聚合用的 label」与 seed。"""
    m = _RUN_RE.match(name)
    if not m:
        return {"label": name, "seed": None}
    return {"label": m.group("stem") + m.group("variant"), "seed": int(m.group("seed"))}


def parse_run_path(path: Path, root: Path) -> dict:
    """run 目录 + 相对 `--results` 的位置 → label/seed。

    label 必须带实验组（`main` / `ablation` / …）：`main` 与 `ablation` 里同名的 run
    （如 `formation_6n_bac_L4_K3_B2.0_nogate`）只差变体后缀时，光看目录名会被当成
    同一次跑的多 seed 合并进同一条曲线。
    """
    info = parse_run_name(Path(path).name)
    try:
        parts = Path(path).relative_to(root).parts
    except ValueError:
        parts = ()
    if len(parts) > 1:
        return {"label": "/".join((*parts[:-1], info["label"])), "seed": info["seed"]}
    return info


def _num(text) -> float | None:
    """空串 / nan / 非数字统一当作缺失。"""
    if text is None:
        return None
    s = str(text).strip()
    if not s or s.lower() in ("nan", "none", "null"):
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return None if v != v else v


def _push(store: dict, label: str, col: str, x: float, v: float) -> None:
    store.setdefault(label, {}).setdefault(col, {}).setdefault(x, []).append(v)


def load_metrics(root: Path) -> dict:
    """读所有 metrics.csv → `{series: {label: {col: {x: [v...]}}}, seeds: {label: {int}}}`。"""
    series: dict = {}
    seeds: dict = {}
    for path in sorted(Path(root).rglob("metrics.csv")):
        info = parse_run_path(path.parent, Path(root))
        label = info["label"]
        if info["seed"] is not None:
            seeds.setdefault(label, set()).add(info["seed"])
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                step = _num(row.get("env_steps"))
                if step is None:
                    continue
                for col, raw in row.items():
                    if not col or col == "env_steps" or "/" not in col:
                        continue
                    v = _num(raw)
                    if v is None:
                        continue
                    _push(series, label, col, step, v)
    return {"series": series, "seeds": seeds}


GATE_COMMS = {"bac", "ic3net"}  # 有门控头的通信模块，其余（none/tarmac）没有


def _comm_of(run_dir: Path) -> str:
    """从 run 的 config.yaml 取 algo.comm；读不到就返回空串（不猜测）。"""
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        return ""
    try:
        import yaml

        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return ""
    return str((cfg.get("algo") or {}).get("comm", ""))


def load_robust(root: Path) -> dict:
    """读所有 robust.csv → `{series: {label: {col: {drop_p: [v...]}}}, seeds: ...}}`。

    注意：`speak_rate` 不能直接采信文件里的那一列。历史 robust.csv 是在口径修正前
    跑出来的——IPPO/MAPPO 写成 1.0（Actor 占位 gate），TarMAC 写成 0.0（按无门控一刀切）。
    这里统一按"到底发没发消息"重算：有门控头的信原值；没门控头时 bits>0 记 1.0、否则 0.0。
    """
    series: dict = {}
    seeds: dict = {}
    for path in sorted(Path(root).rglob("robust.csv")):
        info = parse_run_path(path.parent, Path(root))
        label = info["label"]
        comm = _comm_of(path.parent)
        if info["seed"] is not None:
            seeds.setdefault(label, set()).add(info["seed"])
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                x = _num(row.get("drop_p"))
                if x is None:
                    continue
                vals = {c: _num(v) for c, v in row.items() if c != "drop_p"}
                if comm and comm not in GATE_COMMS:
                    bits = vals.get("bits") or 0.0
                    vals["speak_rate"] = 1.0 if bits > 0 else 0.0
                for col, v in vals.items():
                    if v is None:
                        continue
                    _push(series, label, col, x, v)
    return {"series": series, "seeds": seeds}


def aggregate(points: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """`{x: [per-seed values]}` → (xs, mean, std, n)，xs 升序。"""
    xs = np.array(sorted(points), dtype=float)
    mean, std, n = [], [], []
    for x in xs:
        vals = np.asarray(points[float(x)], dtype=float)
        mean.append(float(vals.mean()))
        n.append(int(vals.size))
        std.append(float(vals.std(ddof=1)) if vals.size > 1 else 0.0)
    return xs, np.array(mean), np.array(std), np.array(n)


def short_labels(labels: list[str]) -> dict[str, str]:
    """削掉一批 label 的共同前缀（`formation_6n_bac_L4_K3_B2.0`），图例才不溢出版面。"""
    if not labels:
        return {}
    pref = labels[0]
    for lb in labels[1:]:
        i = 0
        while i < min(len(pref), len(lb)) and pref[i] == lb[i]:
            i += 1
        pref = pref[:i]
    pref = pref.rstrip("_-")
    out: dict[str, str] = {}
    for lb in labels:
        short = lb[len(pref):].lstrip("_-") if pref and lb.startswith(pref) else lb
        out[lb] = short or "base"
    return out


def summarise(kind: str, data: dict) -> None:
    for label in sorted(data["series"]):
        cols = data["series"][label]
        n = len(data["seeds"].get(label, ()))
        print(f"  [{kind}] {label}: {n or '?'} seed, {len(cols)} 列")
        for name in sorted(cols):
            pts = cols[name]
            xs, mean, std, _cnt = aggregate(pts)
            if not len(xs):
                continue
            print(f"      末点 {name} = {mean[-1]:.3f} ± {std[-1]:.3f} @ x={xs[-1]:g}")


def _setup_mpl():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception as exc:
        print(f"[WARN] 未装 matplotlib，跳过出图（{exc}）")
        return None


def _style() -> dict:
    return {"marker": "o", "ms": 3, "lw": 1.4}


def fig_train_eval(metrics: dict, plt, out: Path, dpi: int, titles: dict) -> Path | None:
    """2×2：return / success_rate / formation_err / collisions，train 实线 + eval 虚线。"""
    if not metrics["series"]:
        return None
    labels = sorted(metrics["series"])
    short = short_labels(labels)
    keys = [k for k in TRAIN_METRICS if any(
        f"{sp}/{k}" in metrics["series"][lb] for lb in labels for sp in ("train", "eval")
    )]
    if not keys:
        return None
    n = len(keys)
    rows = 2 if n > 1 else 1
    fig, axes = plt.subplots(rows, max(2, int(np.ceil(n / rows))), figsize=(11, 4.2 * rows), squeeze=False)
    handles, names = [], []
    for i, ax in enumerate(axes.ravel()):
        if i >= n:
            ax.axis("off")
            continue
        key = keys[i]
        for label in labels:
            for split, ls, alpha in (("train", "-", 1.0), ("eval", "--", 1.0)):
                col = f"{split}/{key}"
                pts = metrics["series"][label].get(col)
                if not pts:
                    continue
                xs, mean, std, _cnt = aggregate(pts)
                name = f"{short[label]} · {split}"
                h = ax.plot(xs, mean, ls, alpha=alpha, label=name, **_style())[0]
                ax.fill_between(xs, mean - std, mean + std, alpha=0.12)
                if name not in names:
                    handles.append(h)
                    names.append(name)
        ax.set_title(f"{key} (train vs eval)")
        ax.set_xlabel("env_steps")
        ax.grid(alpha=0.3)
    fig.suptitle(titles["main"])
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.legend(
        handles,
        names,
        loc="lower center",
        ncol=min(4, max(1, len(names))),
        fontsize=8,
        frameon=False,
    )
    path = out / "train_eval_metrics.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def fig_comm(metrics: dict, plt, out: Path, dpi: int, titles: dict) -> Path | None:
    """1×2：通信预算 bits 与 speak_rate（BAC 的核心观察量）。"""
    made = False
    if not metrics["series"]:
        return None
    labels = sorted(metrics["series"])
    short = short_labels(labels)
    keys = [k for k in COMM_METRICS if any(
        f"{sp}/{k}" in metrics["series"][lb] for lb in labels for sp in ("train", "eval")
    )]
    if not keys:
        return None
    fig, axes = plt.subplots(1, len(keys), figsize=(5.2 * len(keys), 4), squeeze=False)
    for i, ax in enumerate(axes.ravel()):
        key = keys[i]
        for label in labels:
            for split, ls in (("train", "-"), ("eval", "--")):
                pts = metrics["series"][label].get(f"{split}/{key}")
                if not pts:
                    continue
                xs, mean, std, _cnt = aggregate(pts)
                ax.plot(xs, mean, ls, label=f"{short[label]} · {split}", **_style())
                ax.fill_between(xs, mean - std, mean + std, alpha=0.12)
                made = True
        ax.set_title(f"{key} (comm budget)")
        ax.set_xlabel("env_steps")
        ax.grid(alpha=0.3)
    axes.ravel()[0].legend(fontsize=7)
    fig.suptitle(titles["comm"])
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    path = out / "comm_metrics.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path if made else None


def fig_robust(robust: dict, plt, out: Path, dpi: int, titles: dict) -> Path | None:
    """1×3：success_rate / formation_err / bits 随丢包率变化，多 seed 均值 ± 标准差。"""
    if not robust["series"]:
        return None
    labels = sorted(robust["series"])
    short = short_labels(labels)
    keys = [k for k in ROBUST_METRICS if any(k in robust["series"][lb] for lb in labels)]
    if not keys:
        return None
    fig, axes = plt.subplots(1, len(keys), figsize=(4.6 * len(keys), 4.2), squeeze=False)
    for i, ax in enumerate(axes.ravel()):
        key = keys[i]
        for label in labels:
            pts = robust["series"][label].get(key)
            if not pts:
                continue
            xs, mean, std, _cnt = aggregate(pts)
            ax.plot(xs, mean, label=f"{short[label]} ({int(_cnt.max())} seeds)", **_style())
            ax.fill_between(xs, mean - std, mean + std, alpha=0.12)
        ax.set_title(f"robust: {key} vs drop_p")
        ax.set_xlabel("drop_p")
        ax.set_ylabel(key)
        ax.grid(alpha=0.3)
    axes.ravel()[0].legend(fontsize=7)
    fig.suptitle(titles["robust"])
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    path = out / "robust_drop.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def fig_return(metrics: dict, plt, out: Path, dpi: int, titles: dict) -> Path | None:
    """单指标 return 曲线，给文档/汇报直接插图用（等价于旧版 curves.png）。"""
    if not metrics["series"]:
        return None
    labels = sorted(metrics["series"])
    short = short_labels(labels)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    drawn = False
    for label in labels:
        for split, ls in (("train", "-"), ("eval", "--")):
            pts = metrics["series"][label].get(f"{split}/return")
            if not pts:
                continue
            xs, mean, std, _cnt = aggregate(pts)
            ax.plot(xs, mean, ls, label=f"{short[label]} · {split}", **_style())
            ax.fill_between(xs, mean - std, mean + std, alpha=0.12)
            drawn = True
    if not drawn:
        plt.close(fig)
        return None
    ax.set_xlabel("env_steps")
    ax.set_ylabel("return")
    ax.set_title(titles["main"])
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = out / "curves_return.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(RESULTS), help="结果根目录")
    ap.add_argument("--out", default=str(FIG_DIR), help="图片输出目录")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--skip-robust", action="store_true", help="不读 robust.csv")
    ap.add_argument("--list", action="store_true", help="只打印汇总，不出图")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    root = Path(args.results)
    if not root.is_absolute():
        root = ROOT / root
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out

    metrics = load_metrics(root)
    robust = {"series": {}, "seeds": {}} if args.skip_robust else load_robust(root)
    if not metrics["series"] and not robust["series"]:
        print(f"[WARN] {root} 下没有 metrics.csv / robust.csv，先跑训练与评测")
        return 1

    print(f"[INFO] 结果根目录 {root}")
    summarise("metrics", metrics)
    summarise("robust", robust)
    if args.list:
        return 0

    plt = _setup_mpl()
    if plt is None:
        return 0

    out.mkdir(parents=True, exist_ok=True)
    titles = {
        "main": "6 methods × 3 seeds: train vs eval (mean ± std over seeds)",
        "comm": "6 methods × 3 seeds: communication budget (mean ± std over seeds)",
        "robust": "6 methods × 3 seeds: robustness vs packet drop (mean ± std over seeds)",
    }
    made = [
        fig_train_eval(metrics, plt, out, args.dpi, titles),
        fig_comm(metrics, plt, out, args.dpi, titles),
        fig_robust(robust, plt, out, args.dpi, titles),
        fig_return(metrics, plt, out, args.dpi, titles),
    ]
    for p in [p for p in made if p is not None]:
        print(f"[DONE] {p}")
    n = len([p for p in made if p is not None])
    print(f"[DONE] 共 {n} 张图 → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
