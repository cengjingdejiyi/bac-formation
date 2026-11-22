"""从 results/ 生成 18-run 汇总表（负责人 E）。

只读 `results/<实验组>/<run_name>/metrics.csv`，按「去掉 `_s{seed}` 后缀」的运行名聚合，
输出 markdown 表 + 可选 CSV。表内数字与 `check_results` / `make_figures` 同源同口径：

- 末次评测点：每条 run 取 `metrics.csv` 中最后一个非空 `eval/return` 所在行（200k 步时是
  194560 那一档），不是训练滑窗值；
- 跨 seed 聚合：均值 ± **样本标准差**（`ddof=1`）；seed 数不足 3 时表尾标注 `n=1/2`，提醒别当结论；
- `L`/`B`/`K` 从运行名解析（`formation_6n_bac_L4_K3_B2.0_s0`），`L0` 表示无通信；
- `speak_rate` 直接读 `metrics.csv`：无通信方法（`algo.comm=none`）在 #33/#34 之后落盘即为 0，
  历史数据若为占位 1.0 会被标 `!` 提醒。

用法：
    python scripts/make_summary_table.py                       # 打印 markdown
    python scripts/make_summary_table.py --out summary.md      # 另存
    python scripts/make_summary_table.py --csv summary.csv     # 同时存 CSV
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import statistics as stats
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

_NAME_RE = re.compile(
    r"^(?P<env>[a-z0-9]+)_(?P<n>\d+)n_(?P<algo>[a-z0-9]+)_L(?P<l>[\d.]+)_K(?P<k>\d+)_B(?P<b>[\d.]+|inf)"
    r"_s(?P<seed>\d+)(?P<variant>(?:_[A-Za-z0-9.\-]+)?)$"
)

DISPLAY = {
    "bac": "BAC", "mappo": "MAPPO", "ippo": "IPPO", "qmix": "QMIX",
    "ic3net": "IC3Net", "tarmac": "TarMAC",
}

COLS = ("eval/return", "eval/formation_err", "eval/collisions", "comm/bits", "comm/speak_rate")


def parse_name(name: str) -> dict | None:
    m = _NAME_RE.match(name)
    if not m:
        return None
    d = m.groupdict()
    d["seed"] = int(d["seed"])
    d["group"] = d["env"]
    return d


def last_metrics(path: Path) -> dict | None:
    """返回最后一次评测行（含 eval/return 的最后一个非空行）里的关键列。"""
    if not path.exists():
        return None
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    pick = None
    for r in rows:
        v = r.get("eval/return")
        if v not in (None, "", "nan"):
            pick = r
    if pick is None:
        return None
    out = {}
    for c in COLS:
        v = pick.get(c)
        out[c] = None if v in (None, "", "nan") else float(v)
    return out


def collect(root: Path) -> dict:
    """`{组: {运行名(label): {seed: metrics}}}`"""
    data: dict = {}
    for metrics in sorted(root.rglob("metrics.csv")):
        run_dir = metrics.parent
        info = parse_name(run_dir.name)
        if info is None:
            continue
        vals = last_metrics(metrics)
        if vals is None:
            continue
        label = run_dir.name[: -len(f"_s{info['seed']}")] + info.get("variant", "")
        group = run_dir.parent.name
        data.setdefault(group, {}).setdefault(label, {})["meta"] = info
        data[group][label][info["seed"]] = vals
    return data


def _fmt(mean: float | None, sd: float | None, n: int, digits: int = 3) -> str:
    if mean is None:
        return "—"
    if n <= 1:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def render(root: Path) -> tuple[str, list[dict]]:
    data = collect(root)
    lines: list[str] = []
    records: list[dict] = []
    for group in sorted(data):
        lines.append(f"### 实验组 `{group}`\n")
        lines.append("| 方法 | L | B | seed 数 | eval/return | eval/formation_err | eval/collisions | comm/bits | speak_rate |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for label in sorted(data[group]):
            seeds = {k: v for k, v in data[group][label].items() if isinstance(k, int)}
            if not seeds:
                continue
            meta = data[group][label]["meta"]
            agg = {}
            for c in COLS:
                vs = [v[c] for v in seeds.values() if v[c] is not None]
                if vs:
                    m = sum(vs) / len(vs)
                    sd = stats.stdev(vs) if len(vs) > 1 else None
                    agg[c] = (m, sd, len(vs))
                else:
                    agg[c] = (None, None, 0)
            n = len(seeds)
            flag = "" if n >= 3 else f" ⚠n={n}"
            sr = agg["comm/speak_rate"]
            sr_txt = _fmt(*sr[:2], sr[2], 2) if sr[0] is not None else "—"
            if meta["algo"] in ("ippo", "mappo", "qmix") and sr[0] and abs(sr[0] - 1.0) < 1e-9:
                sr_txt += " !旧占位口径"
            row = [
                DISPLAY.get(meta["algo"], meta["algo"]),
                ("L" + str(meta["l"]).rstrip("0").rstrip(".")) if meta["l"] != "0" else "L0",
                ("B∞" if meta["b"] == "inf" else "B" + meta["b"]),
                str(n) + flag,
                _fmt(*agg["eval/return"][:2], agg["eval/return"][2], 3),
                _fmt(*agg["eval/formation_err"][:2], agg["eval/formation_err"][2], 3),
                _fmt(*agg["eval/collisions"][:2], agg["eval/collisions"][2], 3),
                _fmt(*agg["comm/bits"][:2], agg["comm/bits"][2], 2),
                sr_txt,
            ]
            lines.append("| " + " | ".join(row) + " |")
            records.append({
                "group": group, "run": label, "algo": meta["algo"], "L": meta["l"],
                "K": meta["k"], "B": meta["b"], "seeds": n,
                "eval/return": agg["eval/return"][0], "eval/return_std": agg["eval/return"][1],
                "eval/formation_err": agg["eval/formation_err"][0],
                "eval/formation_err_std": agg["eval/formation_err"][1],
                "eval/collisions": agg["eval/collisions"][0],
                "eval/collisions_std": agg["eval/collisions"][1],
                "comm/bits": agg["comm/bits"][0],
                "comm/speak_rate": agg["comm/speak_rate"][0],
            })
        lines.append("")
    n_runs = sum(1 for g in data.values() for lb in g.values() for k in lb if isinstance(k, int))
    lines.append(f"> 共 {n_runs} 个 run（seed 维度展开），{sum(len(g) for g in data.values())} 个运行名。"
                 "口径：末次评测（`eval/*`），跨 seed 样本标准差；无通信方法 `L0`。")
    return "\n".join(lines), records


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--out", default=None, help="markdown 输出路径")
    ap.add_argument("--csv", default=None, help="CSV 输出路径")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    root = Path(args.results)
    if not root.is_absolute():
        root = ROOT / root
    if not root.exists():
        print(f"[FAIL] {root} 不存在")
        return 2
    md, records = render(root)
    print(md)
    if args.out:
        Path(args.out).write_text(md + "\n", encoding="utf-8")
        print(f"[DONE] {args.out}")
    if args.csv and records:
        keys = list(records[0])
        with open(args.csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(records)
        print(f"[DONE] {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
