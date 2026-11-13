"""作图：只读 results/*/metrics.csv，输出均值±标准差曲线（负责人 E）。"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'


def load_all() -> dict:
    series: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for csv_path in RESULTS.rglob('metrics.csv'):
        algo = csv_path.parent.name.split('_')[2] if csv_path.parent.name.count('_') >= 2 else csv_path.parent.name
        with open(csv_path, encoding='utf-8') as f:
            for row in csv.DictReader(f):
                val = row.get('train/return')
                if not val:
                    continue
                series[algo][float(row['env_steps'])].append(float(val))
    return series


def main() -> int:
    series = load_all()
    if not series:
        print('[WARN] 没有 metrics.csv，先跑一次训练')
        return 1
    for algo, pts in series.items():
        xs = sorted(pts)
        means = [float(np.mean(pts[x])) for x in xs]
        stds = [float(np.std(pts[x])) for x in xs]
        print(f'  {algo}: {len(xs)} 点，末点 return = {means[-1]:.3f} ± {stds[-1]:.3f}')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:
        print('[WARN] 未装 matplotlib，跳过出图（pip install matplotlib）')
        return 0
    fig, ax = plt.subplots(figsize=(6, 4))
    for algo, pts in series.items():
        xs = sorted(pts)
        means = np.array([float(np.mean(pts[x])) for x in xs])
        stds = np.array([float(np.std(pts[x])) for x in xs])
        ax.plot(xs, means, marker='o', ms=3, label=algo)
        ax.fill_between(xs, means - stds, means + stds, alpha=0.15)
    ax.set_xlabel('env_steps')
    ax.set_ylabel('train/return')
    ax.legend(fontsize=8)
    out = RESULTS / 'curves.png'
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f'[DONE] {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
