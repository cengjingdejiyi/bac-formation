"""把多人交回的 results 包并进同一个实验组目录（负责人 A/E）。

多人分机器跑的场景：每人交一个 zip 或一个目录，里面是若干个
`formation_6n_<algo>_L.._s<seed>/`。汇总前先把它们合并到 `results/main/`，
再跑 `check_results.py`，避免"图里少一个 seed 却没人发现"。

用法：

    python scripts/merge_results.py --dest results/main \\
        --src D:/交接/E_ippo_qmix.zip --src D:/交接/C_bac_mappo \\
        [--overwrite]

规则：
* 源可以是 zip 也可以是目录，目录会被递归搜索，**含 metrics.csv 的目录**才算一个 run；
* run 名必须含 `_s<seed>`，否则拒绝（防止把 `_logs`、`figures` 之类搬进来）；
* 目标已存在同名 run 默认**跳过并报告**，`--overwrite` 才覆盖；
* 合并清单写到 `<dest>/_merge_manifest.tsv`（check_results 会忽略 `_` 开头的目录/文件）；
* 退出码：0 全部成功，1 有冲突或非法 run，2 参数/路径错误。
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

RUN_NAME_RE = re.compile(r"_s\d+")
MANIFEST = "_merge_manifest.tsv"


def _runs(root: Path) -> list[Path]:
    """找所有 run 目录（含 metrics.csv）；根目录自身是 run 时也算。"""
    candidates = [root] if (root / "metrics.csv").exists() else []
    candidates += [
        p
        for p in root.rglob("*")
        if p.is_dir() and (p / "metrics.csv").exists()
    ]
    candidates.sort(key=lambda p: len(p.parts))
    found: list[Path] = []
    for path in candidates:
        if any(parent in found for parent in path.parents):
            continue
        found.append(path)
    return found


def _prepare_sources(srcs: list[str], tmpdir: Path) -> list[tuple[str, Path]]:
    """返回 (来源标签, 根目录)。zip 解到临时目录。"""
    out = []
    for i, raw in enumerate(srcs):
        p = Path(raw)
        if not p.exists():
            raise FileNotFoundError(raw)
        if p.is_dir():
            out.append((str(p), p))
            continue
        if zipfile.is_zipfile(p):
            dest = tmpdir / f"src{i}"
            dest.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(p) as zf:
                zf.extractall(dest)
            # 不解容器目录：_runs 会递归找，少一层就会漏 run
            out.append((str(p), dest))
            continue
        raise ValueError(f"不是目录也不是 zip：{p}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="合并多人 results 交付")
    ap.add_argument("--dest", default="results/main", help="目标实验组目录")
    ap.add_argument("--src", action="append", required=True, help="源目录或 zip，可多次给")
    ap.add_argument("--overwrite", action="store_true", help="同名 run 覆盖")
    ap.add_argument("--dry-run", action="store_true", help="只报告，不复制")
    args = ap.parse_args(argv)

    dest = Path(args.dest)
    tmp_ctx = tempfile.TemporaryDirectory(prefix="merge_results_")
    tmpdir = Path(tmp_ctx.name)

    try:
        sources = _prepare_sources(args.src, tmpdir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[FAIL] {exc}")
        return 2

    dest.mkdir(parents=True, exist_ok=True)
    existing = {p.name for p in dest.iterdir() if p.is_dir()}
    seen: dict[str, str] = {}
    copied, skipped, rejected = [], [], []

    for label, root in sources:
        runs = _runs(root)
        if not runs:
            print(f"[WARN] {label} 下没找到含 metrics.csv 的 run 目录")
        for run in runs:
            name = run.name
            origin = f"{label}::{run.relative_to(root)}"
            if not RUN_NAME_RE.search(name):
                rejected.append((name, origin, "run 名不含 _s<seed>"))
                continue
            if name in seen:
                skipped.append((name, origin, f"与 {seen[name]} 重名"))
                continue
            if name in existing and not args.overwrite:
                skipped.append((name, origin, "目标已存在"))
                continue
            size = sum(f.stat().st_size for f in run.rglob("*") if f.is_file())
            print(
                f"[{'WOULD ' if args.dry_run else 'COPY  '}] {name}  "
                f"{size / 1e6:.1f} MB  <- {origin}"
            )
            if not args.dry_run:
                target = dest / name
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(run, target)
            seen[name] = origin
            copied.append((name, origin, size))

    if not args.dry_run and copied:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        man = dest / MANIFEST
        new_file = not man.exists()
        with man.open("a", encoding="utf-8") as f:
            if new_file:
                f.write("run\torigin\tMB\tmerged_at\n")
            for name, origin, size in copied:
                f.write(f"{name}\t{origin}\t{size / 1e6:.2f}\t{stamp}\n")

    print(f"\n复制 {len(copied)} 个、跳过 {len(skipped)} 个、拒绝 {len(rejected)} 个 → {dest}")
    for name, origin, why in skipped:
        print(f"  [SKIP] {name}: {why}（{origin}）")
    for name, origin, why in rejected:
        print(f"  [BAD ] {name}: {why}（{origin}）")
    if not args.dry_run and copied:
        print(f"\n下一步：python scripts/check_results.py --results {dest}")
    tmp_ctx.cleanup()
    return 0 if not rejected else 1


if __name__ == "__main__":
    raise SystemExit(main())
