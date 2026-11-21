#!/usr/bin/env bash
# 安装训练环境（macOS / Linux）
#
#   bash scripts/setup_env.sh            # 约束安装（requirements.txt）
#   bash scripts/setup_env.sh --lock     # 冻结复现（requirements-lock.txt）
set -euo pipefail

LOCK=0
[ "${1:-}" = "--lock" ] && LOCK=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3}"
echo "[1/4] Python 版本: $("$PY" -c 'import sys; print(sys.version)')"
VER="$("$PY" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
case "$VER" in
  3.11|3.13) ;;
  *) echo "[WARN] 推荐 3.11（CI 基线）或 3.13（本机已验证），当前 $VER" ;;
esac

echo "[2/4] 安装 torch（CPU 版）"
"$PY" -m pip install "torch>=2.2" --index-url https://download.pytorch.org/whl/cpu

echo "[3/4] 安装其余依赖"
MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"
FILE="requirements.txt"
[ "$LOCK" = "1" ] && FILE="requirements-lock.txt"
"$PY" -m pip install -r "$FILE" -i "$MIRROR"

echo "[4/4] 注册 src 包（可编辑）"
"$PY" -m pip install -e . --no-deps

echo
echo "自检："
"$PY" scripts/smoke.py
"$PY" -m pytest -q
"$PY" -c "import torch, mpe2; print('torch', torch.__version__, '| mpe2', mpe2.__version__)"
echo
echo "[DONE] 看到 backend=mpe2 才算真环境；多 seed 跑法见 docs/RUNBOOK.md"
