# 安装训练环境（Windows / PowerShell）
#
#   pwsh -File scripts/setup_env.ps1            # 约束安装（requirements.txt）
#   pwsh -File scripts/setup_env.ps1 -Lock      # 冻结复现（requirements-lock.txt）
#
# 默认走清华 PyPI 镜像 + PyTorch CPU 源（无 GPU 时不装 CUDA 版，省 2 GB）。

param(
    [switch]$Lock,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

Write-Host "[1/4] Python 版本"
& $Python -c "import sys; print(sys.version)"
$ver = & $Python -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"
if ($ver -notin @("3.11", "3.13")) {
    Write-Host "[WARN] 推荐 3.11（CI 基线）或 3.13（本机已验证），当前 $ver"
}

Write-Host "[2/4] 安装 torch（CPU 版）"
$torchArgs = @("-m", "pip", "install", "torch>=2.2", "--index-url", "https://download.pytorch.org/whl/cpu")
& $Python @torchArgs
if ($LASTEXITCODE -ne 0) { throw "torch 安装失败" }

Write-Host "[3/4] 安装其余依赖"
$Mirror = "https://pypi.tuna.tsinghua.edu.cn/simple"
$file = if ($Lock) { "requirements-lock.txt" } else { "requirements.txt" }
& $Python -m pip install -r $file -i $Mirror
if ($LASTEXITCODE -ne 0) { throw "$file 安装失败" }

Write-Host "[4/4] 注册 src 包（可编辑）"
& $Python -m pip install -e . --no-deps
if ($LASTEXITCODE -ne 0) { throw "pip install -e . 失败" }

Write-Host ""
Write-Host "自检："
& $Python scripts/smoke.py
& $Python -m pytest -q
& $Python -c "import torch, mpe2; print('torch', torch.__version__, '| mpe2', mpe2.__version__)"
Write-Host ""
Write-Host "[DONE] 看到 backend=mpe2 才算真环境；多 seed 跑法见 docs/RUNBOOK.md"
