# GPU environment for an AMD Radeon RX 6800 XT (gfx1030) on Windows, next to the CPU `.venv`.
#
#     powershell -ExecutionPolicy Bypass -File scripts/setup_rocm_windows.ps1
#     .venv-rocm\Scripts\python.exe scripts/train.py ...      # then run scripts with this interpreter
#
# PyTorch comes from AMD's TheRock stable index (ROCm bundled in the wheel; gfx1030 is
# "release ready" there on Windows). Its latest stable torch (2.14) is below the
# `torch>=2.15` of pyproject.toml, which only pins the CPU / CUDA lock: the project is
# installed with --no-deps and every other dependency at its uv.lock version.
# Other RDNA2 / RDNA3 cards: change $Device (e.g. device-gfx1100 for an RX 7900).

$ErrorActionPreference = "Stop"
$Device = "device-gfx1030"
$Index = "https://stable.repo.amd.com/rocm/whl-next/"
$Venv = ".venv-rocm"
$Python = "$Venv\Scripts\python.exe"

Set-Location (Split-Path $PSScriptRoot -Parent)
if (-not (Test-Path $Python)) { uv venv $Venv --python 3.13 }
uv pip install --python $Python --index-url $Index "torch[$Device]"

$req = Join-Path $env:TEMP "tsfm-rocm-requirements.txt"
uv export --frozen --no-hashes --no-emit-project --no-dev --no-header |
    Where-Object { $_ -notmatch '^(torch|nvidia|cuda|triton)' } |
    Set-Content -Encoding utf8 $req
uv pip install --python $Python -r $req pytest
uv pip install --python $Python --no-deps -e .

& $Python -c "import torch; ok = torch.cuda.is_available(); print(torch.__version__, 'GPU', ok, torch.cuda.get_device_name(0) if ok else '-'); assert ok, 'no GPU: update the Adrenalin driver'"
