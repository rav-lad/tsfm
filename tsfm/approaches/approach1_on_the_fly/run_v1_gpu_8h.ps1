# PowerShell entry point of run_v1_gpu_8h.sh (plain `bash` in PowerShell is the WSL launcher).
#
#     powershell -ExecutionPolicy Bypass -File approaches/approach1_on_the_fly/run_v1_gpu_8h.ps1

$ErrorActionPreference = "Stop"
$GitBash = Join-Path (Split-Path (Split-Path (Get-Command git).Source -Parent) -Parent) "bin\bash.exe"
& $GitBash (Join-Path $PSScriptRoot "run_v1_gpu_8h.sh")
