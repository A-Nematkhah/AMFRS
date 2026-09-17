# AMFRS ~12h GPU run (GST + F2 illumination + F3 PPO + robustness).
# Usage (from amfrs_env/):
#   powershell -ExecutionPolicy Bypass -File scripts/run_amfrs_12h.ps1
#   powershell -ExecutionPolicy Bypass -File scripts/run_amfrs_12h.ps1 -Llm groq -OutputDir results/amfrs_12h_groq
#
# Tips:
#   - Leave the machine plugged in; do not sleep/hibernate.
#   - If TEMP is tight:  $env:AMFRS_TEMP = "D:\amfrs_temp"
#   - On Windows OOM with nproc=4, pass:  -NumProcesses 2

param(
    [string]$Llm = "groq",
    [string]$Device = "cuda",
    [string]$OutputDir = "results/amfrs_12h",
    [int]$Seed = 425,
    [int]$NumProcesses = 4
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$Log = Join-Path $OutputDir "console.log"

Write-Host "AMFRS 12h profile -> $OutputDir (llm=$Llm device=$Device nproc=$NumProcesses)"
Write-Host "Logging to $Log"
Write-Host "Expected wall-clock ~10-14h (GPU/LLM dependent)."

$env:PYTHONUNBUFFERED = "1"
python -u scripts/run_amfrs.py `
    --profile 12h `
    --device $Device `
    --llm $Llm `
    --seed $Seed `
    --num-processes $NumProcesses `
    --output-dir $OutputDir `
    2>&1 | Tee-Object -FilePath $Log

exit $LASTEXITCODE
