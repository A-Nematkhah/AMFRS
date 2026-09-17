# AMFRS ~6h GPU run (GST + F2 illumination + F3 PPO + robustness).
# Usage (from amfrs_env/):
#   powershell -ExecutionPolicy Bypass -File scripts/run_amfrs_6h.ps1
#   powershell -ExecutionPolicy Bypass -File scripts/run_amfrs_6h.ps1 -Llm groq -OutputDir results/amfrs_6h_groq

param(
    [string]$Llm = "groq",
    [string]$Device = "cuda",
    [string]$OutputDir = "results/amfrs_6h",
    [int]$Seed = 425,
    [int]$NumProcesses = 4
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$Log = Join-Path $OutputDir "console.log"

Write-Host "AMFRS 6h profile -> $OutputDir (llm=$Llm device=$Device nproc=$NumProcesses)"
Write-Host "Logging to $Log"

$env:PYTHONUNBUFFERED = "1"
python -u scripts/run_amfrs.py `
    --profile 6h `
    --device $Device `
    --llm $Llm `
    --seed $Seed `
    --num-processes $NumProcesses `
    --output-dir $OutputDir `
    2>&1 | Tee-Object -FilePath $Log

exit $LASTEXITCODE
