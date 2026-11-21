# End-to-end rebuild on Windows PowerShell.
# Usage:  powershell -ExecutionPolicy Bypass -File run_all.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv")) {
    uv venv --python 3.12 .venv
    uv pip install --python .venv\Scripts\python.exe -r requirements.txt
    uv pip install --python .venv\Scripts\python.exe -e . --no-deps
}

$py = ".venv\Scripts\python.exe"

& $py scripts\01_fetch_sources.py
& $py scripts\02_mine.py
& $py scripts\03_build_pool.py
& $py scripts\04_build_dataset.py
& $py scripts\05_train_baseline.py
& $py scripts\06_train_transformer.py
& $py scripts\07_evaluate.py
& $py scripts\08_error_analysis.py
& $py scripts\15_record_env.py

Write-Host "Done. See reports\metrics.json and reports\error_analysis.md"
