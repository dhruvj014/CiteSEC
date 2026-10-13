# One-time test run of the answer evaluation. Run after dev is finished and the judge is validated.
# Usage: .\scripts\run_answers_test.ps1   (detach with Start-Process, it takes about 85 minutes)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
$env:PYTHONUTF8 = 1
$py = Join-Path $root ".venv\Scripts\python.exe"
foreach ($stage in "retrieve", "generate", "assemble", "judge", "metrics") {
    Write-Host "$(Get-Date -Format s) stage $stage"
    & $py -m citesec.eval.run_answer_eval --split test --stage $stage
    if ($LASTEXITCODE -ne 0) { Write-Host "stage $stage failed"; exit $LASTEXITCODE }
}
& $py scripts/make_report.py
Write-Host "$(Get-Date -Format s) DONE"
