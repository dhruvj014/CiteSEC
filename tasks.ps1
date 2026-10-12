# Windows equivalent of the Makefile. Usage: .\tasks.ps1 <setup|data|parse|test|lint>
param([Parameter(Position = 0)][string]$Task = "test")
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
switch ($Task) {
    "setup" { python -m venv .venv; & $py -m pip install -e ".[dev]" }
    "data"  { & $py scripts/download_data.py; & $py scripts/parse_data.py; & $py scripts/data_stats.py }
    "parse" { & $py scripts/parse_data.py }
    "test"  { & $py -m pytest -q }
    "lint"  { & $py -m ruff check . }
    default { Write-Error "Unknown task $Task. Phases 2-5 add index, eval, serve." }
}
