# Windows equivalent of the Makefile. Usage: .\tasks.ps1 <setup|data|parse|test|lint>
param([Parameter(Position = 0)][string]$Task = "test")
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
switch ($Task) {
    "setup" { python -m venv .venv; & $py -m pip install -e ".[dev]" }
    "data"  { & $py scripts/download_data.py; & $py scripts/parse_data.py; & $py scripts/data_stats.py }
    "parse" { & $py scripts/parse_data.py }
    "index" { & $py scripts/build_index.py --chunker both; & $py scripts/validate_phase2.py }
    "eval"  { & $py -m citesec.eval.run_retrieval_eval --split dev --select --verify-repeat; & $py scripts/analyze_failures.py; & $py scripts/make_report.py }
    "eval-test" { & $py -m citesec.eval.run_retrieval_eval --split test; & $py scripts/make_report.py }
    "test"  { & $py -m pytest -q }
    "lint"  { & $py -m ruff check . }
    default { Write-Error "Unknown task $Task. Phases 2-5 add index, eval, serve." }
}
