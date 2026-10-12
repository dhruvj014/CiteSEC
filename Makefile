PY := .venv/Scripts/python

.PHONY: setup data parse index eval serve test lint

setup:
	python -m venv .venv
	$(PY) -m pip install -e ".[dev]"

data:
	$(PY) scripts/download_data.py
	$(PY) scripts/parse_data.py
	$(PY) scripts/data_stats.py

parse:
	$(PY) scripts/parse_data.py

index:
	@echo "Phase 2"

eval:
	@echo "Phase 3-4"

serve:
	@echo "Phase 5"

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .
