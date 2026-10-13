PY := .venv/Scripts/python

.PHONY: setup data parse index eval serve ui profile report test lint

setup:
	python -m venv .venv
	$(PY) -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
	$(PY) -m pip install -e ".[ml,serve,ui,dev]"

data:
	$(PY) scripts/download_data.py
	$(PY) scripts/parse_data.py
	$(PY) scripts/data_stats.py

parse:
	$(PY) scripts/parse_data.py

index:
	$(PY) scripts/build_index.py --chunker both
	$(PY) scripts/validate_phase2.py

eval:
	$(PY) -m citesec.eval.run_retrieval_eval --split dev --select --verify-repeat
	$(PY) scripts/analyze_failures.py
	$(PY) scripts/make_report.py

answers-dev:
	$(PY) -m citesec.eval.run_answer_eval --split dev --stage generate
	$(PY) -m citesec.eval.run_answer_eval --split dev --stage assemble
	$(PY) -m citesec.eval.run_answer_eval --split dev --stage judge
	$(PY) -m citesec.eval.run_answer_eval --split dev --stage metrics

answers-test:
	$(PY) -m citesec.eval.run_answer_eval --split test --stage retrieve
	$(PY) -m citesec.eval.run_answer_eval --split test --stage generate
	$(PY) -m citesec.eval.run_answer_eval --split test --stage assemble
	$(PY) -m citesec.eval.run_answer_eval --split test --stage judge
	$(PY) -m citesec.eval.run_answer_eval --split test --stage metrics
	$(PY) scripts/make_report.py

eval-test:
	$(PY) -m citesec.eval.run_retrieval_eval --split test
	$(PY) scripts/make_report.py

serve:
	$(PY) -m uvicorn citesec.api.main:app --port 8765

ui:
	$(PY) -m streamlit run app/streamlit_app.py

profile:
	$(PY) scripts/profile_latency.py

report:
	$(PY) scripts/make_report.py

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .
