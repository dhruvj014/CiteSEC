# Phase 5 notes: serving, observability, packaging, report

## What was built
- `pipeline.py`: one `RagPipeline` (retrieve, rerank, confidence gate, generate, citation verify) used by the API, the UI and the profiler. It is the same sequence the evaluation measures as system S4, with retrieval settings read from `configs/retrieval_selected.json` and the threshold from `configs/calibration.json`. A lock serializes questions because the GPU models and the SQLite store are shared.
- `api/main.py`, `api/metrics.py`: FastAPI service. `POST /query` (question, optional `top_k` and company or year filters), `GET /health`, `GET /metrics` (Prometheus), `/docs` (OpenAPI). Indexes and models load once at startup (about 45 seconds). Prometheus: a latency histogram per stage, query and abstention counters (gate versus model), an invalid-citation counter, an error counter and an index-size gauge. If Ollama is down the API returns 503 with a readable message.
- `app/streamlit_app.py`, `ui.py`: Streamlit demo that calls the API. Answer with citation badges, abstention banner, confidence, per-stage latency, cited passages with page numbers, and everything that was retrieved.
- `scripts/profile_latency.py` and `results/latency.json`: 100 queries end to end plus two optimization experiments.
- `.github/workflows/ci.yml`: ruff and pytest only. Model libraries moved to an optional `ml` extra, so CI installs no torch and downloads no model.
- `scripts/make_report.py` now also rewrites the results block in `README.md` between markers, so README numbers come from the same JSON as `BENCHMARKS.md`.

## Measured (`results/latency.json`, `BENCHMARKS.md`)
End to end through the real pipeline, 100 answerable test questions, RTX 4060 laptop, local Qwen2.5 7B, after warm-up:

| Outcome | n | Total p50 / p95 (ms) |
|---|---|---|
| all | 100 | 4,051 / 13,323 |
| answered | 61 | 6,050 / 14,210 |
| refused by the model | 33 | 1,762 / 4,599 |
| refused by the gate | 6 | 226 / 233 |

- Generation is 87% of the time at p50 (3.5 s of 4.05 s). Retrieval plus rerank is about 0.22 s, of which the cross-encoder is 208 ms.
- The answer table in the report shows a lower S4 p50 (2.7 s) because it includes the 41 unanswerable questions, which are refused quickly.

### Optimizations (measured, one adopted)
| Experiment | Result | Decision |
|---|---|---|
| Rerank 30 candidates instead of 50 | rerank 213 to 129 ms (-39%). Dev MRR 30.7 to 29.7, Hit@1 20 to 18, Hit@5 unchanged | Not adopted. It saves about 85 ms of a 4,051 ms query (2%) and costs a little quality |
| Rerank 20 candidates | rerank 104 ms. Dev Hit@5 44, MRR 30.5 | Not adopted, same reasoning, and 50 questions cannot separate these differences |
| Query-embedding cache | embed 18.4 to 0.0 ms, retrieval p50 246 to 220 ms on a repeated query | Adopted. It only helps repeats (UI reruns, retries), unique queries see no change |

The real lever is generation length and model size, which I did not touch because it changes answers.

## Bugs found by running it for real
- Port 8000 is taken on this machine by a VM service, my first health check hit that service and looked fine. Default port is now 8765.
- The mocked UI tests passed, but the real UI showed money amounts such as `$4,200,000,000 ... $3,800,000,000` parsed as LaTeX math and garbled. Fixed by escaping bare `$` before converting the model's `\[...\]`, with a regression test. A running Streamlit server also kept a stale copy of the fixed module until restarted.
- Typing `FY2022` in the years box was ignored (no word boundary between `Y` and `2`). Fixed.
- The profiler crashed when the filter left no candidates (no rerank stage ran). Fixed.
- The SQLite store is now opened with `check_same_thread=False` for API worker threads, with access serialized by the pipeline lock.

## Deviations from the brief
- **Docker Compose, Prometheus and Grafana dashboard**: skipped by decision. Metrics are exposed at `/metrics` and tested, nothing scrapes them.
- **Demo GIF or video**: skipped by decision.
- **`make setup data index serve` on a clean clone**: not run end to end. The PDF parse takes hours and the downloads are slow here. I documented measured per-stage times in the README instead and simulated the CI install in a fresh virtual environment (158 tests and ruff pass with no ML libraries).
- **CI green**: simulated on Windows only. The workflow itself has not run on GitHub.
- **Public repository**: your step. Before making it public, check `git status` for `.env`, PDFs and `logs/`.

## Limitations of the serving layer
- One question at a time (a lock around the GPU models). No authentication, rate limiting or streaming, and answers arrive only when generation finishes (up to about 14 s).
- An explicit year filter is exact. The evaluation's automatic filter also admits the following year, so a caller filtering to FY2020 may get a refusal where the automatic path would have answered (seen once while testing).
- Latency numbers are for this machine and a 7B model. A different GPU or Ollama version will change them.

## Candidate resume bullets (every number is in `BENCHMARKS.md`)
- Built a citation-grounded RAG system over **207 SEC filings** (FinanceBench plus EDGAR) with a table-aware chunker, **BM25 + dense retrieval, Reciprocal Rank Fusion** and a cross-encoder reranker, lifting page-level Hit@5 on a held-out test split from **24.0% to 45.0%** against a naive-chunk dense baseline.
- Added a calibrated abstention gate and a citation verifier that remove unsupported references. The full system refused **41 of 41** unanswerable questions, cut hallucinated answers from **77.0% to 27.0%** against a closed-book LLM and roughly doubled accuracy (**20.0% to 39.0%**) with **95.9%** valid citations.
- Shipped a **FastAPI** service with **Prometheus** metrics and a **Streamlit** demo. A single RTX 4060 with a local 7B model answers at **4.1 s p50** with retrieval and reranking taking about 0.2 s of that. Evaluated with a fixed dev/test split, a validated LLM judge and run-to-run variance measurements.

Wording notes: do not say hybrid retrieval beat dense (it did not here). Say the improvement comes from structured chunking and reranking. Do not quote the faithfulness number. The judge labels were written by an AI assistant, audit them before presenting them as human-validated.

## Interview questions
1. *Why is generation 87% of latency and what would you do about it?* The 7B model decodes at about 34 tokens per second (median over the 63 longer S2 test answers) and the prompt asks for step-by-step working, so a typical answer is 241 tokens and the cap is 700. Options are a shorter answer format, a smaller or quantized model, or speculative decoding. I measured retrieval optimizations instead and rejected rerank-30 because 2% end-to-end savings did not justify the quality cost.
2. *How do you keep the API safe with shared GPU models?* One lock serializes questions, and the SQLite connection is opened for multi-thread use under that lock. That is fine for a demo, not for throughput. A real deployment would put the models behind a batching server.
3. *What does the abstention gate cost and buy?* It saves generation on 6 of 100 answerable and 18 of 41 unanswerable questions, and removed 2 wrong answers on test, but it is entity-blind (AUC 0.67), so the model's own refusal does most of the work.
4. *How do you know the README numbers are right?* They are generated from `results/*.json` between markers, the same source as `BENCHMARKS.md`, and a test asserts the headline numbers appear in both.
5. *What did running it for real find that tests did not?* Currency being rendered as math, a port collision, and a stale server module. The mocked UI tests could not see any of them, which is why I checked the rendered page.
