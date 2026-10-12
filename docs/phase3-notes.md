# Phase 3 notes: retrieval stack and retrieval evaluation

## What was built
- `eval/splits.py`: stratified (by `question_type`), seeded split of the 150 questions into dev (50) and test (100). Written once to `data/eval/split.json` by `scripts/make_split.py`, which never overwrites it.
- `retrieval/hybrid.py`: `rrf_fuse` (k=60, deterministic tie-break by first appearance), and `Retriever` with modes `dense`, `bm25`, `hybrid`, optional rerank, optional filter, and per-stage timing (embed query, dense, bm25, fuse, filter, fetch, rerank).
- `retrieval/rerank.py`: bge-reranker-base cross-encoder, fp16, scores are sigmoid values in [0, 1], input pairs truncated at 512 tokens.
- `retrieval/filters.py`: company alias table (curated FinanceBench names plus every company and ticker in the index) and a fiscal-year parser, used only in filtered mode.
- `index/dense.py` and `index/bm25.py` gained an `allowed` argument (FAISS ID selector, bm25s weight mask) so the filter restricts candidates before ranking.
- `eval/retrieval_metrics.py`: Hit@k, evidence Recall@k, MRR, doc-level Hit@k and MRR at k in {1, 3, 5, 10}. Unit tested on a hand-checked example.
- `eval/run_retrieval_eval.py`: the A to G ablation grid, per-stage p50/p95 latency, `--verify-repeat` determinism check, `--select` (dev only), and a guard that refuses to overwrite the test results.
- `eval/report.py` and `scripts/make_report.py`: generate `BENCHMARKS.md` from `results/*.json`. Regenerating gives an identical file.
- `scripts/dump_failures.py` and `scripts/analyze_failures.py`: inspect and classify misses. Analysis is in `docs/retrieval_failures.md`.

## Process followed
1. All choices were made on dev. The retrieval config was chosen by a stated rule (max Hit@5, then MRR, then lowest id, among structured configs C to G) and written to `configs/retrieval_selected.json`. It selected **G**.
2. The seeded dev rerun produced identical metrics (`repeat_run_identical: true` in `retrieval_dev.json`).
3. The test grid was then run **once** (`retrieval_test.json`). The harness refuses a second test run without `--overwrite-test`.

## Results (percent, from `BENCHMARKS.md`)
Test split, 100 questions, open retrieval unless noted:

| Config | Hit@5 | MRR | Doc-Hit@5 |
|---|---|---|---|
| A naive + dense | 24.0 | 13.5 | 74.0 |
| C structured + dense | 46.0 | 30.7 | 92.0 |
| E structured + hybrid RRF | 28.0 | 22.5 | 77.0 |
| F E + rerank | 45.0 | 32.6 | 95.0 |
| G F + company/year filter | 49.0 | 35.6 | 97.0 |

- **Chunking is the biggest single effect:** dense Hit@5 goes from 24.0 (naive) to 46.0 (structured) on test, and from 24.0 to 34.0 on dev.
- **Reranking buys ranking quality:** E to F raises test MRR from 22.5 to 32.6 and Hit@5 from 28.0 to 45.0 for about 240 ms more per query (p50 total 61 ms for E, 301 ms for F).
- **Document choice is mostly solved, page choice is not:** Doc-Hit@5 is 95.0 for F while Hit@5 is 45.0.
- **Headline open-retrieval comparison:** naive dense baseline A (24.0) versus the full open pipeline F (45.0). G adds a filter that assumes the question names its company, so quote F as the open-retrieval number and G separately.

## Results that went against the brief's expectation
- **Hybrid RRF is worse than dense alone** in this setup: test Hit@5 28.0 (E) versus 46.0 (C), dev 30.0 versus 34.0. BM25 alone is weak (13.0 test, 14.0 dev), because FinanceBench questions are long natural-language prompts ("You are an investment banker...") that dilute the few discriminating terms, and the corpus has many similar filings. Equal-weight RRF lets the weak ranker pull good dense results down. After reranking, F (45.0) is about level with dense alone (C 46.0), so in this grid BM25 contributes nothing measurable once a reranker is on. I did not run a dense plus rerank config, so I cannot say if BM25 adds value inside the reranked pipeline. That is the obvious next ablation.
- Because of this, do not claim "hybrid beats dense" anywhere. The defensible claim is "structured chunking plus reranking".

## Changes made after looking at dev results (disclosed)
- BM25 tokenizer: `3M` and `10K` were split into a digit and a letter. Fixed. Dev BM25 Hit@5 went from 16.0 to 14.0, so it did not help, it was fixed because it was wrong.
- Year parser: added two-digit fiscal years (`FY22`). Alias table: added `jnj`. Both came from dev misses. No test question was read to build aliases or patterns.
- `Retriever` returns top 50 for every config, so MRR is MRR@50.

## Deviations and limitations
- The reranker weights were fetched with a segmented downloader because `snapshot_download` stalled. The file is `BAAI/bge-reranker-base` `model.safetensors` placed in the Hugging Face cache.
- Reranker scores are sigmoid probabilities, which is what the Phase 4 abstention threshold will be calibrated on.
- Naive chunks span pages, so their page is the start page. This is part of why naive scores low, and is an intended property of the baseline.
- Page-level metrics apply to FinanceBench documents only. EDGAR chunks (`page = None`) are distractors and never count as hits.
- The filter is a hard filter. It returns an empty list when the question's year does not match the filing year that holds the answer (see failure 9 in `retrieval_failures.md`).
- Latency is single-query, one GPU, stage times are noisy at the millisecond level (query embedding varied between 15 and 38 ms across configs).
- Single benchmark, 100 test questions: differences of a few points are within noise (one question is one point).

## Interview questions
1. *Why is the hybrid worse than dense, and what would you do?* Equal-weight RRF treats both rankers as equally trustworthy, but BM25 here has Hit@5 of 13 versus 46 for dense, because long conversational prompts dilute lexical signal. I would run dense plus rerank to see if BM25 adds anything once reranked, try weighted RRF, and strip prompt boilerplate from the BM25 query. I report the negative result rather than hiding it.
2. *Why open retrieval as the primary setting?* A company filter assumes the question names its company and uses a hand-built alias table. Several questions do not (the Foot Locker nominees question). Open retrieval measures the system, filtered mode measures the system plus that assumption.
3. *What does the reranker cost and buy?* About 240 ms per query on an RTX 4060 for fifty pairs, and it moves test MRR from 22.5 to 32.6 and Hit@5 from 28.0 to 45.0 over the fused list.
4. *How do you know you are not overfitting to dev?* The config rule was fixed before looking at test, the test grid ran once, and dev-informed changes are listed. Dev and test move together: structured beats naive and reranking beats plain hybrid on both.
5. *Why is page-level Hit@5 so much lower than document-level?* 22 of 30 dev misses have a gold page that is a financial statement table, and 16 of 30 have duplicate pages occupying top-5 slots. The question asks for a derived number and the evidence is a statement of line items, so there is a semantic gap. Collapsing to one chunk per page and soft filters are the next steps.
