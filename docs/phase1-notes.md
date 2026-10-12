# Phase 1 notes: foundations and data ingestion

## What was built
- Scaffold: `pyproject.toml` with pinned deps, ruff, pytest, `Makefile` and `tasks.ps1` (Windows runner), `.gitignore`, `.env.example`, `configs/default.yaml`.
- `citesec.config` (YAML to typed paths) and `citesec.schemas` (pydantic: Question, Evidence, DocInfo, PageRecord, EdgarRecord, Chunk).
- `scripts/download_data.py`: FinanceBench over HTTPS (GitHub tree API plus raw files, size-checked skip so reruns are no-ops) and EDGAR via `edgartools` (throttled to 5 req/s, per-company manifest checkpoint, HTML cached on disk).
- `ingest/financebench.py`, `ingest/pdf_parser.py`, `ingest/html_parser.py`, `ingest/edgar.py`.
- `scripts/parse_data.py` (parallel PDF parse), `scripts/data_stats.py`, `scripts/validate_phase1.py`.

## Findings that differ from the brief
- **Evidence key name:** released JSONL uses `doc_name` inside each evidence dict, not `evidence_doc_name`. The loader accepts both and normalizes to `evidence_doc_name`.
- **PDF count:** the repo holds 368 PDFs (360 listed in the metadata file), but the 150 questions reference only 84. Only the 84 gold PDFs are parsed by default. The other PDFs can be added as free distractors with `parse_data.py --all-pdfs`.
- **Python 3.12** was used instead of 3.11 (only version installed). `requires-python >=3.11`.
- **Parse speed:** `pymupdf4llm` takes 2 to 4 minutes per long 10-K, so parsing runs in a process pool and is cached per document.
- **No git clone:** FinanceBench is fetched over HTTPS so the tooling never touches git.

## Measured results (from `results/`)
- Gold corpus: 150 questions, 189 evidence entries, 84 gold PDFs, all parsed (`data_stats.json`).
- **Zero-indexing confirmed:** 187 of 189 evidence entries fuzzy-match at page offset 0, none at -1 or +1. The 2 misses score 82.5 and 79.0 against an 85 threshold and are table-heavy pages (Amcor earnings table, Best Buy 10-Q), a layout difference not an index error (`phase1_validation.json`).
- EDGAR: 123 filings (118 10-K, 5 10-Q) across 40 companies, 3,177 section blocks. Holdout tickers were never downloaded.
- Total searchable docs so far: 84 gold PDFs + 123 EDGAR filings = 207. This is slightly above the brief's 100 to 200 target, reported honestly in Phase 3 results.

## Known limitations
- Section splitting: 107 of 119 10-Ks yield an Item 7. Filers without Item headings (Morgan Stanley, Goldman Sachs and a few others) come out as one or two huge blocks, so Phase 2 must window large blocks by size as a fallback.
- The first version of the splitter missed headings rendered as table rows with links and non-breaking spaces (Amgen, BMY). Fixed with `clean_line` plus a guard against prose cross-references like "Item 7 of this report", with a unit test.
- A few `SGML header declares N documents but M parsed` warnings from edgartools. Primary 10-K HTML was still retrieved.
- Parse was slow because 12 workers contend for CPU. Single-page benchmark showed table detection is not the bottleneck (about 1.1 s per page under every option).

## Design choices
- Atomic writes (`.tmp` then rename) for every cache, so interrupted runs never leave partial files and reruns resume.
- Holdout tickers (Type A unanswerable questions) are listed in config and never downloaded. FinanceBench companies are filtered out of the EDGAR pool at runtime by normalized name match.
- Page numbers are kept zero-indexed everywhere, matching the dataset. The off-by-one check is empirical, see `results/phase1_validation.json`.

## Interview questions
1. *Why verify the page index convention?* Retrieval metrics key on (doc, page). An off-by-one would silently zero out Hit@k. The script compares evidence text against pages idx-1, idx, idx+1 over all gold evidence.
2. *Why cache EDGAR HTML rather than markdown?* HTML is the raw source of truth, so parsers can be improved and rerun with no network access or extra SEC load.
3. *Why a process pool for PDF parsing?* Parsing is CPU bound and per-document independent. 12 workers cut roughly 3 hours to well under an hour.
4. *Why drop tiny duplicate Item sections?* The table of contents repeats every Item heading with no body, which would create empty sections.
5. *Why page-chunk parse instead of per-page calls?* One call opens the document once. Per-page calls reopened the file for every page.
