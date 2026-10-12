# CiteSEC: Citation-Grounded RAG over SEC Filings

Answers natural-language questions over SEC 10-K and 10-Q filings with exact citations (document and page), refuses when evidence is weak, and is evaluated on FinanceBench with retrieval ablations.

Status: Phases 1 (data ingestion) and 2 (chunking and indexing) complete. Results are not yet available, see `BENCHMARKS.md` once generated in Phase 5. Every reported number comes from `results/*.json`.

## Quickstart (Windows, venv only)

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[dev]"
# CUDA build of torch (needed for GPU embeddings), installed separately
.\.venv\Scripts\python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
copy .env.example .env      # then set EDGAR_IDENTITY="Your Name you@example.com"
.\tasks.ps1 data            # download + parse + stats (PDF parsing takes hours on CPU)
.\tasks.ps1 index           # chunk, embed, build FAISS and BM25 indexes (about 45 min on an RTX 4060)
.\tasks.ps1 test
```

## Data

- FinanceBench open sample (150 questions) and its source PDFs, fetched over HTTPS by `scripts/download_data.py`.
- EDGAR distractor filings via `edgartools`, rate limited to 5 requests per second with a declared identity.
- `data/` is gitignored. No filings or PDFs are redistributed.

## Citation

FinanceBench: Islam et al., "FinanceBench: A New Benchmark for Financial Question Answering", 2023. https://arxiv.org/abs/2311.11944

```bibtex
@misc{islam2023financebench,
  title={FinanceBench: A New Benchmark for Financial Question Answering},
  author={Pranab Islam and Anand Kannappan and Douwe Kiela and Rebecca Qian and Nino Scherrer and Bertie Vidgen},
  year={2023},
  eprint={2311.11944},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```
