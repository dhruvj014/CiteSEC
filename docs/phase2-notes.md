# Phase 2 notes: chunking and indexing

## What was built
- `chunking/naive.py`: baseline. 512-token model windows (510 content tokens plus 2 special tokens), 64 overlap, over the concatenated document. Ignores pages, headings and tables. A window can span a page break, its `page` is where it starts. No context header.
- `chunking/structured.py` plus `chunking/blocks.py`: the proposed chunker.
  - Never merges across a PDF page or an EDGAR section, so page citation stays exact.
  - Packs paragraphs toward 350 tokens under a hard cap of 512 that includes the context header and special tokens.
  - Tables are atomic: one chunk with a short lead line (caption), or split by rows with the lead line and header rows repeated in every piece. Empty columns are dropped, alignment is preserved.
  - Every chunk has a context header `{company} | {doc_type} {period} | {section}` that is embedded but not displayed.
  - Overlap (up to 64 tokens) only between paragraphs of the same unit.
- `index/embed.py` (bge-base, fp16 on GPU, SQLite cache keyed by sha1 of model plus text, so rebuilds resume), `index/dense.py` (FAISS `IndexFlatIP`, exact cosine), `index/bm25.py` (`bm25s`, number-preserving tokenizer), `index/store.py` (SQLite chunk store).
- `scripts/build_index.py` (parallel chunking, timed stages, writes `results/index_stats.json`), `scripts/validate_phase2.py` (writes `results/phase2_validation.json`).

## Measured results (from `results/`)
| | naive | structured |
|---|---|---|
| documents | 207 | 207 |
| chunks | 75,469 | 128,020 |
| FinanceBench chunks | 22,526 | 46,018 |
| EDGAR chunks | 52,943 | 82,002 |
| table chunks | 0 | 39,374 |
| tokens mean / p95 / max | 511.4 / 512 / 512 | 250.8 / 481 / 509 |
| index size on disk | 527.7 MB | 662.7 MB |
| cold build (chunk / embed / bm25) | 81 s / 946 s / 32 s | 181 s / 1,175 s / 30 s |

Cold-build times come from the first full build, preserved in `results/logs/build_index_first_full_build.log`. The naive index was rebuilt afterwards to fix a cap bug, and that rebuild reused cached embeddings, so the naive row in `index_stats.json` shows 27.7 s of embedding. Hardware: RTX 4060 Laptop, fp16, batch 64.

Verification (`phase2_validation.json`):
- No empty chunks, none over the 512 cap, unique ids, `n_tokens` field equals recomputed length (2,000 sampled per chunker).
- Structured, FinanceBench: 100% of 46,018 chunks have a valid page. 0 chunks span pages.
- Tables, 20 sampled gold pages: 7 kept whole, 14 split with the header repeated on every piece, 0 broken.
- Round trip (300 sampled chunks, chunk's own text as the query): dense rank-1 is 99.67% naive and 100% structured, BM25 rank-1 is 99.33% naive and 99.0% structured. Misses were not investigated, the likely cause is near-duplicate boilerplate chunks.

## Bugs found and fixed during the phase
- **Naive cap overshoot:** the first build had `tokens_max` 515. Re-tokenizing a window's text splits edge words differently than the full-document tokenization. Fixed by shrinking any overshooting window, with a regression test. The naive index was rebuilt.
- **Caption mistakes:** a bare page number ("15") directly before a table was being attached to it as its caption. Lead lines now need at least 6 tokens.
- **Noise tokens:** PDF markdown tables carried `<br>` tags and `**` markers. Stripped before chunking.
- **Tiny chunks:** 6.5% of the first smoke chunks were bare page numbers. Non-table chunks under 6 tokens are dropped.
- **Validator false alarm:** the first page-containment check flagged 17% of chunks. Causes: `<br>` text in the raw page, markdown underscores counted as words, and chunks under 5 words. After fixing the check, 0 chunks span pages. The 745 chunks still under 95% containment are all table pieces whose repeated header sits next to mid-table rows, verified by checking header and body segments separately (745 of 745 explained).

## Differences from the brief and known limitations
- Naive windows use 510 content tokens, because the model adds 2 special tokens to reach 512.
- PDF "section" is the last heading pymupdf4llm detected. It is font-size based and noisy (for example `OVERVI EW`). EDGAR sections are clean Item names.
- Some table header cells from the PDF parser are garbled (`onths Ended Jun`). That is upstream, not chunking. Expect it in the Phase 3 failure analysis.
- EDGAR financial tables split `$` and `)` into their own cells, so they are noisier than PDF tables.
- Only the 84 gold FinanceBench PDFs are indexed, not all 368. Total is 207 documents, above the brief's 100 to 200 target, reported as is.
- Page-level metrics apply to FinanceBench docs only, EDGAR chunks have `page = None`.
- Torch was installed from a locally downloaded wheel because the PyTorch CDN was slow and pip hung. The version is `torch==2.6.0+cu124`.

## Interview questions
1. *Why make tables atomic and repeat the header when splitting?* A split table piece without its header is a column of unlabeled numbers, so both the embedding and the LLM lose what "2,777" means. Repeating the header and caption keeps every piece self-describing. In the sample, 14 of 21 tables needed splitting.
2. *Why embed a context header but not display it?* Many filings contain near-identical text, so "Acme | 10K 2022 | Risk Factors" lets the embedding and BM25 separate companies and years. Displaying it would clutter citations.
3. *Why FAISS `IndexFlatIP` and not an approximate index?* 128k vectors at 768 dims is about 390 MB and a flat search takes milliseconds on CPU, so approximation would only add recall loss to explain.
4. *Why does the BM25 tokenizer keep numbers whole and add a bare form?* Financial questions hinge on exact figures. `$4,200.5` and a table cell `4,200.5` share the token `4200.5`, so formatting differences do not break a match.
5. *How do you know no chunk crosses a page?* By construction (chunking is per page) and by a check. 5-gram containment of every chunk in its own page found 0 spanning chunks, and the 745 remaining misses were explained by the repeated table header.
