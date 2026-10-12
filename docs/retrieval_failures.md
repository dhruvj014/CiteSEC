# Retrieval failure analysis (dev split, config G)

Config G = structured chunks, hybrid RRF, cross-encoder rerank, company and year filter. A miss means no gold page in the top 5 chunks. All numbers come from `results/retrieval_dev.json` and `results/retrieval_failure_stats_dev.json` (script: `scripts/analyze_failures.py`). Regenerate the raw dump with `python scripts/dump_failures.py --split dev --config G`.

## The shape of the problem (50 dev questions, config G)
- 30 of 50 questions miss the top 5 (Hit@5 is 40.0%).
- Only 5 of the 30 misses are "wrong document". In 25 the right document is in the top 10 and the **page** is wrong. Choosing the document is largely solved (Doc-Hit@5 is 90.0%), choosing the page is not.
- 22 of the 30 misses have a gold page that is table-heavy (20 or more table lines), almost always a financial statement (balance sheet, income statement, cash flow).
- 16 of the 30 misses have duplicate pages in the top 5, so several of the 5 slots are spent on different chunks of the same page.
- Where the gold page ranks: 15 at rank 6 to 20, 4 at rank 21 to 50, 11 not in the 50 candidates at all. Roughly half the misses are an ordering problem (the page was retrieved but ranked low), the other half are candidate generation problems.

## Ten dev misses and why
1. **financebench_id_03029, 3M FY2018 capital expenditure (gold rank 13).** Gold is the cash flow statement, where the line is "Purchases of property, plant and equipment". The question says "capital expenditure". Vocabulary mismatch between the analyst's term and the filing's line item. The top results are MD&A pages that discuss capital spending in prose.
2. **financebench_id_00499, 3M capital-intensive (not in top 50).** Gold is three statement pages (income statement, balance sheet). The question never names a number or a line item, and the answer needs derived ratios. Statement pages contain no sentence resembling "capital intensive". Semantic gap between a judgment question and numeric evidence.
3. **financebench_id_04735, Adobe FY2015 operating cash flow ratio (rank 9).** Top result is `ADOBE_2016_10K`. The year filter deliberately admits y+1, and the FY2016 10-K also contains FY2015 columns, so a near-identical page from the next year outranks the gold page. A cost of the y+1 rule.
4. **financebench_id_00540, AES FY2022 inventory turnover (rank 21).** Gold is the balance sheet and statement of operations. The reranker preferred notes pages that discuss inventory. The gold table headers are also garbled by the PDF parser (`excet sh`, `are and er share d`), which weakens both embeddings and lexical match.
5. **financebench_id_01928, Amcor adjusted EBITDA FY2023 (rank 16).** Top results are earnings-release pages 6, 3 and 4 with rerank scores around 0.99, all of which mention adjusted EBITDA in prose. Gold is page 11, a reconciliation table whose header is garbled (`Twelve Months En ded`). The reranker is confidently wrong because three pages are plausible, and the table page is poorly extracted.
6. **financebench_id_01930, Amcor real sales change (right document not in top 10).** Gold is the earnings release `AMCOR_2023Q4_EARNINGS` page 9. The filter keeps all Amcor documents for 2022 to 2024, so 10-Ks and 10-Qs compete, and they win. The filter has no notion of document type. The gold table header is also garbled (`Three M`, `onths Ended Jun`).
7. **financebench_id_01198, AMD what drove FY22 revenue change (rank 15).** Gold is page 42 of the MD&A, plain prose with zero table lines. The top results are other pages of the same filing that also discuss net revenue, and the same page appears twice in the top 5 as two chunks. This is a duplicate-page and near-duplicate-content failure, not a vocabulary one.
8. **financebench_id_00723, AMEX operating margin (not in top 50).** The expected answer is that operating margin is not a meaningful metric for this company. The evidence page is the income statement, and nothing in the question points to it. This is a premise-checking question where retrieval has no lexical or semantic handle.
9. **financebench_id_00702, Pfizer 2019 net income events (filter returns nothing).** The question says "in 2019", but the evidence is in the FY2021 10-K, which reports 2019 as a comparison year. The year filter admits only 2019 and 2020 filings, finds no Pfizer document for those years, and returns an empty list. A filter built on the years mentioned in the question fails whenever the answer lives in a later filing.
10. **financebench_id_00822, Foot Locker board nominees (gold at rank 11, right document not in top 10).** The question never names the company ("board member nominees"), so there is no company to filter on and the top result is a Walmart 10-K. This is the honest cost of open retrieval, and why filtered mode is reported separately.

## Failure categories (counts of the 30 G misses)
| Cause | Questions | Fix direction |
|---|---|---|
| Derived-metric or judgment question, evidence is statement tables | about 22 table-heavy gold pages | Statement-aware retrieval, query rewriting to line-item names |
| Garbled table headers from the PDF parser | seen in 4, 5, 6 | Better PDF table extraction, or OCR fallback |
| Several chunks of one page fill the top 5 | 16 | Deduplicate to the best chunk per page before taking top k |
| Filter side effects (y+1, no document-type notion, empty on mismatched year) | 3, 6, 9 | Soft boosts instead of hard filters |
| No company in the question | 10 | Not fixable by the filter, needs conversation context |

## What I would try next (not done, not tuned)
- Collapse the ranked chunk list to one entry per page before taking top k. Duplicate pages are the clearest wasted capacity (16 of 30 misses).
- Replace the hard year filter with a soft score boost so a wrong year cannot empty the results.
- Rewrite analyst phrasing into filing vocabulary ("capital expenditure" into "purchases of property, plant and equipment").

## Changes made after looking at dev failures (disclosed)
- BM25 tokenizer kept `3M` and `10K` whole (it split them into a digit and a letter). Dev Hit@5 for BM25 moved from 16.0% to 14.0%, so this did not help, it was fixed because it was wrong.
- Year parsing gained two-digit fiscal years (`FY22`), and the Johnson & Johnson alias gained `jnj`. Both came from dev misses. No test question was inspected to build aliases or patterns.
