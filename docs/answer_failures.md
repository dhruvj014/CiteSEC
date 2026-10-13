# Answer failure analysis (test split, full system S4)

All counts come from `results/answers_metrics_test.json` and `results/answers_S4_test.jsonl`. The triage helper is `python -m citesec.eval.answer_failures --split test --system S4`, it only suggests tags and every case below was checked by reading the retrieved blocks. Judge verdicts come from judge v3 (see `BENCHMARKS.md`, judge validation).

## What the 100 answerable test questions look like (S4)
| Retrieval outcome | n | Correct | Incorrect | Refused |
|---|---|---|---|---|
| Gold page in the 5 context blocks | 49 | 30 | 15 | 4 |
| Gold page missing from context | 51 | 9 | 12 | 30 |

- **Retrieval is the ceiling.** With the gold page in context the system answers 61.2% correctly. Without it, 58.8% of questions end in an honest refusal (30 of 51), 23.5% in a confidently wrong answer (12 of 51) and 17.6% are still answered correctly from other pages (9 of 51).
- Of the 34 false refusals, 30 happen when the gold page is missing, so they are mostly the system declining when it should, not over-caution.
- 27 answers are wrong in total (the 27.0% hallucination rate). 15 of them had the gold page in context, so about half of the wrong answers are model errors and half come from retrieval misses.
- On the 41 unanswerable questions every one was refused. The confidence gate refused 18 before generation and the model's own `INSUFFICIENT_EVIDENCE` reply refused the other 23.
- The model does arithmetic and table reading badly. Numerical reasoning questions score 28.6% and logical reasoning (8 questions) scores 0.0%, against 69.6% on information extraction.

## Ten wrong or refused answers, categorized
**A. Model error with the right evidence in context (5)**
1. `financebench_id_10285`, Boeing FY2018 net PP&E, gold $12,645M, answered $12,361M. Block p.51 holds `Property, plant and equipment, net | 12,645`. The model found it, then subtracted an unrelated line ($284M deferred income taxes) and reported the result. **Invented adjustment.** The number 12,361 appears nowhere in the context, so a number-in-context check would flag it.
2. `financebench_id_04254`, American Water Works FY2021 EBITDA, gold $1,832M, answered $1,899M. Operating income ($1,196M) and D&A ($636M) are both in the context and sum to 1,832. The model used net income ($1,263M) in place of operating income. **Wrong operand for the requested definition.**
3. `financebench_id_04660`, Block FY2016 working capital ratio, gold 1.73, answered 1.74. The operands were right (1,001,425 / 577,464) but the division was wrong (the model wrote 1.7389, the true value is 1.734). **Arithmetic error.** A calculator tool would fix this class.
4. `financebench_id_00460`, Best Buy store count change, gold a decline from 982 to 969. Block p.16 contains both numbers. The model answered "930 ... no change". **Table misread.**
5. `financebench_id_01079`, Amcor acquisitions FY2021 to FY2023. The gold page (p.63) is in context. The answer lists one acquisition (the Czech company) and asserts "no acquisitions mentioned" for the other years. **Incomplete extraction with an unsupported negative claim.**

**B. Retrieval miss, then a confident wrong answer (3)**
6. `financebench_id_01964`, American Express largest liability, gold "customer deposits". The balance sheet (p.97) was not retrieved. The model answered "long-term debt of $29,432 million" from other pages. **The model should have refused.** This is the costly failure type, it sounds right and cites a real page.
7. `financebench_id_01865`, 3M segment that dragged growth excluding M&A, gold "consumer segment, -0.9% organic". The gold page (p.24) was not retrieved. The model chose Safety and Industrial from adjacent pages.
8. `financebench_id_07507`, Adobe operating income change FY2015 to FY2016, gold 65.4%. The income statement page (p.61) was not retrieved, the model produced 85.7%, a figure that is in no block. **Number derived from the wrong inputs.**

**C. Refusal although the evidence was present (1)**
9. `financebench_id_03856`, Adobe FY2017 operating cash flow ratio, gold 0.83. The top reranker score was 0.998 and the gold page (p.60) was in context, yet the model replied `INSUFFICIENT_EVIDENCE` with no reason given. This is an over-refusal, 4 of the 100 questions look like this. The cause was not diagnosed.

**D. Question ambiguity (1)**
10. `financebench_id_01902`, Best Buy "best performing product category by top line" in Q2 FY2024. The gold answer is Entertainment (highest growth, +9%). The model answered Computing and Mobile Phones, which has the highest revenue ($3,674M in block p.12). The model's reading is defensible, so part of this failure is the question, not the system.

## Other observations
- `financebench_id_01319` (AES restructuring costs, gold 0): the model wrote `INSUFFICIENT_EVIDENCE  Final answer: 0`. The pre-set judge rule counts the refusal token anywhere as a refusal, so it scored as refused even though the final answer matches. The rule was fixed before the test run and left alone afterwards, it costs at most one point.
- Judge strictness: `financebench_id_00288` (Best Buy cash decline) was answered "yes, a significant drop" without the 42% magnitude and was marked incorrect.

## What would help most
1. **Retrieval** (see `retrieval_failures.md`): 51% of test questions never get the gold page in context. Per-page deduplication and a soft year filter are the first steps.
2. **A calculator tool** for the arithmetic errors (cases 1, 3), plus a check that every number in a computation appears in a cited block (cases 1, 8).
3. **A stronger generator or a larger context policy**, since cases 2, 4 and 5 are table-reading errors with the evidence present.
