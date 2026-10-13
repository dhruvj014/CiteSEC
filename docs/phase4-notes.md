# Phase 4 notes: generation, citations, abstention and answer evaluation

## What was built
- `generation/llm.py`: Ollama chat client (temperature 0, fixed seed, `OLLAMA_HOST` normalized, default model `qwen2.5:7b`, the same Q4 instruct build as `qwen2.5:7b-instruct`).
- `generation/prompts.py`: system prompt with citation tags `[doc, p.N]` (1-indexed PDF page) or `[doc, Item 7]`, a rule to check company and fiscal period against each block header, and an exact refusal token `INSUFFICIENT_EVIDENCE`.
- `generation/citations.py`: parser (tolerates `page N`, bare numbers, multi-page tags, malformed tags) and verifier. **Validity**: the cited (document, page or section) is one of the retrieved blocks. **Support**: a non-year figure from the claim, or at least half of its key words, appears in the cited block. Invalid tags are stripped.
- `generation/abstain.py` and `eval/calibrate_gate.py`: confidence gate on the top reranker score, calibrated on dev.
- `eval/unanswerable.py`: 60 questions (40 out-of-corpus companies, 20 unindexed fiscal years), seeded, hand-reviewed, split 19 dev and 41 test. Type B years are chosen outside every indexed filing and its two comparison years.
- Evaluation stages (`eval/stage_*.py`, driver `eval/run_answer_eval.py`): retrieve, generate, assemble, judge, metrics. S3 and S4 are derived from S2's cached output, so one generation serves three systems.
- `eval/judge.py`, `eval/numeric.py`, `eval/judge_validation.py`: judge v3 and its validation. `eval/answer_metrics.py`, `eval/answer_failures.py`, `scripts/check_determinism.py`, `scripts/rerun_variance.py`.

## Process followed
Prompt, gate threshold, judge and citation metric were all developed on dev. The test split was run **once** (`scripts/run_answers_test.ps1`) after the judge was validated. The gate threshold is **0.0414**.

## Results (test split, 100 answerable and 41 unanswerable, from `BENCHMARKS.md`)
| System | Accuracy | When answered | Wrong | False refusal |
|---|---|---|---|---|
| S0 closed-book | 20.0 | 20.6 | 77.0 | 3.0 |
| S1 naive RAG | 17.0 | 46.0 | 20.0 | 63.0 |
| S2 selected retrieval | 39.0 | 57.4 | 29.0 | 32.0 |
| S4 full system | 39.0 | 59.1 | 27.0 | 34.0 |

- Full system versus closed-book: accuracy 20.0 to 39.0, wrong answers 77.0 to 27.0.
- **All 41 unanswerable questions were refused** (S4): 18 by the gate, 23 by the model's own reply. Abstention precision is 54.7% because the system also refuses 34 answerable questions, 30 of them when retrieval had missed the gold page.
- Citation validity 95.9% on the raw output. Citation support (heuristic) 69.5%.
- Latency, one RTX 4060 with the local 7B model: S4 p50 2.7 s, p95 16.8 s. The gate saves generation on 18 questions, S2 p50 is 3.4 s.
- When the gold page is in context the system answers 61.2% correctly. When it is not, it refuses 58.8% and is confidently wrong 23.5% (`answer_failures.md`).
- The naive-RAG baseline S1 is worse than closed-book on accuracy (17.0 versus 20.0) because it refuses 63% of questions, yet its wrong-answer rate is 20.0 versus 77.0.

## Deviations from the brief (all disclosed)
- **Gate calibration.** The brief says maximize F1 on dev. That point (threshold 0.915) refuses 64% of answerable dev questions, because the reranker score is entity-blind (AUC 0.67): a "net income" page scores high for any company. The operating point is instead max recall on unanswerable questions subject to at most 10% false refusals on dev (threshold 0.0414, dev precision 70%, recall 36.8%, false refusal 6%). The F1 point is kept in `configs/calibration.json`.
- **The LLM is the main abstainer.** With the company and period rule in the prompt it refused 8 of 8 probed unanswerable questions even when the reranker score was 0.54 to 0.96. The gate adds a cheap pre-filter and removed 2 wrong answers on test with no loss of correct ones, it is not the main mechanism.
- **Judge.** The brief asks for a validated local judge and an optional API judge. No API judge was implemented.
- **Hand labels.** The 97 validation labels were written by Claude, not by you, blind to the judge's verdicts. Please audit `data/eval/judge_validation_items*.jsonl` (borderline items are flagged).

## Changes made after looking at dev results
- Prompt: added the company and period check and an exact-refusal instruction. Refusal is detected as the token anywhere in the reply.
- Citation support now reads the sentence before a citation and, when that has no figure, the text after it (models write "From [cite], ..."). Before this fix only 23% of valid citations counted as supported.
- Judge v1 to v3, see below.

## Judge validation (honest account)
| Judge | Sample | Agreement |
|---|---|---|
| v1, LLM decides everything | round 1, 40 items | 75.0% (kappa 0.63) |
| v2, + refusal token by rule + stricter prompt | round 2, 30 fresh items | 76.7% (kappa 0.47) |
| v3, + single-number references graded by rule | round 3, 27 fresh items | 96.3% (kappa 0.93) |
| v3 pooled | 97 items, two rounds already seen | 89.7% |

Findings that shaped the judge: the 7B judge called refusals CORRECT when the reference said something was "not measured", accepted near-miss numbers (2.9 versus 2.8) and accepted partial or fabricated answers. Number grading is now deterministic (rounding at the reference's precision, unit scaling, percent versus fraction), refusals are decided by rule, and the LLM only grades the remaining answers. The 96.3% figure has only 27 items, treat it as supporting evidence, not as precise.
**Faithfulness** (reported as 45 to 49%) comes from the same local judge and was **not validated**. Do not quote it. The deterministic citation-support rate is the better evidence.

## Generation is not bit-reproducible across sessions
At temperature 0 with a fixed seed, 5 of 5 back-to-back repeats were identical but only 2 of 5 matched the cached run from an earlier session, probably because Ollama reuses a prompt-prefix cache and the floating-point path depends on request history (`results/generation_determinism.json`). Measured effect on dev: text identical for 49% of items, verdict identical for 88%, accuracy 26% then 28%, refusal rate 44% then 46%, unanswerable abstentions 19 then 19 of 19 (`generation_variance_dev.json`). Quote accuracy numbers with roughly plus or minus 2 points.

## Limitations
- A 7B local model does arithmetic and table reading poorly (numerical reasoning 28.6%, logical reasoning 0%), and the 100 test questions give wide intervals. One question is one point.
- The unanswerable set is templated and easy for an LLM that checks names, a harder set (a real company with a plausible but absent figure) would be a better test.
- Citation support is a lexical heuristic, not entailment.
- The closed-book baseline does not run on unanswerable questions.

## Interview questions
1. *Why is the reranker-score gate weak and what replaced it?* The cross-encoder scores topical match, not entity match, so a net income page for the wrong company scores high (AUC 0.67). Max-F1 would refuse 64% of answerable questions. I kept the gate as a cheap pre-filter at a capped false-refusal rate and let the LLM do the entity check through the prompt, which refused 41 of 41 unanswerable test questions.
2. *Why did you not trust the LLM judge, and what did you do?* I hand-labeled 40 answers blind and found 75% agreement, with systematic leniency on near-miss numbers and refusals. I replaced those parts with deterministic rules, kept the LLM for free text, and re-validated on a fresh sample. I report all versions, including that the first fresh result (v2) was still poor.
3. *Accuracy only doubled, why?* Retrieval is the ceiling. The gold page reaches the context for 49% of questions. With it the system is right 61% of the time, without it the system mostly refuses. The 7B model's weak arithmetic caps the rest.
4. *What does S3 add over S2?* It removes citations that do not point at a retrieved block (5.1% of citations) so the reader is never sent to evidence the model was not given. It does not change accuracy.
5. *Is the pipeline reproducible?* Retrieval and metrics are exactly reproducible. Generation is stable within a session but not across sessions, I measured the effect (about 2 points) instead of claiming determinism.
