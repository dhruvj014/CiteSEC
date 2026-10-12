"""Page-level and document-level retrieval metrics against FinanceBench gold evidence.

Gold for a question is the set of (doc_name, page_idx) from its evidence entries. A retrieved chunk
maps to (doc_name, page) where page is the chunk's page (the start page for naive chunks, None for
EDGAR chunks, which therefore never match gold).
"""
from __future__ import annotations

from collections.abc import Sequence
from statistics import fmean

from citesec.schemas import Question

KS = (1, 3, 5, 10)
Ranked = Sequence[tuple[str, int | None]]  # (doc_name, page) per retrieved chunk, best first


def gold_pages(q: Question) -> set[tuple[str, int]]:
    return {(e.evidence_doc_name, e.evidence_page_num) for e in q.evidence}


def question_metrics(ranked: Ranked, gold: set[tuple[str, int]], ks: Sequence[int] = KS) -> dict:
    """Metrics for one question.

    hit@k: any gold page among the top-k chunks. recall@k: fraction of gold pages covered.
    doc_hit@k: any chunk from a gold document. rr: reciprocal rank of the first gold-page chunk,
    0 if none in the returned list. doc_rr: same at document level.
    """
    gold_docs = {d for d, _ in gold}
    out: dict[str, float] = {}
    for k in ks:
        top = ranked[:k]
        covered = {p for p in top if p in gold}
        out[f"hit@{k}"] = float(bool(covered))
        out[f"recall@{k}"] = len(covered) / len(gold) if gold else 0.0
        out[f"doc_hit@{k}"] = float(any(d in gold_docs for d, _ in top))
    out["rr"] = next((1 / (i + 1) for i, p in enumerate(ranked) if p in gold), 0.0)
    out["doc_rr"] = next((1 / (i + 1) for i, (d, _) in enumerate(ranked) if d in gold_docs), 0.0)
    return out


def aggregate(per_question: list[dict]) -> dict:
    """Mean of every metric over questions. rr and doc_rr are reported as mrr and doc_mrr."""
    if not per_question:
        return {}
    keys = per_question[0].keys()
    agg = {k: fmean(m[k] for m in per_question) for k in keys}
    agg["mrr"] = agg.pop("rr")
    agg["doc_mrr"] = agg.pop("doc_rr")
    agg["n"] = len(per_question)
    return agg


def percentiles(values: list[float]) -> dict:
    """p50 and p95 by nearest rank."""
    if not values:
        return {}
    s = sorted(values)
    return {
        "p50": s[min(len(s) - 1, int(0.50 * len(s)))],
        "p95": s[min(len(s) - 1, int(0.95 * len(s)))],
        "mean": fmean(s),
    }
