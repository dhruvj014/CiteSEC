"""Latency profile of the full pipeline -> results/latency.json.

1. End to end: 100 answerable test-split questions through RagPipeline (the real model, the real
   indexes), per-stage p50 and p95 overall and split by outcome. The questions are used only for
   timing, their answers are not scored here.
2. Optimization A, rerank fewer candidates: rerank the top 50, 30 or 20 fused candidates. Stage
   latency on the same 100 queries and retrieval quality on the dev split, so the speed is never
   bought silently with accuracy.
3. Optimization B, query-embedding cache: embedding latency for the same 100 queries cold and warm.

    python scripts/profile_latency.py [--skip-generation]
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict

import torch

from citesec.config import load_config
from citesec.eval.retrieval_metrics import aggregate, gold_pages, percentiles, question_metrics
from citesec.eval.splits import load_split, select
from citesec.index.embed import CachingEmbedder
from citesec.ingest.financebench import load_questions
from citesec.pipeline import RagPipeline

log = logging.getLogger("profile")
N_QUERIES = 100
WARMUP = 3
STAGES = ("retrieve", "rerank", "generate", "total")


def stage_stats(rows: list[dict[str, float]]) -> dict:
    return {s: {k: round(v, 1) for k, v in percentiles([r[s] for r in rows]).items()}
            for s in STAGES if rows and s in rows[0]}


def end_to_end(pipe: RagPipeline, questions: list[str]) -> dict:
    for q in questions[:WARMUP]:
        pipe.answer(q)
    rows: dict[str, list[dict]] = defaultdict(list)
    for i, q in enumerate(questions, start=1):
        res = pipe.answer(q)
        outcome = "answered" if not res.abstained else f"abstained_{res.abstain_reason}"
        rows[outcome].append(res.latency_ms)
        if i % 10 == 0:
            log.info("pipeline %d/%d", i, len(questions))
    every = [r for rs in rows.values() for r in rs]
    return {"n": len(every), "overall": stage_stats(every),
            "by_outcome": {k: {"n": len(v), **stage_stats(v)} for k, v in sorted(rows.items())}}


def retrieval_pass(pipe: RagPipeline, questions: list[str], rerank_n: int | None) -> list[dict]:
    out = []
    for q in questions:
        r = pipe.retriever.search(q, pipe.mode, pipe.rerank, pipe.use_filter, pipe.top_n,
                                  rerank_n=rerank_n)
        out.append(r.timings_ms)
    return out


def retrieval_quality(pipe: RagPipeline, dev, rerank_n: int | None) -> dict:
    per_q = []
    for q in dev:
        r = pipe.retriever.search(q.question, pipe.mode, pipe.rerank, pipe.use_filter, pipe.top_n,
                                  rerank_n=rerank_n)
        per_q.append(question_metrics([(c.doc_name, c.page) for c in r.chunks], gold_pages(q)))
    m = aggregate(per_q)
    return {k: round(100 * m[k], 1) for k in ("hit@1", "hit@5", "mrr", "doc_hit@5")}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-generation", action="store_true",
                    help="only the retrieval experiments, no LLM calls")
    args = ap.parse_args()
    cfg = load_config()
    qs = load_questions(cfg.fb_dir)
    split = load_split(cfg)
    texts = [q.question for q in select(qs, split, "test")][:N_QUERIES]
    dev = select(qs, split, "dev")
    pipe = RagPipeline.from_config(cfg)
    cache = pipe.retriever.embedder
    assert isinstance(cache, CachingEmbedder)
    for q in texts[:WARMUP]:  # warm the GPU kernels, then forget those vectors
        pipe.retriever.search(q, pipe.mode, pipe.rerank, pipe.use_filter, pipe.top_n)
    out: dict = {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "generator": cfg["models"]["generator"], "n_queries": len(texts),
        "query_set": "answerable test-split questions, used for timing only",
        "settings": {"top_n": pipe.top_n, "rerank_n": pipe.rerank_n or pipe.top_n,
                     "n_context": pipe.n_context, "num_predict": cfg["generation"]["num_predict"]},
    }

    # B: embedding cache, cold then warm on identical queries
    cache._cache.clear()
    cold = retrieval_pass(pipe, texts, None)
    warm = retrieval_pass(pipe, texts, None)
    out["optimization_query_embedding_cache"] = {
        "embed_query_ms_cold": {k: round(v, 1) for k, v in
                                percentiles([r["embed_query"] for r in cold]).items()},
        "embed_query_ms_warm": {k: round(v, 1) for k, v in
                                percentiles([r["embed_query"] for r in warm]).items()},
        "retrieval_total_ms_cold": {k: round(v, 1) for k, v in
                                    percentiles([r["total"] for r in cold]).items()},
        "retrieval_total_ms_warm": {k: round(v, 1) for k, v in
                                    percentiles([r["total"] for r in warm]).items()},
    }
    log.info("embedding cache done")

    # A: rerank fewer candidates, speed on test queries and quality on dev
    exp = {}
    for n in (50, 30, 20):
        timings = retrieval_pass(pipe, texts, n)
        exp[str(n)] = {
            "rerank_ms": {k: round(v, 1) for k, v in  # 0 when the filter left no candidates
                          percentiles([t.get("rerank", 0.0) for t in timings]).items()},
            "retrieval_total_ms": {k: round(v, 1) for k, v in
                                   percentiles([t["total"] for t in timings]).items()},
            "dev_quality_percent": retrieval_quality(pipe, dev, n),
        }
        log.info("rerank_n=%d done %s", n, exp[str(n)]["dev_quality_percent"])
    out["optimization_rerank_candidates"] = {"dev_questions": len(dev), "by_rerank_n": exp}

    if not args.skip_generation:
        out["end_to_end"] = end_to_end(pipe, texts)
    cfg.results_dir.mkdir(exist_ok=True)
    (cfg.results_dir / "latency.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    log.info("wrote results/latency.json")


if __name__ == "__main__":
    main()
