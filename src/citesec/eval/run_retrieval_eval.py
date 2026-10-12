"""Retrieval ablation grid on the fixed dev or test split.

    python -m citesec.eval.run_retrieval_eval --split dev --select --verify-repeat
    python -m citesec.eval.run_retrieval_eval --split test      # final numbers, run once

Writes results/retrieval_{split}.json. The retrieval config is chosen on dev only (--select writes
configs/retrieval_selected.json). Re-running on test is refused unless --overwrite-test is given.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from citesec.config import Config, load_config
from citesec.eval.retrieval_metrics import KS, aggregate, gold_pages, percentiles, question_metrics
from citesec.eval.splits import load_split, select
from citesec.ingest.financebench import load_questions
from citesec.retrieval.hybrid import Retriever
from citesec.schemas import Question

log = logging.getLogger("retrieval_eval")


@dataclass(frozen=True)
class RunConfig:
    id: str
    chunker: str
    mode: str
    rerank: bool
    use_filter: bool

    @property
    def description(self) -> str:
        parts = [self.chunker, {"hybrid": "hybrid RRF"}.get(self.mode, self.mode)]
        parts += ["rerank"] if self.rerank else []
        parts += ["filter"] if self.use_filter else []
        return " + ".join(parts)


GRID = [
    RunConfig("A", "naive", "dense", False, False),
    RunConfig("B", "naive", "bm25", False, False),
    RunConfig("C", "structured", "dense", False, False),
    RunConfig("D", "structured", "bm25", False, False),
    RunConfig("E", "structured", "hybrid", False, False),
    RunConfig("F", "structured", "hybrid", True, False),
    RunConfig("G", "structured", "hybrid", True, True),
]
SELECT_FROM = {"C", "D", "E", "F", "G"}  # structured configs are eligible for selection


def evaluate(retriever: Retriever, rc: RunConfig, questions: list[Question],
             top_n: int) -> dict:
    per_q: dict[str, dict] = {}
    timings: dict[str, list[float]] = defaultdict(list)
    for q in questions:
        res = retriever.search(q.question, rc.mode, rc.rerank, rc.use_filter, top_n)
        ranked = [(c.doc_name, c.page) for c in res.chunks]
        gold = gold_pages(q)
        m = question_metrics(ranked, gold)
        first = next((i + 1 for i, p in enumerate(ranked) if p in gold), None)
        per_q[q.financebench_id] = {
            "metrics": m,
            "first_gold_rank": first,
            "filter": (None if res.query_filter is None else
                       {"companies": sorted(res.query_filter.companies or []),
                        "years": sorted(res.query_filter.years or [])}),
            "top10": [[c.doc_name, c.page, round(s, 5)]
                      for c, s in zip(res.chunks[:10], res.scores[:10], strict=True)],
        }
        for stage, ms in res.timings_ms.items():
            timings[stage].append(ms)
    by_type: dict[str, list[dict]] = defaultdict(list)
    for q in questions:
        by_type[q.question_type].append(per_q[q.financebench_id]["metrics"])
    return {
        "description": rc.description,
        "chunker": rc.chunker, "mode": rc.mode, "rerank": rc.rerank, "filter": rc.use_filter,
        "metrics": aggregate([v["metrics"] for v in per_q.values()]),
        "by_question_type": {t: aggregate(ms) for t, ms in sorted(by_type.items())},
        "latency_ms": {s: percentiles(v) for s, v in timings.items()},
        "per_question": per_q,
    }


def select_config(results: dict) -> dict:
    """Best structured config on dev by Hit@5, then MRR, then simplicity (lower id)."""
    cands = [(cid, r) for cid, r in results.items() if cid in SELECT_FROM]
    cid, best = max(cands, key=lambda kv: (kv[1]["metrics"]["hit@5"], kv[1]["metrics"]["mrr"],
                                           -ord(kv[0])))
    return {"config_id": cid, "chunker": best["chunker"], "mode": best["mode"],
            "rerank": best["rerank"], "use_filter": best["filter"], "selected_on": "dev",
            "criterion": "max hit@5, then mrr, then lowest id, among structured configs C-G",
            "dev_hit@5": best["metrics"]["hit@5"], "dev_mrr": best["metrics"]["mrr"]}


def metrics_only(results: dict) -> dict:
    return {cid: {"metrics": r["metrics"],
                  "ranks": {q: v["first_gold_rank"] for q, v in r["per_question"].items()}}
            for cid, r in results.items()}


def build_retrievers(cfg: Config, grid: list[RunConfig]) -> dict[str, Retriever]:
    from citesec.index.embed import Embedder
    from citesec.retrieval.rerank import Reranker

    ev = cfg["device_embed"]
    embedder = Embedder(cfg["models"]["embed"], device=ev, fp16=ev.startswith("cuda"))
    reranker = None
    if any(rc.rerank for rc in grid):
        dr = cfg["device_rerank"]
        reranker = Reranker(cfg["models"]["reranker"], device=dr, fp16=dr.startswith("cuda"))
    out = {}
    for chunker in sorted({rc.chunker for rc in grid}):
        out[chunker] = Retriever(cfg.processed_dir / "index" / chunker, embedder, reranker,
                                 cfg["retrieval"]["rrf_k"])
    return out


def run_grid(cfg: Config, grid: list[RunConfig], questions: list[Question],
             retrievers: dict[str, Retriever]) -> dict:
    top_n = cfg["retrieval"]["top_n"]
    for r in retrievers.values():  # warm up GPU kernels so latency is not dominated by first call
        for _ in range(3):
            r.search(questions[0].question, "hybrid", r.reranker is not None, False, top_n)
    results = {}
    for rc in grid:
        t0 = time.perf_counter()
        results[rc.id] = evaluate(retrievers[rc.chunker], rc, questions, top_n)
        m = results[rc.id]["metrics"]
        log.info("%s %-38s hit@5=%.3f mrr=%.3f doc_hit@5=%.3f (%.0fs)", rc.id, rc.description,
                 m["hit@5"], m["mrr"], m["doc_hit@5"], time.perf_counter() - t0)
    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--configs", default="", help="comma list of config ids, default all")
    ap.add_argument("--select", action="store_true", help="dev only: write retrieval_selected.json")
    ap.add_argument("--verify-repeat", action="store_true", help="run twice, check same metrics")
    ap.add_argument("--overwrite-test", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    out_path = cfg.results_dir / f"retrieval_{args.split}.json"
    if args.split == "test" and out_path.exists() and not args.overwrite_test:
        raise SystemExit(f"{out_path} exists. The test split is run once, pass --overwrite-test "
                         "only if you accept a second test run.")
    if args.select and args.split != "dev":
        raise SystemExit("--select is only allowed on dev")
    grid = [rc for rc in GRID if not args.configs or rc.id in args.configs.split(",")]
    questions = select(load_questions(cfg.fb_dir), load_split(cfg), args.split)
    log.info("split=%s questions=%d configs=%s", args.split, len(questions),
             [rc.id for rc in grid])

    retrievers = build_retrievers(cfg, grid)
    results = run_grid(cfg, grid, questions, retrievers)
    repeat_identical = None
    if args.verify_repeat:
        again = run_grid(cfg, grid, questions, retrievers)
        repeat_identical = metrics_only(results) == metrics_only(again)
        log.info("repeat run identical: %s", repeat_identical)

    import torch

    out = {
        "split": args.split,
        "n_questions": len(questions),
        "seed": cfg.seed,
        "retrieval": cfg["retrieval"],
        "models": {**cfg["models"], "device_embed": cfg["device_embed"],
                   "device_rerank": cfg["device_rerank"]},
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "repeat_run_identical": repeat_identical,
        "ks": list(KS),
        "configs": results,
    }
    cfg.results_dir.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    if args.select:
        choice = select_config(results)
        Path(cfg.root / "configs" / "retrieval_selected.json").write_text(
            json.dumps(choice, indent=2), encoding="utf-8")
        log.info("selected on dev: %s", choice)
    log.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
