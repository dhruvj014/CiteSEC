"""Stage 1: retrieve top-5 chunks for every eval question and cache them (GPU, no LLM).

S1 uses naive chunks with plain dense retrieval. S2, S3 and S4 share the config selected on dev
(configs/retrieval_selected.json). Rerank scores are kept because the abstention gate uses them.
"""
from __future__ import annotations

import json
import logging

from citesec.config import Config
from citesec.eval.answer_data import (
    append_jsonl,
    load_items,
    read_jsonl,
    retrieval_cache_path,
)

log = logging.getLogger("retrieve")
TOP_K = 5


def run(cfg: Config, split: str) -> None:
    from citesec.index.embed import Embedder
    from citesec.retrieval.hybrid import Retriever
    from citesec.retrieval.rerank import Reranker

    sel = json.loads((cfg.root / "configs" / "retrieval_selected.json").read_text())
    items = load_items(cfg, split)
    path = retrieval_cache_path(cfg, split)
    done = {(r["id"], r["system"]) for r in read_jsonl(path)}
    ev, dr = cfg["device_embed"], cfg["device_rerank"]
    embedder = Embedder(cfg["models"]["embed"], device=ev, fp16=ev.startswith("cuda"))
    reranker = Reranker(cfg["models"]["reranker"], device=dr, fp16=dr.startswith("cuda"))
    idx = cfg.processed_dir / "index"
    r1 = Retriever(idx / "naive", embedder, None, cfg["retrieval"]["rrf_k"])
    r2 = Retriever(idx / sel["chunker"], embedder, reranker, cfg["retrieval"]["rrf_k"])
    top_n = cfg["retrieval"]["top_n"]
    r2.search(items[0].question, sel["mode"], sel["rerank"], sel["use_filter"], top_n)  # warm up
    for it in items:
        for system, retr, args in (("S1", r1, ("dense", False, False)),
                                   ("S2", r2, (sel["mode"], sel["rerank"], sel["use_filter"]))):
            if (it.id, system) in done:
                continue
            res = retr.search(it.question, *args, top_n)
            append_jsonl(path, {
                "id": it.id, "system": system,
                "chunk_ids": [c.chunk_id for c in res.chunks[:TOP_K]],
                "scores": [round(s, 6) for s in res.scores[:TOP_K]],
                "timings_ms": {k: round(v, 2) for k, v in res.timings_ms.items()},
            })
        log.info("retrieved %s", it.id)
    log.info("done: %d items cached at %s", len(items), path)
