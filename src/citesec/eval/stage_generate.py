"""Stage 2: run the LLM for S0, S1 and S2 and cache the raw outputs (resumable).

S3 and S4 are derived from S2's output in the assemble stage, because at temperature 0 with a fixed
seed they would generate the identical text. Only S2 items that the gate skips differ, and those
need no LLM call at all in a real deployment.
"""
from __future__ import annotations

import logging
import time

from citesec.config import Config
from citesec.eval.answer_data import (
    append_jsonl,
    load_items,
    read_jsonl,
    retrieval_cache_path,
)
from citesec.generation.llm import OllamaClient
from citesec.generation.prompts import build_closed_book_messages, build_messages
from citesec.index.store import ChunkStore

log = logging.getLogger("generate")
LLM_SYSTEMS = ("S0", "S1", "S2")


def llm_cache_path(cfg: Config, split: str):
    return cfg.processed_dir / "answer_cache" / f"llm_{split}.jsonl"


def stores(cfg: Config) -> dict[str, ChunkStore]:
    idx = cfg.processed_dir / "index"
    return {"S1": ChunkStore(idx / "naive" / "chunks.sqlite"),
            "S2": ChunkStore(idx / "structured" / "chunks.sqlite")}


def run(cfg: Config, split: str, limit: int | None = None) -> None:
    items = load_items(cfg, split)[: limit or None]
    retrieval = {(r["id"], r["system"]): r for r in read_jsonl(retrieval_cache_path(cfg, split))}
    cache_path = llm_cache_path(cfg, split)
    done = {(r["id"], r["system"]) for r in read_jsonl(cache_path)}
    st = stores(cfg)
    llm = OllamaClient.from_config(cfg)
    if not llm.is_up():
        raise SystemExit("Ollama is not reachable. Start it and pull the model first.")
    n_ctx = cfg["generation"]["n_context_blocks"]
    t_start = time.perf_counter()
    for k, it in enumerate(items, start=1):
        for system in LLM_SYSTEMS:
            if (it.id, system) in done or (system == "S0" and it.kind == "unanswerable"):
                continue
            if system == "S0":
                messages = build_closed_book_messages(it.question)
            else:
                rec = retrieval.get((it.id, system))
                if rec is None:
                    raise SystemExit(f"no cached retrieval for {it.id} {system}, run retrieve")
                chunks = st[system].get_many(rec["chunk_ids"])[:n_ctx]
                messages = build_messages(it.question, chunks)
            res = llm.chat(messages)
            append_jsonl(cache_path, {
                "id": it.id, "system": system, "raw": res.text,
                "prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens,
                "gen_ms": round(res.total_ms - res.load_ms, 1),
            })
        elapsed = time.perf_counter() - t_start
        log.info("generated %d/%d %s (%.0fs elapsed)", k, len(items), it.id, elapsed)
