"""Regenerate S2 answers for 5 dev questions and compare with the cached outputs.

At temperature 0 with a fixed seed the text must match exactly. Writes
results/generation_determinism.json.
"""
from __future__ import annotations

import json
from difflib import SequenceMatcher

from citesec.config import load_config
from citesec.eval.answer_data import load_items, read_jsonl, retrieval_cache_path
from citesec.eval.stage_generate import llm_cache_path, stores
from citesec.generation.llm import OllamaClient
from citesec.generation.prompts import build_messages

N = 5


def main() -> None:
    cfg = load_config()
    items = [i for i in load_items(cfg, "dev") if i.kind == "answerable"][:N]
    retrieval = {(r["id"], r["system"]): r for r in read_jsonl(retrieval_cache_path(cfg, "dev"))}
    cached = {(r["id"], r["system"]): r["raw"] for r in read_jsonl(llm_cache_path(cfg, "dev"))}
    store = stores(cfg)["S2"]
    llm = OllamaClient.from_config(cfg)
    n_ctx = cfg["generation"]["n_context_blocks"]
    rows = []
    for it in items:
        chunks = store.get_many(retrieval[(it.id, "S2")]["chunk_ids"])[:n_ctx]
        msgs = build_messages(it.question, chunks)
        first, second = llm.chat(msgs).text, llm.chat(msgs).text
        old = cached[(it.id, "S2")]
        diff_at = next((i for i, (a, b) in enumerate(zip(first, old, strict=False)) if a != b),
                       None if first == old else min(len(first), len(old)))
        rows.append({
            "id": it.id,
            "identical_to_cached": first == old,
            "two_fresh_runs_identical": first == second,
            "first_diff_char": diff_at,
            "similarity_to_cached": round(SequenceMatcher(None, first, old).ratio(), 4),
            "chars": len(first),
        })
    out = {"model": cfg["models"]["generator"], "temperature": cfg["generation"]["temperature"],
           "seed": cfg["generation"]["seed"], "n": len(rows),
           "n_identical_to_cached": sum(r["identical_to_cached"] for r in rows),
           "n_fresh_pairs_identical": sum(r["two_fresh_runs_identical"] for r in rows),
           "items": rows}
    (cfg.results_dir / "generation_determinism.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
