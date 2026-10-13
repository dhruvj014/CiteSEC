"""Stage 3: build the per-system answer files from cached retrieval and LLM output (no models).

S0 closed book. S1 naive dense top-5. S2 selected retrieval. S3 = S2 with citation verification
(invalid citations stripped). S4 = S3 plus the confidence gate, which skips generation when the top
reranker score is below the dev-calibrated threshold.
"""
from __future__ import annotations

import json
import logging

from citesec.config import Config
from citesec.eval.answer_data import (
    EvalItem,
    answers_path,
    load_items,
    read_jsonl,
    retrieval_cache_path,
)
from citesec.eval.stage_generate import llm_cache_path, stores
from citesec.generation.abstain import llm_abstained, load_threshold, should_abstain
from citesec.generation.citations import verify_citations
from citesec.generation.prompts import INSUFFICIENT, citation_tag
from citesec.schemas import Chunk

log = logging.getLogger("assemble")
CAL_PATH = "configs/calibration.json"


def base_record(it: EvalItem, system: str) -> dict:
    return {"id": it.id, "kind": it.kind, "question": it.question, "gold_answer": it.gold_answer,
            "question_type": it.question_type, "question_reasoning": it.question_reasoning,
            "system": system}


def context_info(chunks: list[Chunk], scores: list[float]) -> list[dict]:
    return [{"chunk_id": c.chunk_id, "doc": c.doc_name, "page": c.page, "section": c.section,
             "tag": citation_tag(c), "score": s} for c, s in zip(chunks, scores, strict=False)]


def citation_fields(rep) -> dict:
    return {"citations": [{"doc": c.doc, "page": c.page, "section": c.section, "valid": c.valid,
                           "supported": c.supported, "chunk_id": c.chunk_id}
                          for c in rep.citations],
            "n_citations": rep.n_total, "n_valid": rep.n_valid, "n_supported": rep.n_supported}


def run(cfg: Config, split: str, threshold: float | None = None) -> dict:
    items = load_items(cfg, split)
    retrieval = {(r["id"], r["system"]): r for r in read_jsonl(retrieval_cache_path(cfg, split))}
    llm = {(r["id"], r["system"]): r for r in read_jsonl(llm_cache_path(cfg, split))}
    st = stores(cfg)
    if threshold is None:
        threshold = load_threshold(cfg.root / CAL_PATH)
    out: dict[str, list[dict]] = {s: [] for s in ("S0", "S1", "S2", "S3", "S4")}
    for it in items:
        for system in ("S0", "S1", "S2"):
            gen = llm.get((it.id, system))
            if gen is None:
                continue
            rec = base_record(it, system)
            rec.update({"raw_answer": gen["raw"], "answer": gen["raw"],
                        "tokens": {"prompt": gen["prompt_tokens"],
                                   "completion": gen["completion_tokens"]},
                        "latency_ms": {"generate": gen["gen_ms"]}})
            if system == "S0":
                rec.update(abstained=llm_abstained(gen["raw"]), abstain_reason=None,
                           top_score=None, context=[], n_citations=0)
                latency_total(rec)
                out["S0"].append(rec)
                continue
            r = retrieval[(it.id, system)]
            chunks = st[system].get_many(r["chunk_ids"])
            rep = verify_citations(gen["raw"], chunks)
            llm_abs = llm_abstained(gen["raw"])
            gold = set(it.gold_pages)
            rec.update(context=context_info(chunks, r["scores"]),
                       gold_in_context=(any((c.doc_name, c.page) in gold for c in chunks)
                                        if it.kind == "answerable" else None),
                       top_score=r["scores"][0] if r["scores"] else None,
                       abstained=llm_abs, abstain_reason="llm" if llm_abs else None,
                       **citation_fields(rep))
            rec["latency_ms"].update(retrieval_latency(r["timings_ms"]))
            latency_total(rec)
            out[system].append(rec)
            if system == "S2":
                s3 = {**rec, "system": "S3", "answer": rep.clean_text,
                      "n_invalid_removed": rep.n_invalid,
                      "grounded": rep.n_valid > 0 or llm_abs}
                out["S3"].append(s3)
                gated = should_abstain(r["scores"], threshold)
                s4 = {**s3, "system": "S4"}
                if gated:
                    s4.update(answer=INSUFFICIENT, abstained=True, abstain_reason="gate",
                              n_citations=0, n_valid=0, n_supported=0, citations=[],
                              tokens={"prompt": 0, "completion": 0},
                              n_invalid_removed=0, grounded=True)
                    s4["latency_ms"] = {**{k: v for k, v in s4["latency_ms"].items()
                                           if k != "generate"}, "generate": 0.0}
                    latency_total(s4)
                out["S4"].append(s4)
    for system, rows in out.items():
        if rows:
            p = answers_path(cfg, system, split)
            p.parent.mkdir(exist_ok=True)
            p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                         encoding="utf-8")
            log.info("wrote %s (%d rows)", p.name, len(rows))
    return {s: len(r) for s, r in out.items()}


def retrieval_latency(t: dict) -> dict:
    return {"retrieve": round(sum(v for k, v in t.items() if k not in ("rerank", "total")), 1),
            "rerank": round(t.get("rerank", 0.0), 1)}


def latency_total(rec: dict) -> None:
    lat = rec["latency_ms"]
    lat["total"] = round(sum(v for k, v in lat.items() if k != "total"), 1)
