"""Stage 4: judge every system answer (verdict) and the faithfulness of RAG answers. Resumable.

Identical (question, answer) pairs, which are common across S2, S3 and S4, are judged once.
"""
from __future__ import annotations

import hashlib
import logging

from citesec.config import Config
from citesec.eval.answer_data import answers_path, append_jsonl, judgments_path, read_jsonl
from citesec.eval.judge import (
    build_faith_messages,
    build_judge_messages,
    parse_faith,
    parse_verdict,
    rule_refusal,
)
from citesec.eval.numeric import numeric_verdict
from citesec.eval.stage_generate import stores
from citesec.generation.llm import OllamaClient

log = logging.getLogger("judge")
SYSTEMS = ("S0", "S1", "S2", "S3", "S4")
JUDGE_NUM_PREDICT = 8


def pair_key(item_id: str, answer: str) -> str:
    return hashlib.sha1(f"{item_id}\n{answer}".encode()).hexdigest()


def judge_client(cfg: Config) -> OllamaClient:
    g = cfg["generation"]
    return OllamaClient(cfg["models"].get("judge", cfg["models"]["generator"]),
                        temperature=0.0, seed=g["seed"], num_ctx=g["num_ctx"],
                        num_predict=JUDGE_NUM_PREDICT, keep_alive=g["keep_alive"])


def evidence_text(row: dict, store) -> str:
    """The cited blocks if any citation is valid, else all context blocks given to the model."""
    valid_ids = {c["chunk_id"] for c in row.get("citations", []) if c.get("valid")}
    ctx = [c["chunk_id"] for c in row.get("context", [])]
    ids = [i for i in ctx if i in valid_ids] or ctx
    chunks = store.get_many(ids)
    return "\n\n".join(f"{c.header}\n{c.text}" for c in chunks)


def run(cfg: Config, split: str, limit: int | None = None) -> None:
    llm = judge_client(cfg)
    if not llm.is_up():
        raise SystemExit("Ollama is not reachable")
    st = stores(cfg)
    memo: dict[str, dict] = {}
    for system in SYSTEMS:
        rows = read_jsonl(answers_path(cfg, system, split))
        if not rows:
            continue
        out_path = judgments_path(cfg, system, split)
        done = {r["id"] for r in read_jsonl(out_path)}
        for r in rows[: limit or None]:
            if r["id"] in done:
                continue
            key = pair_key(r["id"], r["answer"])
            if key not in memo:
                numeric = numeric_verdict(r["gold_answer"], r["answer"])
                if rule_refusal(r["answer"]):
                    verdict, perr = "REFUSED", False
                elif numeric is not None:
                    verdict, perr = numeric, False
                else:
                    verdict, perr = parse_verdict(llm.chat(build_judge_messages(
                        r["question"], r["gold_answer"], r["answer"])).text)
                faithful = None
                f_err = False
                needs = (system != "S0" and r["kind"] == "answerable" and verdict != "REFUSED"
                         and not r.get("abstained"))
                if needs:
                    store = st["S1" if system == "S1" else "S2"]
                    faithful, f_err = parse_faith(llm.chat(build_faith_messages(
                        r["answer"], evidence_text(r, store))).text)
                memo[key] = {"verdict": verdict, "parse_error": perr, "faithful": faithful,
                             "faith_parse_error": f_err}
            append_jsonl(out_path, {"id": r["id"], **memo[key]})
        log.info("judged %s %s", system, split)
