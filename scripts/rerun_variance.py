"""Measure run-to-run generation variance on DEV: regenerate every S2 answer once more.

Ollama output at temperature 0 is stable within a session but can differ across sessions (prompt
cache state), see results/generation_determinism.json. This quantifies the effect on the metric:
verdict agreement and accuracy of run 1 (cached) versus run 2 (fresh), judged by the same judge.
Writes results/generation_variance_dev.json. Dev only.
"""
from __future__ import annotations

import json
import logging

from citesec.config import load_config
from citesec.eval.answer_data import load_items, read_jsonl, retrieval_cache_path
from citesec.eval.judge import build_judge_messages, parse_verdict, rule_refusal
from citesec.eval.numeric import numeric_verdict
from citesec.eval.stage_generate import llm_cache_path, stores
from citesec.eval.stage_judge import judge_client
from citesec.generation.llm import OllamaClient
from citesec.generation.prompts import build_messages

log = logging.getLogger("variance")


def verdict(judge, item, answer: str, memo: dict) -> str:
    key = (item.id, answer)
    if key not in memo:
        num = numeric_verdict(item.gold_answer, answer)
        if rule_refusal(answer):
            memo[key] = "REFUSED"
        elif num is not None:
            memo[key] = num
        else:
            memo[key] = parse_verdict(judge.chat(build_judge_messages(
                item.question, item.gold_answer, answer)).text)[0]
    return memo[key]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    items = load_items(cfg, "dev")
    retrieval = {(r["id"], r["system"]): r for r in read_jsonl(retrieval_cache_path(cfg, "dev"))}
    run1 = {r["id"]: r["raw"] for r in read_jsonl(llm_cache_path(cfg, "dev"))
            if r["system"] == "S2"}
    store = stores(cfg)["S2"]
    gen, judge = OllamaClient.from_config(cfg), judge_client(cfg)
    n_ctx = cfg["generation"]["n_context_blocks"]
    memo: dict = {}
    rows = []
    for k, it in enumerate(items, start=1):
        chunks = store.get_many(retrieval[(it.id, "S2")]["chunk_ids"])[:n_ctx]
        run2 = gen.chat(build_messages(it.question, chunks)).text
        rows.append({"id": it.id, "kind": it.kind, "same_text": run2 == run1[it.id],
                     "v1": verdict(judge, it, run1[it.id], memo),
                     "v2": verdict(judge, it, run2, memo)})
        log.info("%d/%d", k, len(items))
    ans = [r for r in rows if r["kind"] == "answerable"]
    una = [r for r in rows if r["kind"] == "unanswerable"]
    out = {
        "n_items": len(rows),
        "text_identical": round(sum(r["same_text"] for r in rows) / len(rows), 4),
        "verdict_agreement": round(sum(r["v1"] == r["v2"] for r in rows) / len(rows), 4),
        "answerable": {
            "n": len(ans),
            "accuracy_run1": round(sum(r["v1"] == "CORRECT" for r in ans) / len(ans), 4),
            "accuracy_run2": round(sum(r["v2"] == "CORRECT" for r in ans) / len(ans), 4),
            "refusal_rate_run1": round(sum(r["v1"] == "REFUSED" for r in ans) / len(ans), 4),
            "refusal_rate_run2": round(sum(r["v2"] == "REFUSED" for r in ans) / len(ans), 4),
        },
        "unanswerable_abstained_run1": sum(r["v1"] == "REFUSED" for r in una),
        "unanswerable_abstained_run2": sum(r["v2"] == "REFUSED" for r in una),
        "n_unanswerable": len(una),
    }
    (cfg.results_dir / "generation_variance_dev.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
