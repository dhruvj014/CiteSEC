"""Triage wrong answers into likely failure causes. Tags are hints for manual review, not verdicts.

    python -m citesec.eval.answer_failures --split test --system S4 --limit 30
Categories: retrieval_miss (no gold page in the 5 context blocks), over_refusal, invalid_citation,
number_not_in_context (final answer figures appear in no block: likely hallucinated number),
check_arithmetic_or_table (gold page was in context and the figures exist there: likely arithmetic
or a table misread).
"""
from __future__ import annotations

import argparse
import re

from citesec.config import load_config
from citesec.eval.answer_data import answers_path, judgments_path, read_jsonl
from citesec.eval.answer_metrics import abstained_final
from citesec.generation.citations import numbers
from citesec.ingest.financebench import load_questions

FINAL_RE = re.compile(r"final answer\s*:?\s*(.*)", re.IGNORECASE | re.DOTALL)


def final_line(answer: str) -> str:
    m = FINAL_RE.search(answer)
    return (m.group(1) if m else answer[-300:]).strip()[:300]


def triage(row: dict, judgment: dict, gold_pages: set[tuple[str, int]],
           context_numbers: set[str]) -> list[str]:
    """Tags for one wrong answerable item. `context_numbers` are the figures in the 5 blocks."""
    tags: list[str] = []
    if abstained_final(row, judgment):
        return ["over_refusal"] + (
            [] if _gold_in_context(row, gold_pages) else ["retrieval_miss"])
    if not _gold_in_context(row, gold_pages):
        tags.append("retrieval_miss")
    if row.get("n_citations", 0) > row.get("n_valid", 0):
        tags.append("invalid_citation")
    figs = {n for n in numbers(final_line(row["answer"])) if not re.fullmatch(r"(19|20)\d{2}", n)}
    if figs and not figs & context_numbers:
        tags.append("number_not_in_context")
    elif "retrieval_miss" not in tags:
        tags.append("check_arithmetic_or_table")
    return tags


def _gold_in_context(row: dict, gold_pages: set[tuple[str, int]]) -> bool:
    return any((c["doc"], c["page"]) in gold_pages for c in row.get("context", []))


def main() -> None:
    from citesec.eval.stage_generate import stores

    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--system", default="S4")
    ap.add_argument("--limit", type=int, default=30)
    args = ap.parse_args()
    cfg = load_config()
    rows = read_jsonl(answers_path(cfg, args.system, args.split))
    judg = {j["id"]: j for j in read_jsonl(judgments_path(cfg, args.system, args.split))}
    gold = {q.financebench_id: ({(e.evidence_doc_name, e.evidence_page_num) for e in q.evidence}, q)
            for q in load_questions(cfg.fb_dir)}
    store = stores(cfg)["S1" if args.system == "S1" else "S2"]
    shown = 0
    counts: dict[str, int] = {}
    for r in rows:
        j = judg.get(r["id"])
        if r["kind"] != "answerable" or not j or j["verdict"] == "CORRECT":
            continue
        pages, q = gold[r["id"]]
        ctx_nums: set[str] = set()
        for c in store.get_many([c["chunk_id"] for c in r.get("context", [])]):
            ctx_nums |= numbers(c.text)
        tags = triage(r, j, pages, ctx_nums)
        for t in tags:
            counts[t] = counts.get(t, 0) + 1
        if shown < args.limit:
            shown += 1
            print("=" * 90)
            print(f"{r['id']} [{r['question_type']}] verdict={j['verdict']} tags={tags}")
            print("Q:", r["question"][:260])
            print("GOLD:", q.answer[:200].replace("\n", " "))
            print("GOT:", final_line(r["answer"])[:260].replace("\n", " "))
            print("gold pages:", sorted(pages), "| context:",
                  [(c["doc"], c["page"]) for c in r.get("context", [])][:5])
    print("\ntag counts over all wrong answers:", counts)


if __name__ == "__main__":
    main()
