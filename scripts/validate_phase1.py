"""Phase 1 checks: every gold doc parsed, and the evidence page index convention.

For every gold evidence entry, fuzzy-match its text against the parsed markdown of pages
idx-1, idx, idx+1. If offset 0 wins nearly always, page numbers are zero-indexed as assumed.
Writes results/phase1_validation.json.
"""
from __future__ import annotations

import json
import random
import re

from rapidfuzz import fuzz

from citesec.config import load_config
from citesec.ingest.financebench import load_questions, missing_pdfs, referenced_doc_names
from citesec.ingest.pdf_parser import cache_path, read_pages

SNIPPET_CHARS = 400
MATCH_THRESHOLD = 85.0


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def score(snippet: str, page_md: str) -> float:
    a, b = norm(snippet)[:SNIPPET_CHARS], norm(page_md)
    if not a or not b:
        return 0.0
    return fuzz.partial_ratio(a, b)


def main() -> None:
    cfg = load_config()
    qs = load_questions(cfg.fb_dir)
    missing = missing_pdfs(cfg.fb_dir, qs)
    names = referenced_doc_names(qs)
    unparsed = sorted(n for n in names if not cache_path(cfg.processed_dir, n).exists())
    pages = {n: read_pages(cache_path(cfg.processed_dir, n)) for n in names if n not in unparsed}

    rows = []
    for q in qs:
        for e in q.evidence:
            doc = pages.get(e.evidence_doc_name)
            if doc is None:
                continue
            res = {}
            for off in (-1, 0, 1):
                i = e.evidence_page_num + off
                res[off] = score(e.evidence_text, doc[i].markdown) if 0 <= i < len(doc) else 0.0
            best = max(res, key=lambda k: res[k])
            rows.append(
                {
                    "id": q.financebench_id,
                    "doc": e.evidence_doc_name,
                    "page": e.evidence_page_num,
                    "scores": res,
                    "best_offset": best if res[best] >= MATCH_THRESHOLD else None,
                }
            )

    n = len(rows)
    counts = {str(o): sum(1 for r in rows if r["best_offset"] == o) for o in (-1, 0, 1)}
    counts["no_match"] = sum(1 for r in rows if r["best_offset"] is None)
    sample = random.Random(cfg.seed).sample(rows, min(10, n))
    out = {
        "n_questions": len(qs),
        "n_gold_docs": len(names),
        "missing_pdfs": missing,
        "unparsed_docs": unparsed,
        "n_evidence": n,
        "match_threshold": MATCH_THRESHOLD,
        "offset_counts": counts,
        "zero_indexed_confirmed": n > 0 and counts["0"] / n >= 0.9,
        "random_sample_10": sample,
    }
    cfg.results_dir.mkdir(exist_ok=True)
    (cfg.results_dir / "phase1_validation.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "random_sample_10"}, indent=2))
    ok = sum(1 for r in sample if r["best_offset"] == 0)
    print(f"random sample of {len(sample)}: {ok} match at offset 0")


if __name__ == "__main__":
    main()
