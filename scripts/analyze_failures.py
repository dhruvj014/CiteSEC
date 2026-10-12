"""Classify retrieval misses for each config -> results/retrieval_failure_stats_{split}.json.

A miss is a question whose first gold page is not in the top 5 chunks. Categories are descriptive
counts, not tuning inputs.
"""
from __future__ import annotations

import argparse
import json
import re

from citesec.config import load_config
from citesec.eval.splits import load_split, select
from citesec.ingest.financebench import load_questions
from citesec.ingest.pdf_parser import cache_path, read_pages

TABLE_HEAVY_LINES = 20
FY2 = re.compile(r"\bFY\s?'?\d{2}\b", re.IGNORECASE)
YEAR4 = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")


def table_lines(cfg, doc: str, page: int) -> int:
    md = read_pages(cache_path(cfg.processed_dir, doc))[page].markdown
    return sum(1 for ln in md.splitlines() if ln.startswith("|"))


def classify(cfg, q, r) -> dict:
    gold = sorted({(e.evidence_doc_name, e.evidence_page_num) for e in q.evidence})
    gold_docs = {d for d, _ in gold}
    top10 = r["top10"]
    top5_pages = [(d, p) for d, p, _ in top10[:5]]
    rank = r["first_gold_rank"]
    return {
        "first_gold_rank": rank,
        "right_doc_in_top10": any(d in gold_docs for d, _, _ in top10),
        "gold_page_table_heavy": any(table_lines(cfg, d, p) >= TABLE_HEAVY_LINES for d, p in gold),
        "duplicate_pages_in_top5": len(set(top5_pages)) < len(top5_pages),
        "bucket": "not_in_top50" if rank is None else ("6_to_20" if rank <= 20 else "21_to_50"),
        "two_digit_fy_without_year": bool(FY2.search(q.question)) and not YEAR4.search(q.question),
        "filter_years_empty": (r["filter"] is not None and not r["filter"]["years"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    args = ap.parse_args()
    cfg = load_config()
    res = json.loads((cfg.results_dir / f"retrieval_{args.split}.json").read_text())
    qs = {q.financebench_id: q for q in select(load_questions(cfg.fb_dir),
                                               load_split(cfg), args.split)}
    out: dict = {"split": args.split, "definition": "miss = first gold page not in top 5 chunks",
                 "table_heavy_threshold_lines": TABLE_HEAVY_LINES, "configs": {}}
    for cid, conf in res["configs"].items():
        misses = [classify(cfg, qs[qid], r) for qid, r in conf["per_question"].items()
                  if r["first_gold_rank"] is None or r["first_gold_rank"] > 5]
        n = len(misses)
        out["configs"][cid] = {
            "n_questions": len(conf["per_question"]),
            "n_misses": n,
            "wrong_document_not_in_top10": sum(not m["right_doc_in_top10"] for m in misses),
            "right_doc_wrong_page": sum(m["right_doc_in_top10"] for m in misses),
            "gold_page_table_heavy": sum(m["gold_page_table_heavy"] for m in misses),
            "duplicate_pages_in_top5": sum(m["duplicate_pages_in_top5"] for m in misses),
            "rank_6_to_20": sum(m["bucket"] == "6_to_20" for m in misses),
            "rank_21_to_50": sum(m["bucket"] == "21_to_50" for m in misses),
            "not_in_top50": sum(m["bucket"] == "not_in_top50" for m in misses),
            "two_digit_fy_without_year": sum(m["two_digit_fy_without_year"] for m in misses),
        }
    path = cfg.results_dir / f"retrieval_failure_stats_{args.split}.json"
    path.write_text(json.dumps(out, indent=2))
    for cid, s in out["configs"].items():
        print(cid, {k: v for k, v in s.items() if k != "n_questions"})


if __name__ == "__main__":
    main()
