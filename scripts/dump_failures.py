"""Print retrieval failures for one config on one split, with diagnostics for manual analysis.

    python scripts/dump_failures.py --split dev --config F --limit 12 > failures.txt
A failure is a question whose first gold page is not in the top 5 chunks.
"""
from __future__ import annotations

import argparse
import json

from citesec.config import load_config
from citesec.eval.splits import load_split, select
from citesec.index.store import ChunkStore
from citesec.ingest.financebench import load_questions
from citesec.ingest.pdf_parser import cache_path, read_pages


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--config", default="F")
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--top", type=int, default=5, help="failure means first gold rank > top")
    args = ap.parse_args()
    cfg = load_config()
    res = json.loads((cfg.results_dir / f"retrieval_{args.split}.json").read_text())
    conf = res["configs"][args.config]
    store = ChunkStore(cfg.processed_dir / "index" / conf["chunker"] / "chunks.sqlite")
    qs = {q.financebench_id: q for q in select(load_questions(cfg.fb_dir),
                                               load_split(cfg), args.split)}
    shown = 0
    for qid, r in conf["per_question"].items():
        rank = r["first_gold_rank"]
        if rank is not None and rank <= args.top:
            continue
        q = qs[qid]
        gold = sorted({(e.evidence_doc_name, e.evidence_page_num) for e in q.evidence})
        top10 = r["top10"]
        gold_docs = {d for d, _ in gold}
        print("=" * 100)
        print(f"{qid} [{q.question_type} / {q.question_reasoning}] first_gold_rank={rank}")
        print("Q:", q.question[:300])
        print("A:", q.answer[:160])
        print("gold:", gold)
        print("filter:", r["filter"])
        print("right doc in top10:", any(d in gold_docs for d, _, _ in top10),
              "| top1 doc:", top10[0][0] if top10 else None)
        for d, p in gold[:2]:
            md = read_pages(cache_path(cfg.processed_dir, d))[p].markdown
            n_tab = sum(1 for ln in md.splitlines() if ln.startswith("|"))
            print(f"  gold page {d} p{p}: {len(md)} chars, {n_tab} table lines | "
                  f"{md.strip()[:140]!r}")
        for d, p, s in top10[:3]:
            print(f"  got {d} p{p} score={s}")
        shown += 1
        if shown >= args.limit:
            break
    print(f"\n{shown} shown")
    store.close()


if __name__ == "__main__":
    main()
