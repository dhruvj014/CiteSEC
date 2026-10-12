"""Dataset statistics -> results/data_stats.json."""
from __future__ import annotations

import json
from collections import Counter

from citesec.config import load_config
from citesec.ingest import edgar as ed
from citesec.ingest.financebench import load_docs, load_questions, referenced_doc_names
from citesec.ingest.pdf_parser import cache_path, read_pages


def main() -> None:
    cfg = load_config()
    qs = load_questions(cfg.fb_dir)
    docs = load_docs(cfg.fb_dir)
    gold = referenced_doc_names(qs)
    n_pages = n_chars = n_parsed = 0
    for n in gold:
        cp = cache_path(cfg.processed_dir, n)
        if cp.exists():
            recs = read_pages(cp)
            n_parsed += 1
            n_pages += len(recs)
            n_chars += sum(r.n_chars for r in recs)

    manifest = ed.read_manifest(cfg.edgar_dir / "manifest.csv")
    e_blocks = e_chars = 0
    for r in manifest:
        sc = ed.sections_cache(cfg.processed_dir, r.doc_name)
        if sc.exists():
            for line in sc.read_text(encoding="utf-8").splitlines():
                e_blocks += 1
                e_chars += json.loads(line)["n_chars"]

    out = {
        "financebench": {
            "n_questions": len(qs),
            "n_docs_in_metadata": len(docs),
            "n_gold_docs": len(gold),
            "n_gold_docs_parsed": n_parsed,
            "gold_pages": n_pages,
            "gold_chars": n_chars,
            "n_evidence_entries": sum(len(q.evidence) for q in qs),
            "n_companies": len({q.company for q in qs}),
            "gold_docs_by_type": dict(Counter(docs[n].doc_type for n in gold if n in docs)),
            "questions_by_type": dict(Counter(q.question_type for q in qs)),
            "questions_by_reasoning": dict(Counter(q.question_reasoning for q in qs)),
            "questions_by_company": dict(Counter(q.company for q in qs)),
        },
        "edgar": {
            "n_filings": len(manifest),
            "n_companies": len({r.ticker for r in manifest}),
            "by_form": dict(Counter(r.form for r in manifest)),
            "section_blocks": e_blocks,
            "chars": e_chars,
        },
    }
    cfg.results_dir.mkdir(exist_ok=True)
    (cfg.results_dir / "data_stats.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
