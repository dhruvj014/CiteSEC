"""Parse FinanceBench PDFs to page markdown and EDGAR HTML to sections. Cached and resumable.

By default only PDFs referenced by the 150 questions are parsed (the gold corpus).
Use --all-pdfs to also parse the other FinanceBench PDFs as extra distractors.
pymupdf4llm takes minutes per long 10-K, so PDFs are parsed in parallel processes.
"""
from __future__ import annotations

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

from citesec.config import load_config
from citesec.ingest import edgar as ed
from citesec.ingest.financebench import (
    load_docs,
    load_questions,
    missing_pdfs,
    pdf_path,
    referenced_doc_names,
)
from citesec.ingest.pdf_parser import cache_path, get_pages

log = logging.getLogger("parse")


def _parse_one(name: str) -> str:
    cfg = load_config()
    get_pages(pdf_path(cfg.fb_dir, name), name, cfg.processed_dir)
    return name


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-pdfs", action="store_true")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    cfg = load_config()
    questions = load_questions(cfg.fb_dir)
    missing = missing_pdfs(cfg.fb_dir, questions)
    if missing:
        raise SystemExit(f"{len(missing)} referenced PDFs missing: {missing}")
    names = sorted(load_docs(cfg.fb_dir) if args.all_pdfs else referenced_doc_names(questions))
    todo = [n for n in names if not cache_path(cfg.processed_dir, n).exists()
            and pdf_path(cfg.fb_dir, n).exists()]
    log.info("%d docs to parse (%d cached)", len(todo), len(names) - len(todo))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_parse_one, n) for n in todo]
        for f in tqdm(as_completed(futs), total=len(futs), desc="pdf pages"):
            f.result()
    manifest = ed.read_manifest(cfg.edgar_dir / "manifest.csv")
    for rec in tqdm(manifest, desc="edgar sections"):
        if ed.html_cache(cfg.edgar_dir, rec.doc_name).exists():
            ed.parse_filing(rec, cfg)


if __name__ == "__main__":
    main()
