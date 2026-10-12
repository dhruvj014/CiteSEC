"""Corpus-level input model shared by both chunkers, and loaders that build it from caches."""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from citesec.config import Config
from citesec.ingest import edgar as ed
from citesec.ingest.financebench import load_docs, load_questions, pdf_path, referenced_doc_names
from citesec.ingest.pdf_parser import cache_path, read_pages


@dataclass
class Unit:
    """One citable unit of a document: a PDF page or an EDGAR section block."""

    text: str
    page: int | None = None  # zero-indexed PDF page, None for EDGAR
    section: str | None = None  # EDGAR item name (e.g. "Item 7"), None for PDFs


@dataclass
class Document:
    doc_name: str
    company: str
    doc_type: str
    period: str
    units: list[Unit] = field(default_factory=list)


def iter_documents(cfg: Config) -> Iterator[Document]:
    """FinanceBench gold PDFs first, then EDGAR filings. Order is deterministic."""
    questions = load_questions(cfg.fb_dir)
    docs = load_docs(cfg.fb_dir)
    for name in sorted(referenced_doc_names(questions)):
        cp = cache_path(cfg.processed_dir, name)
        if not cp.exists() or not pdf_path(cfg.fb_dir, name).exists():
            continue
        info = docs[name]
        pages = read_pages(cp)
        yield Document(
            doc_name=name,
            company=info.company,
            doc_type=info.doc_type,
            period=str(info.doc_period or ""),
            units=[Unit(text=p.markdown, page=p.page_idx) for p in pages],
        )
    for rec in sorted(ed.read_manifest(cfg.edgar_dir / "manifest.csv"), key=lambda r: r.doc_name):
        sc = ed.sections_cache(cfg.processed_dir, rec.doc_name)
        if not Path(sc).exists():
            continue
        rows = ed.parse_filing(rec, cfg)
        yield Document(
            doc_name=rec.doc_name,
            company=rec.company,
            doc_type=rec.form,
            period=rec.period,
            units=[Unit(text=r["text"], section=r["section"]) for r in rows],
        )
