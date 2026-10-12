"""Page-level PDF to markdown using pymupdf4llm. Cached as JSONL per document."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pymupdf
import pymupdf4llm

from citesec.schemas import PageRecord

log = logging.getLogger(__name__)


def parse_pdf(pdf: Path, doc_name: str) -> list[PageRecord]:
    """One PageRecord per page, page_idx is zero-indexed.

    A single page_chunks call opens the file once, which is far faster than one call per page.
    """
    with pymupdf.open(pdf) as doc:
        chunks = pymupdf4llm.to_markdown(doc, page_chunks=True, show_progress=False)
    records: list[PageRecord] = []
    for i, ch in enumerate(chunks):
        md = ch["text"]
        records.append(PageRecord(doc_name=doc_name, page_idx=i, markdown=md, n_chars=len(md)))
    return records


def cache_path(processed_dir: Path, doc_name: str) -> Path:
    return processed_dir / "pages" / f"{doc_name}.jsonl"


def write_pages(records: list[PageRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(r.model_dump_json() + "\n")
    tmp.replace(path)  # atomic, so an interrupted run never leaves a partial cache


def read_pages(path: Path) -> list[PageRecord]:
    with open(path, encoding="utf-8") as f:
        return [PageRecord(**json.loads(line)) for line in f if line.strip()]


def get_pages(pdf: Path, doc_name: str, processed_dir: Path) -> list[PageRecord]:
    """Return cached pages if present, else parse and cache."""
    cp = cache_path(processed_dir, doc_name)
    if cp.exists():
        return read_pages(cp)
    records = parse_pdf(pdf, doc_name)
    write_pages(records, cp)
    return records
