"""Load and join the FinanceBench question and document-metadata files."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from citesec.schemas import DocInfo, Evidence, Question

log = logging.getLogger(__name__)

QUESTIONS_FILE = "data/financebench_open_source.jsonl"
DOCS_FILE = "data/financebench_document_information.jsonl"


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _to_evidence(e: dict) -> Evidence:
    """The released file keys the doc as `doc_name`, the dataset card says `evidence_doc_name`."""
    name = e.get("evidence_doc_name") or e.get("doc_name")
    return Evidence(
        evidence_text=e.get("evidence_text", ""),
        evidence_doc_name=name,
        evidence_page_num=e["evidence_page_num"],
        evidence_text_full_page=e.get("evidence_text_full_page", ""),
    )


def load_docs(fb_dir: Path) -> dict[str, DocInfo]:
    docs: dict[str, DocInfo] = {}
    for r in _read_jsonl(fb_dir / DOCS_FILE):
        # upstream misspells the sector key as "comany_sector_gics"
        sector = r.get("comany_sector_gics") or r.get("company_sector_gics")
        docs[r["doc_name"]] = DocInfo(
            doc_name=r["doc_name"],
            doc_type=r["doc_type"],
            doc_period=r.get("doc_period"),
            doc_link=r.get("doc_link"),
            company=r["company"],
            sector=sector,
        )
    return docs


def load_questions(fb_dir: Path) -> list[Question]:
    docs = load_docs(fb_dir)
    out: list[Question] = []
    for r in _read_jsonl(fb_dir / QUESTIONS_FILE):
        out.append(
            Question(
                financebench_id=r["financebench_id"],
                question=r["question"],
                answer=r["answer"],
                evidence=[_to_evidence(e) for e in r.get("evidence", [])],
                justification=r.get("justification"),
                question_type=r["question_type"],
                question_reasoning=r.get("question_reasoning"),
                company=r["company"],
                doc_name=r["doc_name"],
                doc=docs.get(r["doc_name"]),
            )
        )
    return out


def pdf_path(fb_dir: Path, doc_name: str) -> Path:
    return fb_dir / "pdfs" / f"{doc_name}.pdf"


def referenced_doc_names(questions: list[Question]) -> set[str]:
    names = {q.doc_name for q in questions}
    names |= {e.evidence_doc_name for q in questions for e in q.evidence}
    return names


def missing_pdfs(fb_dir: Path, questions: list[Question]) -> list[str]:
    """doc_names referenced by questions or evidence with no PDF on disk."""
    missing = sorted(n for n in referenced_doc_names(questions) if not pdf_path(fb_dir, n).exists())
    for n in missing:
        log.warning("missing PDF: %s", n)
    return missing
