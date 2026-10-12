"""Pydantic models shared across the pipeline."""
from __future__ import annotations

from pydantic import BaseModel


class Evidence(BaseModel):
    evidence_text: str
    evidence_doc_name: str
    evidence_page_num: int  # zero-indexed, validated in Phase 1
    evidence_text_full_page: str = ""


class DocInfo(BaseModel):
    doc_name: str
    doc_type: str
    doc_period: int | str | None = None
    doc_link: str | None = None
    company: str
    sector: str | None = None


class Question(BaseModel):
    financebench_id: str
    question: str
    answer: str
    evidence: list[Evidence]
    justification: str | None = None
    question_type: str
    question_reasoning: str | None = None
    company: str
    doc_name: str
    doc: DocInfo | None = None


class PageRecord(BaseModel):
    doc_name: str
    page_idx: int
    markdown: str
    n_chars: int


class EdgarRecord(BaseModel):
    company: str
    ticker: str
    cik: int
    accession_number: str
    form: str
    period: str
    filing_date: str
    url: str
    doc_name: str  # stable id used for caches and citations


class Chunk(BaseModel):
    chunk_id: str
    doc_name: str
    company: str
    doc_type: str
    period: str
    page: int | None = None  # zero-indexed, None for EDGAR
    section: str | None = None
    text: str
    is_table: bool = False
    n_tokens: int
