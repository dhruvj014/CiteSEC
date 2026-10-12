import json
from pathlib import Path

import pymupdf
import pytest

from citesec.config import load_config
from citesec.ingest import edgar as ed
from citesec.ingest.financebench import (
    load_docs,
    load_questions,
    missing_pdfs,
    pdf_path,
    referenced_doc_names,
)
from citesec.ingest.html_parser import split_sections
from citesec.ingest.pdf_parser import get_pages, read_pages


@pytest.fixture
def fb_dir(tmp_path: Path) -> Path:
    d = tmp_path / "fb"
    (d / "data").mkdir(parents=True)
    (d / "pdfs").mkdir()
    q = {
        "financebench_id": "financebench_id_1",
        "company": "Acme",
        "doc_name": "ACME_2022_10K",
        "question_type": "metrics-generated",
        "question_reasoning": "Numerical reasoning",
        "question": "What was revenue?",
        "answer": "$5",
        "justification": None,
        # released files key the doc as `doc_name`, not `evidence_doc_name`
        "evidence": [
            {"doc_name": "ACME_2022_10K", "evidence_page_num": 1, "evidence_text": "Revenue 5",
             "evidence_text_full_page": "Revenue 5"}
        ],
    }
    (d / "data/financebench_open_source.jsonl").write_text(json.dumps(q) + "\n")
    meta = {"doc_name": "ACME_2022_10K", "doc_type": "10K", "doc_period": 2022,
            "doc_link": "http://x", "company": "Acme", "comany_sector_gics": "Industrials"}
    (d / "data/financebench_document_information.jsonl").write_text(json.dumps(meta) + "\n")
    return d


def make_pdf(path: Path, pages: list[str]) -> None:
    doc = pymupdf.open()
    for text in pages:
        doc.new_page().insert_text((72, 72), text)
    doc.save(path)
    doc.close()


def test_load_and_join(fb_dir: Path) -> None:
    qs = load_questions(fb_dir)
    assert len(qs) == 1
    assert qs[0].doc is not None and qs[0].doc.sector == "Industrials"
    assert qs[0].evidence[0].evidence_doc_name == "ACME_2022_10K"
    assert qs[0].evidence[0].evidence_page_num == 1
    assert set(load_docs(fb_dir)) == {"ACME_2022_10K"}
    assert referenced_doc_names(qs) == {"ACME_2022_10K"}


def test_missing_pdfs(fb_dir: Path) -> None:
    qs = load_questions(fb_dir)
    assert missing_pdfs(fb_dir, qs) == ["ACME_2022_10K"]
    make_pdf(pdf_path(fb_dir, "ACME_2022_10K"), ["a", "b"])
    assert missing_pdfs(fb_dir, qs) == []


def test_pdf_pages_zero_indexed_and_cached(tmp_path: Path) -> None:
    pdf = tmp_path / "d.pdf"
    make_pdf(pdf, ["first page alpha", "second page beta", "third page gamma"])
    proc = tmp_path / "proc"
    pages = get_pages(pdf, "D", proc)
    assert [p.page_idx for p in pages] == [0, 1, 2]
    assert "beta" in pages[1].markdown and "alpha" not in pages[1].markdown
    pdf.unlink()  # second call must come from cache
    assert [p.page_idx for p in get_pages(pdf, "D", proc)] == [0, 1, 2]
    assert len(read_pages(proc / "pages" / "D.jsonl")) == 3


def test_split_sections_drops_toc() -> None:
    md = (
        "Cover page text\n"
        "Item 1. Business\n"
        "Item 1A. Risk Factors\n"
        "Item 7. MD&A\n"
        "# Item 1. Business\n" + "We sell widgets. " * 50 + "\n"
        "## ITEM 1A. Risk Factors\n" + "Risks abound. " * 50 + "\n"
        "**Item 7.** Management's Discussion\n" + "Revenue grew. " * 50 + "\n"
    )
    secs = split_sections(md)
    names = [s.name for s in secs]
    assert names == ["Preamble", "Item 1", "Item 1A", "Item 7"]
    assert "widgets" in secs[1].text


def test_split_sections_table_row_headings_and_xrefs() -> None:
    body = "Content sentence. " * 40
    md = (
        "| Item\xa01A. | | | [RISK FACTORS](#abc_1) | | | [29](#abc_2) |\n"
        "| Item\xa07. | | | [MD&A](#abc_3) | | | [60](#abc_4) |\n"
        "| Item\xa01A. | | | RISK FACTORS | |\n" + body + "\n"
        "See Item 7 of this report for details.\n" + body + "\n"
        "|  | | | [Item 7.](#abc_5) | | | [Management Discussion](#abc_6) |\n" + body + "\n"
    )
    secs = split_sections(md)
    assert [s.name for s in secs] == ["Item 1A", "Item 7"]
    assert "See Item 7 of this report" in secs[0].text


def test_norm_and_overlap() -> None:
    assert ed.norm_name("Microsoft Corporation") == "microsoft"
    assert ed.overlaps_financebench("JOHNSON & JOHNSON", {"Johnson & Johnson"})
    assert not ed.overlaps_financebench("Chevron Corp", {"Johnson & Johnson", "Microsoft"})


def test_manifest_roundtrip_and_doc_name(tmp_path: Path) -> None:
    from citesec.schemas import EdgarRecord

    name = ed.make_doc_name("CVX", "10-K", "2023-12-31")
    rec = EdgarRecord(company="Chevron", ticker="CVX", cik=93410, accession_number="0001-23-4",
                      form="10-K", period="2023-12-31", filing_date="2024-02-20",
                      url="https://sec.gov/x", doc_name=name)
    p = tmp_path / "m.csv"
    ed.write_manifest([rec], p)
    assert ed.read_manifest(p) == [rec]
    assert rec.doc_name == "EDGAR_CVX_10K_2023-12-31"
    assert ed.read_manifest(tmp_path / "none.csv") == []


def test_throttle_enforces_gap() -> None:
    import time

    t = ed.Throttle(50)  # 20 ms gap
    start = time.monotonic()
    for _ in range(5):
        t.wait()
    assert time.monotonic() - start >= 4 * 0.02 * 0.9


def test_config_loads() -> None:
    cfg = load_config()
    assert cfg.seed == 42
    assert cfg["edgar"]["rate_limit_per_sec"] <= 5
    assert not set(cfg["edgar"]["holdout_tickers"]) & set(cfg["edgar"]["candidates"])
