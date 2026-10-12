"""EDGAR distractor corpus: candidate selection, download with caching, section parsing."""
from __future__ import annotations

import csv
import json
import logging
import re
import time
from pathlib import Path

from citesec.config import Config
from citesec.ingest.html_parser import split_sections
from citesec.schemas import EdgarRecord

log = logging.getLogger(__name__)

MANIFEST_FIELDS = list(EdgarRecord.model_fields)
_SUFFIXES = r"\b(inc|corp|corporation|co|company|plc|ltd|the|group|holdings|de)\b"


def norm_name(s: str) -> str:
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    s = re.sub(_SUFFIXES, " ", s)
    return " ".join(s.split())


def overlaps_financebench(edgar_name: str, fb_companies: set[str]) -> bool:
    """True if a company name matches a FinanceBench company (equality or containment)."""
    e = norm_name(edgar_name)
    if not e:
        return False
    for c in fb_companies:
        n = norm_name(c)
        if n and (n == e or n in e or e in n):
            return True
    return False


def make_doc_name(ticker: str, form: str, period: str) -> str:
    return f"EDGAR_{ticker}_{form.replace('-', '')}_{period}"


def read_manifest(path: Path) -> list[EdgarRecord]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return [EdgarRecord(**r) for r in csv.DictReader(f)]


def write_manifest(records: list[EdgarRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        for r in records:
            w.writerow(r.model_dump())
    tmp.replace(path)


class Throttle:
    """Sleep so calls never exceed max_per_sec."""

    def __init__(self, max_per_sec: float) -> None:
        self.gap = 1.0 / max_per_sec
        self.last = 0.0

    def wait(self) -> None:
        delta = time.monotonic() - self.last
        if delta < self.gap:
            time.sleep(self.gap - delta)
        self.last = time.monotonic()


def html_cache(edgar_dir: Path, doc_name: str) -> Path:
    return edgar_dir / "html" / f"{doc_name}.html"


def sections_cache(processed_dir: Path, doc_name: str) -> Path:
    return processed_dir / "edgar" / f"{doc_name}.jsonl"


def download_company(
    ticker: str, cfg: Config, fb_companies: set[str], throttle: Throttle, n_10q: int = 0
) -> list[EdgarRecord]:
    """List filings for one ticker and cache HTML on disk. Returns [] on FinanceBench overlap."""
    from edgar import Company  # lazy: unit tests do not need the network library

    ecfg = cfg["edgar"]
    throttle.wait()
    company = Company(ticker)
    if overlaps_financebench(company.name, fb_companies):
        log.info("skip %s (%s): already in FinanceBench", ticker, company.name)
        return []
    plan = [("10-K", ecfg["n_10k_per_company"])] + ([("10-Q", n_10q)] if n_10q else [])
    records: list[EdgarRecord] = []
    for form, n in plan:
        throttle.wait()
        filings = company.get_filings(form=form)
        for filing in list(filings.head(n)):
            period = str(filing.period_of_report)
            doc_name = make_doc_name(ticker, form, period)
            target = html_cache(cfg.edgar_dir, doc_name)
            if not target.exists():
                throttle.wait()
                html = filing.html()
                if not html:
                    log.warning("no html for %s", doc_name)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(html, encoding="utf-8")
            records.append(
                EdgarRecord(
                    company=company.name,
                    ticker=ticker,
                    cik=int(company.cik),
                    accession_number=filing.accession_number,
                    form=form,
                    period=period,
                    filing_date=str(filing.filing_date),
                    url=filing.filing_url,
                    doc_name=doc_name,
                )
            )
    return records


def html_to_markdown(html: str) -> str:
    """Convert filing HTML to markdown with tables preserved."""
    from markdownify import markdownify

    return markdownify(html, heading_style="ATX")


def parse_filing(rec: EdgarRecord, cfg: Config) -> list[dict]:
    """Section records for one filing, cached as JSONL."""
    out = sections_cache(cfg.processed_dir, rec.doc_name)
    if out.exists():
        with open(out, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    html = html_cache(cfg.edgar_dir, rec.doc_name).read_text(encoding="utf-8")
    sections = split_sections(html_to_markdown(html))
    rows = [
        {
            "doc_name": rec.doc_name,
            "block_idx": i,
            "section": s.name,
            "title": s.title,
            "text": s.text,
            "n_chars": len(s.text),
        }
        for i, s in enumerate(sections)
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tmp.replace(out)
    return rows
