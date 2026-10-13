"""Parse citations out of an answer and verify them against the retrieved context.

validity: the cited (document, page or section) is one of the retrieved blocks.
support: the cited block actually contains a number or key phrase from the claim it is attached to.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from citesec.generation.prompts import INSUFFICIENT, page_label
from citesec.schemas import Chunk

# [DOC_NAME, rest]: the doc id has no spaces, the rest is a page, pages or a section
CITE_RE = re.compile(r"\[\s*([A-Za-z0-9][A-Za-z0-9_.\-]*)\s*,\s*([^\[\]]*?)\s*\]")
PAGE_RE = re.compile(r"(?:pp?\.?|pages?)\s*(\d+)", re.IGNORECASE)
ITEM_RE = re.compile(r"item\s*(\d{1,2}[ab]?)", re.IGNORECASE)
NUM_RE = re.compile(r"\$?\(?\d[\d,]*(?:\.\d+)?\)?%?")
WORD_RE = re.compile(r"[a-z][a-z\-]{4,}")
STOP = frozenset("their which these those would could should about there where while other "
                 "being under between during through".split())
CLAIM_WINDOW = 300


@dataclass
class Citation:
    raw: str
    doc: str
    page: int | None = None  # 1-indexed PDF page as written in the tag
    section: str | None = None  # normalized "Item 7A"
    start: int = 0
    end: int = 0
    valid: bool = False
    supported: bool | None = None  # None when the citation is invalid
    chunk_id: str | None = None


@dataclass
class CitationReport:
    citations: list[Citation] = field(default_factory=list)
    clean_text: str = ""  # answer with invalid citations removed
    abstained: bool = False

    @property
    def n_total(self) -> int:
        return len(self.citations)

    @property
    def n_valid(self) -> int:
        return sum(c.valid for c in self.citations)

    @property
    def n_supported(self) -> int:
        return sum(bool(c.supported) for c in self.citations)

    @property
    def n_invalid(self) -> int:
        return self.n_total - self.n_valid


def parse_citations(text: str) -> list[Citation]:
    """Every [doc, p.N] or [doc, Item 7] tag in the text. One Citation per cited page."""
    out: list[Citation] = []
    for m in CITE_RE.finditer(text):
        doc, rest = m.group(1), m.group(2)
        pages = [int(p) for p in PAGE_RE.findall(rest)]
        if not pages and re.fullmatch(r"\d+", rest.strip()):
            pages = [int(rest.strip())]
        item = ITEM_RE.search(rest)
        if pages:
            out += [Citation(m.group(0), doc, page=p, start=m.start(), end=m.end()) for p in pages]
        elif item:
            out.append(Citation(m.group(0), doc, section=f"Item {item.group(1).upper()}",
                                start=m.start(), end=m.end()))
        elif doc.upper().startswith(("EDGAR_",)) or "_" in doc:
            out.append(Citation(m.group(0), doc, start=m.start(), end=m.end()))  # malformed
    return out


def _norm_num(s: str) -> str:
    s = s.replace("$", "").replace(",", "").replace("(", "").replace(")", "").rstrip("%")
    return re.sub(r"\.0+$", "", s) if "." in s else s


def _is_year(n: str) -> bool:
    return bool(re.fullmatch(r"(19|20)\d{2}", n))


def numbers(text: str) -> set[str]:
    return {_norm_num(n) for n in NUM_RE.findall(text)} - {""}


def claim_segment(text: str, cite: Citation) -> str:
    """The claim a citation backs. Models write both "revenue was X [cite]" and "From [cite],
    revenue was X". If the sentence before the citation holds a figure, that is the claim.
    Otherwise the citation opens a clause and the text after it, up to the next citation, is."""
    before = text[max(0, cite.start - CLAIM_WINDOW) : cite.start]
    before = CITE_RE.sub(" ", before)
    parts = re.split(r"(?<=[.!?])\s+|\n+", before)
    prev = parts[-1] if parts[-1].strip() else (parts[-2] if len(parts) > 1 else "")
    if any(not _is_year(n) for n in numbers(prev)):
        return prev
    after = text[cite.end : cite.end + CLAIM_WINDOW]
    nxt = CITE_RE.search(after)
    after = after[: nxt.start()] if nxt else after
    return f"{prev} {after}".strip()


def is_supported(segment: str, chunk_text: str) -> bool:
    """A claim is supported if a non-year number from it appears in the block, or when the claim
    has no such number, if it has two or more longer words and at least half appear in the block."""
    claim_nums = {n for n in numbers(segment) if not _is_year(n)}
    block_lower = chunk_text.lower()
    if claim_nums:
        return bool(claim_nums & numbers(chunk_text))
    words = {w for w in WORD_RE.findall(segment.lower()) if w not in STOP}
    if len(words) < 2:  # one shared word is not evidence
        return False
    return sum(w in block_lower for w in words) / len(words) >= 0.5


def verify_citations(text: str, chunks: list[Chunk]) -> CitationReport:
    """Mark each citation valid (names a retrieved block) and supported (block backs the claim)."""
    report = CitationReport(abstained=text.strip().startswith(INSUFFICIENT))
    cites = parse_citations(text)
    by_doc: dict[str, list[Chunk]] = {}
    for c in chunks:
        by_doc.setdefault(c.doc_name.lower(), []).append(c)
    for cite in cites:
        for c in by_doc.get(cite.doc.lower(), []):
            if cite.page is not None and c.page is not None and page_label(c.page) == cite.page:
                cite.valid, cite.chunk_id = True, c.chunk_id
            elif cite.section and c.page is None and c.section == cite.section:
                cite.valid, cite.chunk_id = True, c.chunk_id
            if cite.valid:
                seg = claim_segment(text, cite)
                same = [x for x in by_doc[cite.doc.lower()] if _same_target(x, cite)]
                cite.supported = any(is_supported(seg, x.text) for x in same)
                break
    report.citations = cites
    report.clean_text = strip_invalid(text, cites)
    return report


def _same_target(chunk: Chunk, cite: Citation) -> bool:
    if cite.page is not None:
        return chunk.page is not None and page_label(chunk.page) == cite.page
    return chunk.page is None and chunk.section == cite.section


def strip_invalid(text: str, cites: list[Citation]) -> str:
    """Remove invalid citation tags so a reader is never pointed at evidence that was not given."""
    # a multi-page tag has one span, keep it if any of its pages is valid
    spans: dict[tuple[int, int], bool] = {}
    for c in cites:
        spans[(c.start, c.end)] = spans.get((c.start, c.end), False) or c.valid
    out, last = [], 0
    for (start, end), ok in sorted(spans.items()):
        if ok:
            continue
        out.append(text[last:start])
        last = end
    out.append(text[last:])
    return re.sub(r"[ \t]{2,}", " ", "".join(out)).strip()
