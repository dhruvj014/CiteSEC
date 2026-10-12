"""Structure-aware chunker.

Rules:
- Never merges across units (a PDF page or an EDGAR section), so page citations stay exact.
- Packs paragraphs toward a target size under a hard cap that includes the context header.
- Tables are atomic: one chunk with a short lead line, or split by rows with the header repeated.
- Every chunk carries a context header, embedded with the text but not displayed.
"""
from __future__ import annotations

import re

from citesec.chunking.blocks import (
    Block,
    drop_empty_columns,
    parse_blocks,
    split_by_budget,
    split_table,
)
from citesec.chunking.common import Document, Unit
from citesec.chunking.tokens import Tokenizer, count_tokens, model_len
from citesec.schemas import Chunk

SAFETY = 6  # tokens kept free for boundary effects when re-tokenizing joined text
LEAD_MAX_TOKENS = 40  # a short paragraph right before a table is treated as its caption
MIN_CONTENT_TOKENS = 6  # non-table chunks shorter than this are dropped as noise
MIN_HEADING_FLUSH = 100  # a heading only starts a new chunk if the buffer has this many tokens
_EMPH = re.compile(r"[*_#`]+")


def make_header(doc: Document, section: str | None) -> str:
    head = f"{doc.company} | {doc.doc_type} {doc.period}".rstrip()
    return f"{head} | {section}" if section else head


def heading_label(text: str) -> str:
    return " ".join(_EMPH.sub(" ", text).split())[:80]


class _Chunker:
    def __init__(self, doc: Document, tok: Tokenizer, target: int, cap: int, overlap: int) -> None:
        self.doc, self.tok, self.target, self.cap, self.overlap = doc, tok, target, cap, overlap
        self.out: list[Chunk] = []
        self.heading: str | None = None  # last PDF heading, carried across pages
        self._hdr: dict[str, int] = {}

    def n(self, text: str) -> int:
        return count_tokens(self.tok, text)

    def window(self, text: str, budget: int) -> list[str]:
        """Hard split by tokens, the last resort for text with no usable boundaries."""
        offs = self.tok.offsets(text)
        return [
            text[offs[i][0] : offs[min(i + budget, len(offs)) - 1][1]]
            for i in range(0, len(offs), budget)
        ]

    def hdr_tokens(self, section: str | None) -> int:
        header = make_header(self.doc, section) + "\n"
        if header not in self._hdr:
            self._hdr[header] = self.n(header)
        return self._hdr[header]

    def budget(self, section: str | None) -> int:
        """Content tokens for a packed chunk: the target, bounded by the cap."""
        return min(self.target, self.cap_budget(section))

    def cap_budget(self, section: str | None) -> int:
        """Content tokens that fit under the hard cap after header and special tokens."""
        return self.cap - self.tok.n_special - self.hdr_tokens(section) - SAFETY

    def emit(self, text: str, unit: Unit, section: str | None, is_table: bool) -> None:
        text = text.strip()
        if not re.search(r"\w", text):
            return
        if not is_table and self.n(text) < MIN_CONTENT_TOKENS:  # page numbers, stray labels
            return
        header = make_header(self.doc, section)
        pieces = [text]
        if model_len(self.tok, f"{header}\n{text}") > self.cap:  # exact check, rare fallback
            pieces = self.window(text, max(self.cap_budget(section) - SAFETY, 16))
        for piece in pieces:
            self.out.append(
                Chunk(
                    chunk_id=f"structured:{self.doc.doc_name}:{len(self.out):05d}",
                    doc_name=self.doc.doc_name,
                    company=self.doc.company,
                    doc_type=self.doc.doc_type,
                    period=self.doc.period,
                    page=unit.page,
                    section=section,
                    text=piece,
                    is_table=is_table,
                    n_tokens=model_len(self.tok, f"{header}\n{piece}"),
                    header=header,
                )
            )

    def run_unit(self, unit: Unit) -> None:
        section_of = (lambda: unit.section) if unit.section else (lambda: self.heading)
        buf: list[Block] = []
        buf_n = 0
        buf_section: str | None = None

        def flush(carry: bool = False) -> None:
            nonlocal buf, buf_n, buf_section
            if buf:
                self.emit("\n\n".join(b.text for b in buf), unit, buf_section, False)
            last = buf[-1] if buf else None
            buf, buf_n, buf_section = [], 0, None
            if carry and last and last.kind == "para" and self.n(last.text) <= self.overlap:
                buf, buf_n, buf_section = [last], self.n(last.text), section_of()

        for blk in parse_blocks(unit.text):
            if blk.kind == "heading":
                if not unit.section:
                    self.heading = heading_label(blk.text)
                if buf_n >= MIN_HEADING_FLUSH:
                    flush()
            if blk.kind == "table":
                lead = ""
                if (buf and buf[-1].kind == "para"
                        and MIN_CONTENT_TOKENS <= self.n(buf[-1].text) <= LEAD_MAX_TOKENS):
                    lead = buf.pop().text
                    buf_n = sum(self.n(b.text) for b in buf)
                flush()
                self.run_table(blk, lead, unit, section_of())
                continue
            sec = section_of()
            for piece in self.fit(blk, sec):
                pn = self.n(piece.text)
                if buf and buf_n + pn > self.budget(buf_section):
                    flush(carry=True)
                if not buf:
                    buf_section = sec
                buf.append(piece)
                buf_n += pn
        flush()

    def fit(self, blk: Block, section: str | None) -> list[Block]:
        """Split a block that alone exceeds the budget (long paragraphs, EDGAR walls of text)."""
        b = self.budget(section)
        if self.n(blk.text) <= b:
            return [blk]
        parts = split_by_budget(blk.text, b, self.n, self.window)
        return [Block(blk.kind, p) for p in parts]

    def run_table(self, blk: Block, lead: str, unit: Unit, section: str | None) -> None:
        lines = drop_empty_columns(blk.text.splitlines())
        if not lines:
            return
        budget = self.cap_budget(section)
        whole = "\n".join(([lead] if lead else []) + lines)
        if self.n(whole) <= budget:
            self.emit(whole, unit, section, True)
            return
        for piece in split_table(lines, lead, budget, self.n, self.window):
            self.emit(piece, unit, section, True)


def chunk_structured(doc: Document, tok: Tokenizer, target: int = 350, cap: int = 512,
                     overlap: int = 64) -> list[Chunk]:
    c = _Chunker(doc, tok, target, cap, overlap)
    for unit in doc.units:
        c.run_unit(unit)
    return c.out
