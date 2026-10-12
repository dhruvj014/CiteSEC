"""Baseline chunker: fixed token windows over the concatenated document text.

Deliberately ignores pages, headings and tables. A window can span a page break, so its `page`
is the page where the window starts. No context header is added.
"""
from __future__ import annotations

import bisect

from citesec.chunking.common import Document
from citesec.chunking.tokens import Tokenizer, model_len
from citesec.schemas import Chunk

SEP = "\n\n"


def chunk_naive(doc: Document, tok: Tokenizer, size: int = 512, overlap: int = 64) -> list[Chunk]:
    """`size` is the model input length, so content windows are size minus special tokens."""
    window = size - tok.n_special
    if overlap >= window:
        raise ValueError("overlap must be smaller than the window")
    parts: list[str] = []
    starts: list[int] = []  # char offset where each unit starts in the joined text
    pos = 0
    for u in doc.units:
        starts.append(pos)
        parts.append(u.text)
        pos += len(u.text) + len(SEP)
    full = SEP.join(parts)
    offs = tok.offsets(full)
    chunks: list[Chunk] = []
    step = window - overlap
    for i in range(0, len(offs), step):
        j = min(i + window, len(offs))
        text = full[offs[i][0] : offs[j - 1][1]]
        # re-tokenizing a substring can split edge words differently, so enforce the cap exactly
        while j - 1 > i and model_len(tok, text) > size:
            j -= 1
            text = full[offs[i][0] : offs[j - 1][1]]
        if text.strip():
            u = doc.units[bisect.bisect_right(starts, offs[i][0]) - 1]
            chunks.append(
                Chunk(
                    chunk_id=f"naive:{doc.doc_name}:{len(chunks):05d}",
                    doc_name=doc.doc_name,
                    company=doc.company,
                    doc_type=doc.doc_type,
                    period=doc.period,
                    page=u.page,
                    section=u.section,
                    text=text,
                    is_table=False,
                    n_tokens=model_len(tok, text),
                )
            )
        if i + window >= len(offs):
            break
    return chunks
