"""Prompt construction. Citation tags are the single source of truth for how a chunk is cited."""
from __future__ import annotations

from citesec.schemas import Chunk

INSUFFICIENT = "INSUFFICIENT_EVIDENCE"

SYSTEM_PROMPT = f"""You are a careful financial analyst answering questions about SEC filings.

You are given numbered context blocks taken from filings. Each block starts with a citation tag in square brackets, for example [ACME_2022_10K, p.51].

Rules:
1. Use ONLY the context blocks. Never use outside knowledge and never guess a number.
2. Cite every factual claim by copying the citation tag of the block it came from, exactly as written, for example [ACME_2022_10K, p.51]. A claim with no citation is not allowed.
3. For numeric questions, first write the formula, then list each operand with its value, unit, fiscal period and citation, then compute step by step, then give the result with units.
4. Check the company and the fiscal period in the question against the header of each block. Blocks about a different company or a different fiscal period do not answer the question. Never substitute another company's or another year's figure.
5. If the context blocks do not contain what is needed to answer for that company and period, your entire reply must be exactly {INSUFFICIENT}. Do not explain and do not add any other words.
6. Otherwise finish with one line that starts with "Final answer:" and states the answer concisely."""

CLOSED_BOOK_PROMPT = """You are a financial analyst. Answer the question from your own knowledge as \
accurately as you can. Give specific figures where asked. Finish with one line that starts with \
"Final answer:" and states the answer concisely."""


def page_label(page: int) -> int:
    """Citation pages are the 1-indexed PDF page. Chunks store the 0-indexed page."""
    return page + 1


def citation_tag(chunk: Chunk) -> str:
    """[doc, p.N] for PDF chunks, [doc, Item 7] for EDGAR chunks."""
    if chunk.page is not None:
        return f"[{chunk.doc_name}, p.{page_label(chunk.page)}]"
    return f"[{chunk.doc_name}, {chunk.section or 'section'}]"


def format_context(chunks: list[Chunk]) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        blocks.append(f"Block {i} {citation_tag(c)}\n{c.header}\n{c.text}")
    return "\n\n".join(blocks)


def build_messages(question: str, chunks: list[Chunk]) -> list[dict[str, str]]:
    user = f"Context blocks:\n\n{format_context(chunks)}\n\nQuestion: {question}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def build_closed_book_messages(question: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": CLOSED_BOOK_PROMPT},
            {"role": "user", "content": f"Question: {question}"}]
