"""Markdown block parsing and table helpers used by the structured chunker."""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

HEADING_RE = re.compile(r"^#{1,6}\s+\S")
SEP_CELL_RE = re.compile(r"^:?-{2,}:?$")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass
class Block:
    kind: str  # "heading" | "table" | "para"
    text: str


def clean_markup(line: str) -> str:
    """Drop PDF-markdown noise that only costs tokens: <br> tags and ** bold markers."""
    line = re.sub(r"<br\s*/?>", " ", line).replace("**", "")
    return re.sub(r"[ \t]{2,}", " ", line).rstrip()


def parse_blocks(md: str) -> list[Block]:
    """Split markdown into headings, tables (consecutive pipe lines) and paragraphs."""
    blocks: list[Block] = []
    buf: list[str] = []
    mode = ""

    def flush() -> None:
        nonlocal buf, mode
        if buf:
            blocks.append(Block(mode, "\n".join(buf).strip()))
        buf, mode = [], ""

    for raw in md.replace("\xa0", " ").splitlines():
        line = clean_markup(raw)
        stripped = line.strip()
        if not stripped:
            flush()
        elif HEADING_RE.match(stripped):
            flush()
            blocks.append(Block("heading", stripped))
        elif stripped.startswith("|"):
            if mode != "table":
                flush()
                mode = "table"
            buf.append(stripped)
        else:
            if mode != "para":
                flush()
                mode = "para"
            buf.append(stripped)
    flush()
    return [b for b in blocks if b.text]


def split_cells(row: str) -> list[str]:
    row = row.strip()
    if row.startswith("|"):
        row = row[1:]
    if row.endswith("|"):
        row = row[:-1]
    return [c.strip() for c in row.split("|")]


def is_separator(row: str) -> bool:
    cells = split_cells(row)
    return bool(cells) and all(SEP_CELL_RE.match(c) for c in cells if c) and any(cells)


def drop_empty_columns(lines: list[str]) -> list[str]:
    """Remove columns that are empty in every data row (common in HTML-derived tables).

    Keeps column alignment between the header and the rows intact.
    """
    rows = [split_cells(r) for r in lines]
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    keep = [
        c
        for c in range(width)
        if any(rows[i][c] and not SEP_CELL_RE.match(rows[i][c]) for i in range(len(rows)))
    ]
    if not keep:
        return []
    return ["| " + " | ".join(r[c] for c in keep) + " |" for r in rows]


def table_header(lines: list[str]) -> list[str]:
    """Header row plus separator if present. Repeated on every piece of a split table."""
    if len(lines) >= 2 and is_separator(lines[1]):
        return lines[:2]
    return lines[:1]


def split_by_budget(text: str, budget: int, count: Callable[[str], int],
                    window: Callable[[str, int], list[str]]) -> list[str]:
    """Split text into pieces of at most `budget` tokens, preferring sentence boundaries.

    `window` hard-splits a single over-long sentence by tokens as a last resort.
    """
    if count(text) <= budget:
        return [text]
    pieces: list[str] = []
    cur: list[str] = []
    cur_n = 0
    for sent in (s for s in SENTENCE_RE.split(text) if s.strip()):
        n = count(sent)
        if n > budget:
            if cur:
                pieces.append(" ".join(cur))
                cur, cur_n = [], 0
            pieces.extend(window(sent, budget))
            continue
        if cur_n + n > budget:
            pieces.append(" ".join(cur))
            cur, cur_n = [], 0
        cur.append(sent)
        cur_n += n
    if cur:
        pieces.append(" ".join(cur))
    return pieces


def split_table(lines: list[str], lead: str, budget: int, count: Callable[[str], int],
                window: Callable[[str, int], list[str]]) -> list[str]:
    """Split a table by rows. Every piece repeats the lead line and the header rows."""
    head = table_header(lines)
    prefix = ([lead] if lead else []) + head
    prefix_n = count("\n".join(prefix))
    body_budget = max(budget - prefix_n, 16)
    pieces: list[str] = []
    cur: list[str] = []
    cur_n = 0

    def emit() -> None:
        if cur:
            pieces.append("\n".join(prefix + cur))

    for row in lines[len(head):]:
        n = count(row)
        if n > body_budget:  # a single giant row: hard split its text
            emit()
            cur, cur_n = [], 0
            pieces.extend("\n".join(prefix + [w]) for w in window(row, body_budget))
            continue
        if cur_n + n > body_budget:
            emit()
            cur, cur_n = [], 0
        cur.append(row)
        cur_n += n
    emit()
    return pieces or ["\n".join(prefix)]
