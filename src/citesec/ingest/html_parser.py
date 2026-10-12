"""Split EDGAR 10-K/10-Q markdown into Item sections."""
from __future__ import annotations

import re
from dataclasses import dataclass

# Matches "Item 1A", "ITEM 7.", optionally with markdown heading/bold markers and a title.
ITEM_RE = re.compile(
    r"^[#*\s]*item\s+(\d{1,2}[ab]?)\s*[.:\-–—]?\s*(.*)$",
    re.IGNORECASE,
)


_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
# "Item 7 of this report", "Item 1A in Part I": prose cross-references, not headings
_XREF_RE = re.compile(r"^(of|in|and|for|to|under|above|below|herein|,)\b", re.IGNORECASE)


def clean_line(line: str) -> str:
    """Normalize a markdown line so table-row headings match: nbsp, links, pipes, bold."""
    line = line.replace("\xa0", " ")
    line = _LINK_RE.sub(r"\1", line)
    line = re.sub(r"[|*_]+", " ", line)
    return " ".join(line.split())


@dataclass
class Section:
    name: str  # e.g. "Item 1A"
    title: str
    text: str


def split_sections(markdown: str) -> list[Section]:
    """Split on Item headings. Text before the first heading becomes 'Preamble'.

    Table-of-contents lines also look like Item headings, so a tiny section is dropped
    when another section with the same name has far more text.
    """
    sections: list[Section] = []
    state = {"name": "Preamble", "title": ""}
    buf: list[str] = []

    def flush() -> None:
        text = "\n".join(buf).strip()
        if text:
            sections.append(Section(state["name"], state["title"], text))

    for line in markdown.splitlines():
        cleaned = clean_line(line)
        m = ITEM_RE.match(cleaned) if len(cleaned) < 200 else None
        if m and not _XREF_RE.match(m.group(2).strip()):
            flush()
            buf.clear()
            state["name"] = f"Item {m.group(1).upper()}"
            state["title"] = m.group(2).strip(" *#|")
        else:
            buf.append(line)
    flush()
    return _drop_toc_duplicates(sections)


def _drop_toc_duplicates(sections: list[Section]) -> list[Section]:
    longest: dict[str, int] = {}
    for s in sections:
        longest[s.name] = max(longest.get(s.name, 0), len(s.text))
    kept: list[Section] = []
    for s in sections:
        if s.name != "Preamble" and len(s.text) < 0.1 * longest[s.name] and len(s.text) < 400:
            continue
        kept.append(s)
    return kept
