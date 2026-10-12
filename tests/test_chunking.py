import re

from citesec.chunking.blocks import drop_empty_columns, parse_blocks, split_table
from citesec.chunking.common import Document, Unit
from citesec.chunking.naive import chunk_naive
from citesec.chunking.structured import chunk_structured
from citesec.chunking.tokens import RegexTokenizer, count_tokens

TOK = RegexTokenizer()


def make_doc(units: list[Unit]) -> Document:
    return Document("ACME_2022_10K", "Acme Corp", "10K", "2022", units)


def words(n: int, tag: str = "w") -> str:
    return " ".join(f"{tag}{i}" for i in range(n))


def table(n_rows: int) -> str:
    rows = ["| Item | 2022 | 2021 |", "| --- | --- | --- |"]
    rows += [f"| Row{i} | ${i * 10} | ${i * 9} |" for i in range(n_rows)]
    return "\n".join(rows)


def test_parse_blocks_kinds() -> None:
    md = "# Title\n\nSome para\nsecond line\n\n| a | b |\n| - | - |\n| 1 | 2 |\n\nTail"
    kinds = [b.kind for b in parse_blocks(md)]
    assert kinds == ["heading", "para", "table", "para"]


def test_drop_empty_columns_keeps_alignment() -> None:
    lines = ["| A |  | B |", "| --- | --- | --- |", "| 1 |  | 2 |"]
    out = drop_empty_columns(lines)
    assert out[0] == "| A | B |" and out[2] == "| 1 | 2 |"


def test_naive_windows_and_overlap() -> None:
    doc = make_doc([Unit(words(300), page=0), Unit(words(300, "x"), page=1)])
    chunks = chunk_naive(doc, TOK, size=100, overlap=20)
    assert all(c.n_tokens <= 100 and c.text.strip() for c in chunks)
    assert chunks[0].page == 0 and chunks[-1].page == 1
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    # consecutive windows share the configured overlap
    a, b = chunks[0].text.split(), chunks[1].text.split()
    assert a[-20:] == b[:20]


class ShiftyTokenizer(RegexTokenizer):
    """Counts one extra token for any text that starts mid-stream, like WordPiece edge effects."""

    def offsets(self, text: str) -> list[tuple[int, int]]:
        offs = super().offsets(text)
        if text.startswith("w") and len(offs) > 50:
            offs = offs + [(len(text), len(text) + 1)]
        return offs


def test_naive_enforces_cap_when_retokenization_overshoots() -> None:
    doc = make_doc([Unit(words(500), page=0)])
    for c in chunk_naive(doc, ShiftyTokenizer(), size=100, overlap=10):
        assert c.n_tokens <= 100


def test_naive_can_span_pages_but_structured_cannot() -> None:
    doc = make_doc([Unit("alpha " * 40, page=0), Unit("beta " * 40, page=1)])
    naive = chunk_naive(doc, TOK, size=100, overlap=10)
    assert any("alpha" in c.text and "beta" in c.text for c in naive)
    for c in chunk_structured(doc, TOK, target=60, cap=100, overlap=10):
        assert not ("alpha" in c.text and "beta" in c.text)
        assert c.page in (0, 1)
        assert ("alpha" in c.text) == (c.page == 0)


def test_structured_respects_cap_including_header() -> None:
    para = ". ".join(words(15, f"s{i}") for i in range(40)) + "."
    doc = make_doc([Unit(f"# Risk Factors\n\n{para}\n\n{para}", page=3)])
    chunks = chunk_structured(doc, TOK, target=80, cap=120, overlap=20)
    assert len(chunks) > 3
    for c in chunks:
        assert c.n_tokens <= 120
        assert c.header.startswith("Acme Corp | 10K 2022")
        assert "Risk Factors" in c.header
        assert c.page == 3 and c.text.strip()
        assert count_tokens(TOK, c.embed_text) == c.n_tokens


def test_small_table_is_atomic_with_lead_line() -> None:
    md = f"Intro text here.\n\nConsolidated results ($ millions)\n\n{table(5)}\n\nAfter the table."
    chunks = chunk_structured(make_doc([Unit(md, page=0)]), TOK, target=200, cap=300)
    tabs = [c for c in chunks if c.is_table]
    assert len(tabs) == 1
    assert tabs[0].text.startswith("Consolidated results ($ millions)")
    assert all(f"Row{i}" in tabs[0].text for i in range(5))
    assert not any("Row0" in c.text for c in chunks if not c.is_table)


def test_large_table_split_repeats_header() -> None:
    md = table(120)
    chunks = chunk_structured(make_doc([Unit(md, page=7)]), TOK, target=100, cap=150)
    tabs = [c for c in chunks if c.is_table]
    assert len(tabs) > 2
    for c in tabs:
        assert c.text.splitlines()[0] == "| Item | 2022 | 2021 |"
        assert c.text.splitlines()[1].startswith("| ---")
        assert c.n_tokens <= 150 and c.page == 7
    body = [r for c in tabs for r in c.text.splitlines() if r.startswith("| Row")]
    assert body == [f"| Row{i} | ${i * 10} | ${i * 9} |" for i in range(120)]  # no row lost


def test_split_table_giant_row_is_hard_split() -> None:
    lines = ["| a | b |", "| --- | --- |", "| " + words(200) + " | x |"]
    count = lambda s: count_tokens(TOK, s)  # noqa: E731

    def window(t: str, b: int) -> list[str]:
        w = t.split()
        return [" ".join(w[i : i + b]) for i in range(0, len(w), b)]

    pieces = split_table(lines, "", 60, count, window)
    assert len(pieces) > 1 and all(p.startswith("| a | b |") for p in pieces)


def test_edgar_units_keep_section_and_no_page() -> None:
    doc = make_doc([Unit(words(400), section="Item 7"), Unit(words(50, "r"), section="Item 1A")])
    chunks = chunk_structured(doc, TOK, target=120, cap=200)
    assert all(c.page is None for c in chunks)
    assert {c.section for c in chunks} == {"Item 7", "Item 1A"}
    assert all("Item" in c.header for c in chunks)
    for c in chunks:  # sections never mix
        assert ("r0" in c.text) == (c.section == "Item 1A") or "r" not in c.text


def test_markup_noise_removed_and_tiny_chunks_dropped() -> None:
    md = (
        "15\n\n| **Period** | <br>**Total**<br>**Shares** |\n| --- | --- |\n"
        "| Jan 2018 | 714,575 |\n\n" + words(30)
    )
    chunks = chunk_structured(make_doc([Unit(md, page=2)]), TOK, target=100, cap=200)
    tab = next(c for c in chunks if c.is_table)
    assert "<br>" not in tab.text and "**" not in tab.text
    assert tab.text.splitlines()[0] == "| Period | Total Shares |"
    assert all(c.text.strip() != "15" for c in chunks)  # bare page number dropped
    assert any(not c.is_table for c in chunks)


def test_no_empty_chunks() -> None:
    chunks = chunk_structured(make_doc([Unit("\n\n \n", page=0), Unit("---\n\n***", page=1)]), TOK)
    assert chunks == []
    assert not any(re.fullmatch(r"\W*", c.text) for c in chunks)
