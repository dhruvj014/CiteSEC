from citesec.generation.citations import (
    is_supported,
    numbers,
    parse_citations,
    strip_invalid,
    verify_citations,
)
from citesec.generation.llm import normalize_host
from citesec.generation.prompts import (
    INSUFFICIENT,
    build_closed_book_messages,
    build_messages,
    citation_tag,
)
from citesec.schemas import Chunk


def chunk(doc: str, page: int | None, text: str, section: str | None = None) -> Chunk:
    return Chunk(chunk_id=f"{doc}:{page}:{section}", doc_name=doc, company="Acme", doc_type="10K",
                 period="2022", page=page, section=section, text=text, n_tokens=10,
                 header="Acme | 10K 2022")


PDF = chunk("ACME_2022_10K", 50, "Net sales were $5,577 million in 2022, up 12.5%.")  # PDF p.51
EDGAR = chunk("EDGAR_CVX_10K_2023-12-31", None, "Capital expenditures were $15.8 billion.",
              "Item 7")
CHUNKS = [PDF, EDGAR]


def test_citation_tags_are_one_indexed_for_pdf_and_section_for_edgar() -> None:
    assert citation_tag(PDF) == "[ACME_2022_10K, p.51]"
    assert citation_tag(EDGAR) == "[EDGAR_CVX_10K_2023-12-31, Item 7]"


def test_prompt_contains_blocks_question_and_rules() -> None:
    msgs = build_messages("What were net sales?", CHUNKS)
    assert msgs[0]["role"] == "system" and INSUFFICIENT in msgs[0]["content"]
    user = msgs[1]["content"]
    assert "Block 1 [ACME_2022_10K, p.51]" in user and "Block 2 [EDGAR_CVX" in user
    assert user.rstrip().endswith("Question: What were net sales?")
    assert "Context" not in build_closed_book_messages("q")[1]["content"]


def test_parse_standard_and_variant_formats() -> None:
    t = ("Sales were $5,577 million [ACME_2022_10K, p.51]. Capex $15.8 billion "
         "[EDGAR_CVX_10K_2023-12-31, Item 7] and [ACME_2022_10K, page 12] "
         "and [ACME_2022_10K, 7] and [ACME_2022_10K, p.51, p.52].")
    c = parse_citations(t)
    assert [(x.doc, x.page, x.section) for x in c] == [
        ("ACME_2022_10K", 51, None), ("EDGAR_CVX_10K_2023-12-31", None, "Item 7"),
        ("ACME_2022_10K", 12, None), ("ACME_2022_10K", 7, None),
        ("ACME_2022_10K", 51, None), ("ACME_2022_10K", 52, None)]


def test_parse_malformed_citations() -> None:
    assert parse_citations("see [Note 5, p. 3] and (ACME_2022_10K, p.51) and [1]") == []
    bad = parse_citations("value [ACME_2022_10K, somewhere]")
    assert len(bad) == 1 and bad[0].page is None and bad[0].section is None
    assert parse_citations("") == [] and parse_citations("no citations here") == []
    assert parse_citations("[ACME_2022_10K, p.] [ACME_2022_10K,]")[0].page is None


def test_validity_checks_against_retrieved_blocks_only() -> None:
    t = ("Net sales were $5,577 million [ACME_2022_10K, p.51]. "
         "Other [ACME_2022_10K, p.99] and [OTHER_2020_10K, p.1] and "
         "[EDGAR_CVX_10K_2023-12-31, Item 8]")
    r = verify_citations(t, CHUNKS)
    assert [c.valid for c in r.citations] == [True, False, False, False]
    assert r.n_total == 4 and r.n_valid == 1 and r.n_invalid == 3
    assert "p.99" not in r.clean_text and "[ACME_2022_10K, p.51]" in r.clean_text


def test_case_insensitive_doc_and_edgar_section_match() -> None:
    r = verify_citations("Capex $15.8 billion [edgar_cvx_10k_2023-12-31, item 7].", CHUNKS)
    assert r.citations[0].valid and r.citations[0].supported


def test_support_uses_numbers_then_phrases() -> None:
    assert is_supported("Net sales were $5,577 million", PDF.text)
    assert not is_supported("Net sales were $9,999 million", PDF.text)
    assert not is_supported("In 2022 sales rose", PDF.text.replace("5,577", ""))  # year only
    assert is_supported("capital expenditures increased", EDGAR.text)  # phrase overlap
    assert not is_supported("dividend policy changed materially", EDGAR.text)
    assert numbers("$1,577.00 and (45) and 12.5%") == {"1577", "45", "12.5"}


def test_support_reads_claims_that_follow_the_citation() -> None:
    t = ("From [ACME_2022_10K, p.51], we have the values:\n- Net sales: $5,577 million\n"
         "- Growth 12.5%")
    r = verify_citations(t, CHUNKS)
    assert r.citations[0].valid and r.citations[0].supported is True
    wrong = verify_citations("From [ACME_2022_10K, p.51], net sales were $9,999 million", CHUNKS)
    assert wrong.citations[0].supported is False


def test_claim_after_one_citation_does_not_leak_past_the_next() -> None:
    t = ("Sales $5,577 million [ACME_2022_10K, p.51] and capex $15.8 billion "
         "[EDGAR_CVX_10K_2023-12-31, Item 7]")
    r = verify_citations(t, CHUNKS)
    assert [c.supported for c in r.citations] == [True, True]
    swapped = verify_citations("Sales $15.8 billion [ACME_2022_10K, p.51] and "
                               "capex $5,577 million [EDGAR_CVX_10K_2023-12-31, Item 7]", CHUNKS)
    # the first claim carries its own figure, so it must not borrow the second claim's number
    assert swapped.citations[0].supported is False


def test_support_flags_wrong_number_on_valid_citation() -> None:
    r = verify_citations("Net sales were $1,234 million [ACME_2022_10K, p.51].", CHUNKS)
    assert r.citations[0].valid and r.citations[0].supported is False
    assert r.n_supported == 0


def test_abstention_text_has_no_citations() -> None:
    r = verify_citations(INSUFFICIENT, CHUNKS)
    assert r.abstained and r.n_total == 0


def test_strip_keeps_multi_page_tag_if_any_page_valid() -> None:
    t = "x [ACME_2022_10K, p.51, p.77] y"
    r = verify_citations(t, CHUNKS)
    assert [c.valid for c in r.citations] == [True, False]
    assert "p.51, p.77" in strip_invalid(t, r.citations)
    assert strip_invalid("a [ZZZ_1, p.2] b", parse_citations("a [ZZZ_1, p.2] b")) == "a b"


def test_normalize_host() -> None:
    assert normalize_host("0.0.0.0") == "http://localhost:11434"
    assert normalize_host("127.0.0.1:5000") == "http://127.0.0.1:5000"
    assert normalize_host("http://box:11434/") == "http://box:11434"
