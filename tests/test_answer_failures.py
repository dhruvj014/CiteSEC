from citesec.eval.answer_failures import final_line, triage

GOLD = {("ACME_2022_10K", 50)}


def row(answer, context_pages, abstained=False, n_cit=2, n_valid=2):
    return {"answer": answer, "abstained": abstained, "n_citations": n_cit, "n_valid": n_valid,
            "context": [{"doc": "ACME_2022_10K", "page": p} for p in context_pages]}


def test_final_line_extracts_after_marker() -> None:
    assert final_line("blah\nFinal answer: $1,577 million") == "$1,577 million"
    assert final_line("no marker here").startswith("no marker")


def test_over_refusal_and_retrieval_miss() -> None:
    r = row("INSUFFICIENT_EVIDENCE", [3, 4], abstained=True)
    assert triage(r, {"verdict": "REFUSED"}, GOLD, set()) == ["over_refusal", "retrieval_miss"]
    r2 = row("INSUFFICIENT_EVIDENCE", [50], abstained=True)
    assert triage(r2, {"verdict": "REFUSED"}, GOLD, set()) == ["over_refusal"]


def test_wrong_answer_tags() -> None:
    nums = {"1577", "6439"}
    miss = triage(row("Final answer: $2,654 million", [3]), {"verdict": "INCORRECT"}, GOLD, nums)
    assert miss == ["retrieval_miss", "number_not_in_context"]
    hall = triage(row("Final answer: $2,654 million", [50]), {"verdict": "INCORRECT"}, GOLD, nums)
    assert hall == ["number_not_in_context"]
    arith = triage(row("Final answer: 6,439 million", [50]), {"verdict": "INCORRECT"}, GOLD, nums)
    assert arith == ["check_arithmetic_or_table"]


def test_invalid_citation_tag_and_years_ignored() -> None:
    r = row("Final answer: in 2022 it was 6,439", [50], n_cit=3, n_valid=1)
    tags = triage(r, {"verdict": "INCORRECT"}, GOLD, {"6439"})
    assert "invalid_citation" in tags and "number_not_in_context" not in tags
