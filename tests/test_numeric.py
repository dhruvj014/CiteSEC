import pytest

from citesec.eval.numeric import decimals, extract, final_text, numeric_reference, numeric_verdict


def test_decimals_specified_by_reference() -> None:
    assert decimals("$1577.00") == 0  # trailing zeros are not precision
    assert decimals("30.8%") == 1
    assert decimals("3.46") == 2
    assert decimals("-0.02") == 2
    assert decimals("$400,000,000") == 0


def test_extract_handles_symbols_units_and_negatives() -> None:
    text = "net $1,615.9 million, (45), -0.02, 12.5%, $2.5 billion"
    got = {raw: (v, u) for raw, v, u in extract(text)}
    assert got["$1,615.9"] == (1615.9, 1e6)
    assert got["(45)"][0] == -45.0 and got["-0.02"][0] == -0.02
    assert got["12.5%"] == (12.5, 1.0) and got["$2.5"] == (2.5, 1e9)
    assert extract("FY2022 and Q3") == []  # numbers glued to letters are not figures


@pytest.mark.parametrize("ref,ans,expected", [
    # cases taken from the hand-labeled validation items
    ("$1616.00", "Final answer: Amcor's net AR is $1,615.9 million.", "CORRECT"),
    ("$9068.00", "Final answer: $7,884 million", "INCORRECT"),
    ("30.8%", "Final answer: 30.9%", "INCORRECT"),
    ("30.8%", "Final answer: 30.8%", "CORRECT"),
    ("2.8%", "Rounded to one decimal place: Final answer: 2.9%", "INCORRECT"),
    ("-0.02", "Final answer: -0.02 or -1.53%", "CORRECT"),
    ("$400,000,000 increase.", "Final answer: PepsiCo increased it by $400,000,000.", "CORRECT"),
    ("3.46", "Final answer: approximately 6.01", "INCORRECT"),
    ("AES converted inventory 9.5 times in FY 2022.", "Final answer: roughly 9.64", "INCORRECT"),
    ("AES converted inventory 9.5 times in FY 2022.", "Final answer: 9.5 times", "CORRECT"),
    ("$1577.00", "Final answer: about $1.577 billion", "CORRECT"),  # unit scaling
    ("0.125", "Final answer: 12.5%", "CORRECT"),  # percent versus fraction
    ("1.33", "Final answer: 1.27", "INCORRECT"),
])
def test_numeric_verdicts(ref: str, ans: str, expected: str) -> None:
    assert numeric_verdict(ref, ans) == expected


@pytest.mark.parametrize("ref", [
    "Yes. Commercial Airplanes comprises 39% and Defence comprises 35% of revenue.",  # two figures
    "Data Center",  # no number
    "In 2022, AMD reported higher sales of EPYC server processors and semi-custom products today",
    "2022",  # a lone year is not an answer figure
])
def test_non_numeric_references_go_to_the_llm(ref: str) -> None:
    assert numeric_reference(ref) is None
    assert numeric_verdict(ref, "Final answer: 39%") is None


def test_candidate_without_a_figure_goes_to_the_llm() -> None:
    assert numeric_verdict("$1577.00", "Final answer: the figure is not available") is None
    assert numeric_verdict(None, "Final answer: 5") is None


def test_uses_final_answer_line_not_the_working() -> None:
    ans = "Revenue was $1,577 million last year. Then we compute... Final answer: $2,654 million"
    assert final_text(ans) == "$2,654 million"
    assert numeric_verdict("$1577.00", ans) == "INCORRECT"  # the working's 1,577 must not count


def test_half_up_rounding_at_the_boundary() -> None:
    assert numeric_verdict("2.85", "Final answer: 2.85") == "CORRECT"
    assert numeric_verdict("2.9", "Final answer: 2.85") == "CORRECT"  # 2.85 rounds half up to 2.9
    assert numeric_verdict("2.8", "Final answer: 2.85") == "INCORRECT"
