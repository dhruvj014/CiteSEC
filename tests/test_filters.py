import numpy as np

from citesec.retrieval.filters import AliasTable, ChunkMeta, QueryFilter, find_years, parse_query

COMPANIES = {
    "3M": None, "Johnson & Johnson": None, "American Express": None, "Microsoft": None,
    "Block": None, "CHEVRON CORP": "CVX", "Amcor": None,
}
AT = AliasTable(COMPANIES)


def test_find_years() -> None:
    assert find_years("FY2019 vs fiscal 2018, revenue of $12,345 in 2020.") == {2018, 2019, 2020}
    assert find_years("a number like 123456 or 2020123") == set()
    assert find_years("From FY21 to FY22 and fy'19") == {2019, 2021, 2022}
    assert find_years("FY2022 and a 5% FY ratio of 12") == {2022}  # "FY ratio" is not a year


def test_company_detection_names_aliases_tickers() -> None:
    assert AT.find_companies("What was 3M's revenue in FY2018?") == {"3M"}
    assert AT.find_companies("Does J&J pay a dividend?") == {"Johnson & Johnson"}
    assert AT.find_companies("Is JnJ growing?") == {"Johnson & Johnson"}
    both = AT.find_companies("Compare AMEX and Microsoft margins")
    assert both == {"American Express", "Microsoft"}
    assert AT.find_companies("What is CVX capital expenditure?") == {"CHEVRON CORP"}
    assert AT.find_companies("Chevron production volumes") == {"CHEVRON CORP"}


def test_no_false_positive_on_common_words() -> None:
    assert AT.find_companies("How many blocks of shares were purchased?") == set()
    assert AT.find_companies("What is the operating margin?") == set()
    assert AT.find_companies("What is the 3 month average?") == set()  # "3m" needs word boundary


def test_parse_query_adds_next_year() -> None:
    f = parse_query("3M FY2018 gross margin", AT)
    assert f.companies == frozenset({"3M"}) and f.years == frozenset({2018, 2019})
    none = parse_query("what is the gross margin?", AT)
    assert none == QueryFilter() and not none.active


def test_chunk_meta_allowed() -> None:
    meta = ChunkMeta(["3M", "3M", "Amcor", "3M"], ["2018", "2019", "2018", "2023-12-31"])
    assert meta.allowed(QueryFilter()) is None
    assert list(meta.allowed(QueryFilter(companies=frozenset({"3M"})))) == [0, 1, 3]
    f = QueryFilter(companies=frozenset({"3M"}), years=frozenset({2018, 2019}))
    assert list(meta.allowed(f)) == [0, 1]
    assert list(meta.allowed(QueryFilter(years=frozenset({2023})))) == [3]
    assert isinstance(meta.allowed(QueryFilter(companies=frozenset({"Nobody"}))), np.ndarray)
    assert len(meta.allowed(QueryFilter(companies=frozenset({"Nobody"})))) == 0
