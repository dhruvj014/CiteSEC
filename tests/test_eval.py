import pytest

from citesec.eval.retrieval_metrics import aggregate, gold_pages, percentiles, question_metrics
from citesec.eval.splits import make_split
from citesec.schemas import Evidence, Question


def make_q(i: int, qtype: str, pages: list[tuple[str, int]]) -> Question:
    return Question(
        financebench_id=f"id{i:03d}", question="q", answer="a", question_type=qtype,
        company="Acme", doc_name=pages[0][0] if pages else "D",
        evidence=[Evidence(evidence_text="t", evidence_doc_name=d, evidence_page_num=p)
                  for d, p in pages],
    )


def test_gold_pages() -> None:
    q = make_q(1, "t", [("A", 3), ("A", 5), ("B", 0)])
    assert gold_pages(q) == {("A", 3), ("A", 5), ("B", 0)}


def test_question_metrics_known_example() -> None:
    gold = {("A", 3), ("A", 5)}
    ranked = [("B", 1), ("A", 9), ("A", 3), ("C", None), ("A", 5)]
    m = question_metrics(ranked, gold)
    assert m["hit@1"] == 0.0 and m["hit@3"] == 1.0 and m["hit@5"] == 1.0
    assert m["recall@3"] == 0.5 and m["recall@5"] == 1.0
    assert m["rr"] == pytest.approx(1 / 3)  # first gold page at rank 3
    assert m["doc_hit@1"] == 0.0 and m["doc_hit@3"] == 1.0  # doc A first appears at rank 2
    assert m["doc_rr"] == pytest.approx(1 / 2)


def test_question_metrics_miss_and_short_list() -> None:
    m = question_metrics([("B", 1)], {("A", 3)})
    assert m["hit@10"] == 0.0 and m["recall@10"] == 0.0 and m["rr"] == 0.0 and m["doc_rr"] == 0.0


def test_edgar_chunks_never_match() -> None:
    m = question_metrics([("A", None)], {("A", 0)})
    assert m["hit@1"] == 0.0 and m["doc_hit@1"] == 1.0  # right doc, no page


def test_aggregate_and_percentiles() -> None:
    a = question_metrics([("A", 1)], {("A", 1)})
    b = question_metrics([("B", 1)], {("A", 1)})
    agg = aggregate([a, b])
    assert agg["hit@1"] == 0.5 and agg["mrr"] == 0.5 and agg["n"] == 2
    p = percentiles([1.0, 2.0, 3.0, 4.0, 100.0])
    assert p["p50"] == 3.0 and p["p95"] == 100.0


def test_split_is_stratified_seeded_and_disjoint() -> None:
    qs = [make_q(i, ["x", "y", "z"][i % 3], [("D", 0)]) for i in range(150)]
    s1, s2 = make_split(qs, 42), make_split(qs, 42)
    assert s1 == s2  # deterministic
    assert make_split(qs, 7)["dev"] != s1["dev"]
    assert len(s1["dev"]) == 50 and len(s1["test"]) == 100
    assert not set(s1["dev"]) & set(s1["test"])
    assert set(s1["dev"]) | set(s1["test"]) == {q.financebench_id for q in qs}
    assert s1["dev_by_type"] == {"x": 17, "y": 17, "z": 16}
