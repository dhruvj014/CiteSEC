import pytest
from tests.fakes import FakeReranker

from citesec.retrieval.hybrid import Retriever, rrf_fuse


def test_rrf_known_example() -> None:
    a = [("x", 9.0), ("y", 8.0), ("z", 7.0)]
    b = [("y", 5.0), ("w", 4.0), ("x", 3.0)]
    fused = dict(rrf_fuse([a, b], k=60))
    assert fused["x"] == pytest.approx(1 / 61 + 1 / 63)
    assert fused["y"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["z"] == pytest.approx(1 / 63)
    order = [c for c, _ in rrf_fuse([a, b], k=60)]
    assert fused["w"] == pytest.approx(1 / 62)
    assert order == ["y", "x", "w", "z"]  # w (rank 2 in b, 1/62) beats z (rank 3 in a, 1/63)
    assert len(rrf_fuse([a, b], top=2)) == 2


def test_rrf_ties_are_deterministic() -> None:
    a = [("p", 1.0), ("q", 0.5)]
    b = [("q", 1.0), ("p", 0.5)]
    assert [c for c, _ in rrf_fuse([a, b])] == ["p", "q"]  # equal scores, first seen wins
    assert rrf_fuse([]) == []


def test_modes_and_timings(retriever: Retriever) -> None:
    r = retriever.search("revenue $4,200", mode="bm25")
    assert r.chunks[0].chunk_id == "c18" and set(r.timings_ms) >= {"bm25", "fetch", "total"}
    d = retriever.search("revenue $4,200", mode="dense")
    assert "embed_query" in d.timings_ms and "bm25" not in d.timings_ms
    h = retriever.search("revenue $4,200", mode="hybrid")
    assert "fuse" in h.timings_ms and h.chunks[0].chunk_id == "c18"
    assert len(h.chunks) == len(h.scores)


def test_rerank_reorders(retriever: Retriever) -> None:
    base = retriever.search("revenue", mode="hybrid", rerank=False)
    rer = retriever.search("revenue", mode="hybrid", rerank=True)
    assert rer.chunks[0].chunk_id == "c19" and rer.scores[0] == 0.9
    assert {c.chunk_id for c in base.chunks} == {c.chunk_id for c in rer.chunks}
    assert "rerank" in rer.timings_ms


def test_filter_restricts_results(retriever: Retriever) -> None:
    r = retriever.search("Globex widgets 2020", mode="hybrid", use_filter=True)
    assert r.query_filter is not None and r.query_filter.companies == frozenset({"Globex"})
    assert r.chunks and all(c.company == "Globex" and c.period in ("2020", "2021")
                            for c in r.chunks)
    open_ = retriever.search("Globex widgets 2020", mode="hybrid", use_filter=False)
    assert any(c.company == "Acme" for c in open_.chunks)


def test_explicit_filter_overrides_parsed_one(retriever: Retriever) -> None:
    from citesec.retrieval.filters import QueryFilter

    r = retriever.search("widgets", mode="hybrid", query_filter=QueryFilter(
        companies=frozenset({"Acme"}), years=frozenset({2021})))
    assert r.chunks and all(c.company == "Acme" and c.period == "2021" for c in r.chunks)
    empty = retriever.search("widgets", mode="hybrid",
                             query_filter=QueryFilter(companies=frozenset({"Nobody"})))
    assert empty.chunks == []


def test_rerank_n_limits_candidates_scored(retriever: Retriever) -> None:
    seen: list[int] = []

    class Counting(FakeReranker):
        def score(self, query, texts):
            seen.append(len(texts))
            return super().score(query, texts)

    retriever.reranker = Counting()
    full = retriever.search("revenue", mode="hybrid", rerank=True)
    part = retriever.search("revenue", mode="hybrid", rerank=True, rerank_n=5)
    assert seen == [len(full.chunks), 5]
    assert len(part.chunks) == len(full.chunks)  # the rest keep their place after the head
    assert len({c.chunk_id for c in part.chunks}) == len(part.chunks)


def test_invalid_args(retriever: Retriever) -> None:
    with pytest.raises(ValueError):
        retriever.search("x", mode="nope")
    retriever.reranker = None
    with pytest.raises(ValueError):
        retriever.search("x", rerank=True)
