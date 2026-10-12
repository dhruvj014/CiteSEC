from pathlib import Path

import numpy as np
import pytest
from tests.test_index import fake_embed

from citesec.index.bm25 import BM25Index
from citesec.index.dense import DenseIndex
from citesec.index.store import ChunkStore
from citesec.retrieval.hybrid import Retriever, rrf_fuse
from citesec.schemas import Chunk


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


class FakeEmbedder:
    dim = 64

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return fake_embed(texts)

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        return fake_embed(queries)


class FakeReranker:
    """Prefers chunks whose text contains the word 'target'."""

    def score(self, query: str, texts: list[str]) -> list[float]:
        return [10.0 if "target" in t else 0.0 for t in texts]


@pytest.fixture
def retriever(tmp_path: Path) -> Retriever:
    chunks = [
        Chunk(chunk_id=f"c{i}", doc_name=f"D{i % 2}", company="Acme" if i % 2 == 0 else "Globex",
              doc_type="10K", period="2020" if i < 10 else "2021", page=i, text=t, n_tokens=5,
              header="")
        for i, t in enumerate(
            [f"filler text number {i} about widgets" for i in range(18)]
            + ["the revenue was $4,200 million", "this is the target passage about revenue"]
        )
    ]
    store = ChunkStore(tmp_path / "chunks.sqlite")
    store.add(chunks)
    store.close()
    texts, ids = [c.embed_text for c in chunks], [c.chunk_id for c in chunks]
    DenseIndex.build(fake_embed(texts), ids).save(tmp_path)
    BM25Index.build(texts, ids).save(tmp_path)
    return Retriever(tmp_path, FakeEmbedder(), FakeReranker())


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
    assert rer.chunks[0].chunk_id == "c19" and rer.scores[0] == 10.0
    assert {c.chunk_id for c in base.chunks} == {c.chunk_id for c in rer.chunks}
    assert "rerank" in rer.timings_ms


def test_filter_restricts_results(retriever: Retriever) -> None:
    r = retriever.search("Globex widgets 2020", mode="hybrid", use_filter=True)
    assert r.query_filter is not None and r.query_filter.companies == frozenset({"Globex"})
    assert r.chunks and all(c.company == "Globex" and c.period in ("2020", "2021")
                            for c in r.chunks)
    open_ = retriever.search("Globex widgets 2020", mode="hybrid", use_filter=False)
    assert any(c.company == "Acme" for c in open_.chunks)


def test_invalid_args(retriever: Retriever) -> None:
    with pytest.raises(ValueError):
        retriever.search("x", mode="nope")
    retriever.reranker = None
    with pytest.raises(ValueError):
        retriever.search("x", rerank=True)
