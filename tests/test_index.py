from pathlib import Path

import numpy as np
import pytest
from tests.fakes import fake_embed

from citesec.index.bm25 import BM25Index, tokenize
from citesec.index.dense import DenseIndex
from citesec.index.embed import EmbeddingCache, text_key
from citesec.index.store import ChunkStore
from citesec.schemas import Chunk


def make_chunks(n: int = 30) -> list[Chunk]:
    return [
        Chunk(chunk_id=f"c{i}", doc_name=f"D{i % 3}", company="Acme", doc_type="10K", period="2022",
              page=i, section=None, text=f"unique{i} revenue was ${i * 100},500 in segment s{i}",
              is_table=i % 5 == 0, n_tokens=12, header="Acme | 10K 2022")
        for i in range(n)
    ]


def test_tokenize_keeps_financial_numbers() -> None:
    toks = tokenize("Revenue was $4,200.5 million, up 12.5% from (1,234) last year")
    assert "$4,200.5" in toks and "4200.5" in toks
    assert "12.5%" in toks and "12.5" in toks
    assert "1,234" in toks and "1234" in toks  # parentheses (negatives) stripped
    assert "the" not in tokenize("the revenue")
    # digit-led names stay whole, they must not split into a number and a letter
    assert tokenize("Is 3M's 10K filing here?") == ["3m", "10k", "filing", "here"]
    assert tokenize("grew 3 million") == ["grew", "3", "million"]
    # a table cell "4,200.5" and a query "$4,200.5" share the bare token
    assert set(tokenize("4,200.5")) & set(toks) == {"4200.5"}


def test_store_roundtrip_and_order(tmp_path: Path) -> None:
    chunks = make_chunks()
    store = ChunkStore(tmp_path / "c.sqlite")
    store.add(chunks[:10])
    store.add(chunks[10:])
    assert store.count() == 30
    assert [c.chunk_id for c in store.iter_all()] == [c.chunk_id for c in chunks]
    assert store.get("c5") == chunks[5]
    assert [c.chunk_id for c in store.get_many(["c9", "c1", "nope"])] == ["c9", "c1"]
    store.reset()
    assert store.count() == 0


def test_embedding_cache_roundtrip(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path / "e.sqlite")
    keys = [text_key("m", "a"), text_key("m", "b")]
    vecs = np.random.default_rng(0).normal(size=(2, 8)).astype(np.float32)
    cache.put_many(keys, vecs)
    got = cache.get_many(keys + ["missing"])
    assert set(got) == set(keys)
    assert np.allclose(got[keys[0]].astype(np.float32), vecs[0], atol=1e-2)
    assert text_key("m1", "a") != text_key("m2", "a")


def test_caching_embedder_skips_the_model_for_repeated_queries() -> None:
    from tests.fakes import FakeEmbedder

    from citesec.index.embed import CachingEmbedder

    class Counting(FakeEmbedder):
        calls: list[list[str]] = []

        def encode_queries(self, queries):
            self.calls.append(list(queries))
            return super().encode_queries(queries)

    inner = Counting()
    inner.calls = []
    cache = CachingEmbedder(inner, max_size=2)
    a = cache.encode_queries(["alpha"])
    assert np.array_equal(cache.encode_queries(["alpha"]), a) and inner.calls == [["alpha"]]
    both = cache.encode_queries(["alpha", "beta", "beta"])  # only beta is new, order is kept
    assert inner.calls[-1] == ["beta"] and both.shape[0] == 3
    assert np.array_equal(both[1], both[2]) and np.array_equal(both[0], a[0])
    cache.encode_queries(["gamma"])  # evicts the least recently used (alpha), size stays 2
    cache.encode_queries(["alpha"])
    # hits: alpha (2nd call), alpha and the repeated beta (3rd call). misses: alpha, beta, gamma,
    # and alpha again after it was evicted
    assert inner.calls[-1] == ["alpha"] and cache.hits == 3 and cache.misses == 4
    assert cache.dim == inner.dim


def test_dense_round_trip_rank_one_and_save_load(tmp_path: Path) -> None:
    chunks = make_chunks()
    vecs = fake_embed([c.embed_text for c in chunks])
    idx = DenseIndex.build(vecs, [c.chunk_id for c in chunks])
    idx.save(tmp_path)
    loaded = DenseIndex.load(tmp_path)
    res = loaded.search(vecs, k=3)
    assert [r[0][0] for r in res] == [c.chunk_id for c in chunks]
    assert res[0][0][1] == pytest.approx(1.0, abs=1e-4)
    assert len(loaded.search(vecs[:1], k=999)[0]) == 30  # k larger than the index is clamped


def test_bm25_round_trip_rank_one_and_save_load(tmp_path: Path) -> None:
    chunks = make_chunks()
    idx = BM25Index.build([c.embed_text for c in chunks], [c.chunk_id for c in chunks])
    idx.save(tmp_path)
    loaded = BM25Index.load(tmp_path)
    for c in chunks:
        assert loaded.search(c.embed_text, k=3)[0][0] == c.chunk_id
    assert loaded.search("zzzzunknownterm", k=3) == []


def test_allowed_restricts_dense_and_bm25() -> None:
    chunks = make_chunks()
    texts = [c.embed_text for c in chunks]
    ids = [c.chunk_id for c in chunks]
    allowed = np.array([3, 4, 5])
    dense = DenseIndex.build(fake_embed(texts), ids)
    res = dense.search(fake_embed(texts[:1]), k=10, allowed=allowed)[0]
    assert {r[0] for r in res} <= {"c3", "c4", "c5"} and len(res) == 3
    bm25 = BM25Index.build(texts, ids)
    hits = bm25.search("revenue segment", k=10, allowed=allowed)
    assert hits and {h[0] for h in hits} <= {"c3", "c4", "c5"}
    assert bm25.search("revenue", k=5, allowed=np.array([], dtype=int)) == []
    assert dense.search(fake_embed(texts[:1]), k=5, allowed=np.array([], dtype=int)) == [[]]


def test_bm25_exact_number_match() -> None:
    texts = ["revenue was $4,200 million", "revenue was $5,100 million", "headcount grew"]
    idx = BM25Index.build(texts, ["a", "b", "c"])
    assert idx.search("what was revenue of $4,200", k=3)[0][0] == "a"
    assert idx.search("revenue 5100", k=3)[0][0] == "b"  # bare number matches comma form
