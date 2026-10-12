import hashlib
from pathlib import Path

import numpy as np
import pytest

from citesec.index.bm25 import BM25Index, tokenize
from citesec.index.dense import DenseIndex
from citesec.index.embed import EmbeddingCache, text_key
from citesec.index.store import ChunkStore
from citesec.schemas import Chunk

DIM = 64


def fake_embed(texts: list[str]) -> np.ndarray:
    """Deterministic bag-of-words hash embedding so texts sharing words are close."""
    out = np.zeros((len(texts), DIM), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in t.lower().split():
            out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1.0
    return out


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


def test_bm25_exact_number_match() -> None:
    texts = ["revenue was $4,200 million", "revenue was $5,100 million", "headcount grew"]
    idx = BM25Index.build(texts, ["a", "b", "c"])
    assert idx.search("what was revenue of $4,200", k=3)[0][0] == "a"
    assert idx.search("revenue 5100", k=3)[0][0] == "b"  # bare number matches comma form
