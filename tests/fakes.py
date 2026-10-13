"""Shared test doubles: a hash embedder, a keyword reranker, a canned LLM and a tiny index."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from citesec.generation.llm import LLMResult
from citesec.index.bm25 import BM25Index
from citesec.index.dense import DenseIndex
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


class FakeEmbedder:
    dim = DIM

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return fake_embed(texts)

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        return fake_embed(queries)


class FakeReranker:
    """Scores 0.9 for chunks containing the word 'target', 0.01 for the rest."""

    def score(self, query: str, texts: list[str]) -> list[float]:
        return [0.9 if "target" in t else 0.01 for t in texts]


class FakeLLM:
    """Returns canned text and records the messages it was given."""

    model = "fake-llm"

    def __init__(self, text: str = "") -> None:
        self.text = text
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages: list[dict[str, str]]) -> LLMResult:
        self.calls.append(messages)
        return LLMResult(self.text, prompt_tokens=100, completion_tokens=20, total_ms=1200.0,
                         load_ms=200.0)

    def is_up(self) -> bool:
        return True


def tiny_chunks() -> list[Chunk]:
    texts = ([f"filler text number {i} about widgets" for i in range(18)]
             + ["the revenue was $4,200 million", "this is the target passage about revenue"])
    return [
        Chunk(chunk_id=f"c{i}", doc_name=f"D{i % 2}", company="Acme" if i % 2 == 0 else "Globex",
              doc_type="10K", period="2020" if i < 10 else "2021", page=i, text=t, n_tokens=5,
              header="")
        for i, t in enumerate(texts)
    ]


def build_tiny_index(directory: Path) -> list[Chunk]:
    chunks = tiny_chunks()
    store = ChunkStore(directory / "chunks.sqlite")
    store.add(chunks)
    store.close()
    texts, ids = [c.embed_text for c in chunks], [c.chunk_id for c in chunks]
    DenseIndex.build(fake_embed(texts), ids).save(directory)
    BM25Index.build(texts, ids).save(directory)
    return chunks
