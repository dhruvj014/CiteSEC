"""Dense, BM25 and hybrid retrieval with Reciprocal Rank Fusion, optional rerank and filter."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from citesec.index.bm25 import BM25Index
from citesec.index.dense import DenseIndex
from citesec.index.embed import EmbedderLike
from citesec.index.store import ChunkStore
from citesec.retrieval.filters import AliasTable, ChunkMeta, QueryFilter, parse_query
from citesec.retrieval.rerank import RerankerLike
from citesec.schemas import Chunk

MODES = ("dense", "bm25", "hybrid")


def rrf_fuse(rankings: list[list[tuple[str, float]]], k: int = 60,
             top: int = 50) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: score(d) = sum over rankers of 1 / (k + rank), rank starting at 1.

    Ties break by first appearance across the input rankings, so results are deterministic.
    """
    score: dict[str, float] = {}
    first: dict[str, int] = {}
    for ranking in rankings:
        for rank, (cid, _) in enumerate(ranking, start=1):
            score[cid] = score.get(cid, 0.0) + 1.0 / (k + rank)
            first.setdefault(cid, len(first))
    ordered = sorted(score, key=lambda c: (-score[c], first[c]))
    return [(c, score[c]) for c in ordered[:top]]


@dataclass
class RetrievalResult:
    chunks: list[Chunk]  # best first
    scores: list[float]  # rerank score in [0, 1] if reranked, else the retriever's own score
    timings_ms: dict[str, float] = field(default_factory=dict)
    query_filter: QueryFilter | None = None


class Retriever:
    def __init__(self, index_dir: Path, embedder: EmbedderLike,
                 reranker: RerankerLike | None = None, rrf_k: int = 60,
                 aliases: AliasTable | None = None) -> None:
        self.store = ChunkStore(index_dir / "chunks.sqlite")
        self.dense = DenseIndex.load(index_dir)
        self.bm25 = BM25Index.load(index_dir)
        self.embedder, self.reranker, self.rrf_k = embedder, reranker, rrf_k
        companies, periods = self.store.company_period()
        self.meta = ChunkMeta(companies, periods)
        self.aliases = aliases or AliasTable({n: None for n in self.meta.names})

    def search(self, query: str, mode: str = "hybrid", rerank: bool = False,
               use_filter: bool = False, top_n: int = 50) -> RetrievalResult:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if rerank and self.reranker is None:
            raise ValueError("rerank=True needs a reranker")
        t: dict[str, float] = {}

        def tick(name: str, t0: float) -> None:
            t[name] = (time.perf_counter() - t0) * 1000

        qf = None
        allowed = None
        if use_filter:
            t0 = time.perf_counter()
            qf = parse_query(query, self.aliases)
            allowed = self.meta.allowed(qf)
            tick("filter", t0)

        rankings: list[list[tuple[str, float]]] = []
        if mode in ("dense", "hybrid"):
            t0 = time.perf_counter()
            qvec = self.embedder.encode_queries([query])
            tick("embed_query", t0)
            t0 = time.perf_counter()
            rankings.append(self.dense.search(qvec, top_n, allowed)[0])
            tick("dense", t0)
        if mode in ("bm25", "hybrid"):
            t0 = time.perf_counter()
            rankings.append(self.bm25.search(query, top_n, allowed))
            tick("bm25", t0)
        if mode == "hybrid":
            t0 = time.perf_counter()
            fused = rrf_fuse(rankings, self.rrf_k, top_n)
            tick("fuse", t0)
        else:
            fused = rankings[0]

        t0 = time.perf_counter()
        chunks = self.store.get_many([cid for cid, _ in fused])
        tick("fetch", t0)
        score_of = dict(fused)
        scores = [score_of[c.chunk_id] for c in chunks]
        if rerank and chunks:
            t0 = time.perf_counter()
            rs = self.reranker.score(query, [c.embed_text for c in chunks])
            order = sorted(range(len(chunks)), key=lambda i: -rs[i])  # stable for equal scores
            chunks = [chunks[i] for i in order]
            scores = [rs[i] for i in order]
            tick("rerank", t0)
        t["total"] = sum(t.values())
        return RetrievalResult(chunks, scores, t, qf)
