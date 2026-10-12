"""Exact dense search with FAISS IndexFlatIP over L2-normalized vectors (cosine similarity)."""
from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

INDEX_FILE = "dense.faiss"
IDS_FILE = "dense_ids.json"


class DenseIndex:
    def __init__(self, index: faiss.Index, ids: list[str]) -> None:
        self.index, self.ids = index, ids

    @classmethod
    def build(cls, vectors: np.ndarray, ids: list[str]) -> DenseIndex:
        if len(vectors) != len(ids):
            raise ValueError("vectors and ids must have the same length")
        vecs = np.ascontiguousarray(vectors, dtype=np.float32).copy()
        faiss.normalize_L2(vecs)
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        return cls(index, list(ids))

    def search(self, qvecs: np.ndarray, k: int,
               allowed: np.ndarray | None = None) -> list[list[tuple[str, float]]]:
        """Top-k per query vector. `allowed` restricts results to those row positions."""
        q = np.ascontiguousarray(qvecs, dtype=np.float32).copy()
        faiss.normalize_L2(q)
        params = None
        limit = len(self.ids)
        if allowed is not None:
            ids = np.ascontiguousarray(allowed, dtype=np.int64)
            params = faiss.SearchParameters(sel=faiss.IDSelectorBatch(ids))
            limit = len(ids)
        if limit == 0:
            return [[] for _ in range(len(q))]
        scores, idx = self.index.search(q, min(k, limit), params=params)
        return [
            [(self.ids[j], float(s)) for j, s in zip(row_i, row_s, strict=True) if j >= 0]
            for row_i, row_s in zip(idx, scores, strict=True)
        ]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / INDEX_FILE))
        (directory / IDS_FILE).write_text(json.dumps(self.ids), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> DenseIndex:
        index = faiss.read_index(str(directory / INDEX_FILE))
        ids = json.loads((directory / IDS_FILE).read_text(encoding="utf-8"))
        return cls(index, ids)
