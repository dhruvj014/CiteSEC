"""bge embeddings with a resumable on-disk cache keyed by hash of (model, text)."""
from __future__ import annotations

import hashlib
import logging
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np

log = logging.getLogger(__name__)

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
CALL_SIZE = 2048  # texts per encode call: sentence-transformers length-sorts within a call


class EmbedderLike(Protocol):
    dim: int

    def encode_passages(self, texts: list[str]) -> np.ndarray: ...

    def encode_queries(self, queries: list[str]) -> np.ndarray: ...


def text_key(model: str, text: str) -> str:
    return hashlib.sha1(f"{model}\n{text}".encode()).hexdigest()


class EmbeddingCache:
    """SQLite blob cache of float16 vectors. Safe to interrupt: every call commits."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS emb (h TEXT PRIMARY KEY, v BLOB)")

    def get_many(self, keys: list[str]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for i in range(0, len(keys), 500):
            part = keys[i : i + 500]
            q = f"SELECT h, v FROM emb WHERE h IN ({','.join('?' * len(part))})"
            for h, v in self.db.execute(q, part):
                out[h] = np.frombuffer(v, dtype=np.float16)
        return out

    def put_many(self, keys: list[str], vecs: np.ndarray) -> None:
        self.db.executemany(
            "INSERT OR REPLACE INTO emb VALUES (?, ?)",
            [(k, v.astype(np.float16).tobytes()) for k, v in zip(keys, vecs, strict=True)],
        )
        self.db.commit()


class Embedder:
    def __init__(self, model_name: str, device: str = "cuda", fp16: bool = True,
                 batch_size: int = 64, cache_path: Path | None = None, max_len: int = 512) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)
        self.model.max_seq_length = max_len
        if fp16 and device.startswith("cuda"):
            self.model.half()
        self.dim = int(self.model.get_sentence_embedding_dimension())
        self.cache = EmbeddingCache(cache_path) if cache_path else None

    def _encode(self, texts: list[str]) -> np.ndarray:
        v = self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                              convert_to_numpy=True, show_progress_bar=False)
        return v.astype(np.float32)

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        return self._encode([QUERY_PREFIX + q for q in queries])

    def encode_passages(self, texts: list[str],
                        progress: Callable[[int, int], None] | None = None) -> np.ndarray:
        """Passages get no prefix. Cached vectors are reused, so reruns resume."""
        keys = [text_key(self.model_name, t) for t in texts]
        have = self.cache.get_many(list(set(keys))) if self.cache else {}
        todo = [i for i, k in enumerate(keys) if k not in have]
        seen: set[str] = set()
        todo = [i for i in todo if not (keys[i] in seen or seen.add(keys[i]))]
        log.info("embedding %d new of %d passages (%d cached)", len(todo), len(texts),
                 len(texts) - len(todo))
        for s in range(0, len(todo), CALL_SIZE):
            idx = todo[s : s + CALL_SIZE]
            vecs = self._encode([texts[i] for i in idx])
            if self.cache:
                self.cache.put_many([keys[i] for i in idx], vecs)
            for i, v in zip(idx, vecs, strict=True):
                have[keys[i]] = v.astype(np.float16)
            if progress:
                progress(min(s + CALL_SIZE, len(todo)), len(todo))
        return np.stack([have[k] for k in keys]).astype(np.float32)
