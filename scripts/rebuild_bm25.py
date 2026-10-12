"""Rebuild only the BM25 index from the stored chunks, after a tokenizer change (about 1 minute).

Dense vectors depend only on chunk text, so they are not touched.
"""
from __future__ import annotations

import time

from citesec.config import load_config
from citesec.index.bm25 import BM25Index
from citesec.index.store import ChunkStore


def main() -> None:
    cfg = load_config()
    for chunker in ("naive", "structured"):
        d = cfg.processed_dir / "index" / chunker
        t0 = time.perf_counter()
        chunks = list(ChunkStore(d / "chunks.sqlite").iter_all())
        BM25Index.build([c.embed_text for c in chunks], [c.chunk_id for c in chunks]).save(d)
        print(f"{chunker}: {len(chunks)} chunks, {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
