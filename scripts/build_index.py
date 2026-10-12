"""Build the chunk store, dense (FAISS) and BM25 indexes for one or both chunkers.

Usage:
    python scripts/build_index.py --chunker structured
    python scripts/build_index.py --chunker both
Stages are timed and written to results/index_stats.json. Embeddings are cached, so reruns resume.
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from citesec.chunking.common import Document, iter_documents
from citesec.chunking.naive import chunk_naive
from citesec.chunking.structured import chunk_structured
from citesec.chunking.tokens import HFTokenizer
from citesec.config import Config, load_config
from citesec.index.bm25 import BM25Index
from citesec.index.dense import DenseIndex
from citesec.index.store import ChunkStore
from citesec.schemas import Chunk

log = logging.getLogger("build_index")
_TOK: HFTokenizer | None = None
_CFG: dict = {}


def index_dir(cfg: Config, chunker: str) -> Path:
    return cfg.processed_dir / "index" / chunker


def _init_worker(model: str, params: dict) -> None:
    global _TOK, _CFG
    _TOK, _CFG = HFTokenizer(model), params


def _chunk_doc(args: tuple[str, Document]) -> list[Chunk]:
    chunker, doc = args
    assert _TOK is not None
    if chunker == "naive":
        return chunk_naive(doc, _TOK, _CFG["naive_tokens"], _CFG["naive_overlap"])
    return chunk_structured(doc, _TOK, _CFG["structured_target"], _CFG["structured_cap"],
                            _CFG["structured_overlap"])


def chunk_all(cfg: Config, chunker: str, workers: int, limit_docs: int | None) -> list[Chunk]:
    docs = list(iter_documents(cfg))
    if limit_docs:
        docs = docs[:limit_docs]
    params = dict(cfg["chunking"])
    with ProcessPoolExecutor(workers, initializer=_init_worker,
                             initargs=(cfg["models"]["embed"], params)) as pool:
        per_doc = list(pool.map(_chunk_doc, [(chunker, d) for d in docs], chunksize=2))
    return [c for chunks in per_doc for c in chunks]


def dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def build(cfg: Config, chunker: str, workers: int, limit_docs: int | None, device: str) -> dict:
    from citesec.index.embed import Embedder

    out = index_dir(cfg, chunker + ("_smoke" if limit_docs else ""))
    t = {}
    t0 = time.perf_counter()
    chunks = chunk_all(cfg, chunker, workers, limit_docs)
    t["chunk_s"] = time.perf_counter() - t0
    log.info("%s: %d chunks in %.1fs", chunker, len(chunks), t["chunk_s"])

    store = ChunkStore(out / "chunks.sqlite")
    store.reset()
    store.add(chunks)
    store.close()

    texts = [c.embed_text for c in chunks]
    ids = [c.chunk_id for c in chunks]
    slug = cfg["models"]["embed"].replace("/", "_")
    emb = Embedder(cfg["models"]["embed"], device=device, fp16=device.startswith("cuda"),
                   cache_path=cfg.processed_dir / "embcache" / f"{slug}.sqlite")
    t0 = time.perf_counter()
    vecs = emb.encode_passages(texts, progress=lambda a, b: log.info("embedded %d/%d", a, b))
    t["embed_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    DenseIndex.build(vecs, ids).save(out)
    t["dense_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    BM25Index.build(texts, ids).save(out)
    t["bm25_s"] = time.perf_counter() - t0

    toks = sorted(c.n_tokens for c in chunks)
    stats = {
        "n_docs": len({c.doc_name for c in chunks}),
        "n_chunks": len(chunks),
        "n_chunks_financebench": sum(1 for c in chunks if c.page is not None),
        "n_chunks_edgar": sum(1 for c in chunks if c.page is None),
        "n_table_chunks": sum(1 for c in chunks if c.is_table),
        "tokens_mean": round(statistics.fmean(toks), 1),
        "tokens_p50": toks[len(toks) // 2],
        "tokens_p95": toks[int(len(toks) * 0.95)],
        "tokens_max": toks[-1],
        "seconds": {k: round(v, 1) for k, v in t.items()},
        "index_bytes": dir_size(out),
        "embed_model": cfg["models"]["embed"],
        "embed_device": device,
        "limit_docs": limit_docs,
    }
    return stats


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunker", choices=["naive", "structured", "both"], default="both")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit-docs", type=int, default=None, help="smoke test on first N docs")
    ap.add_argument("--device", default=None, help="override device_embed from config")
    args = ap.parse_args()
    cfg = load_config()
    device = args.device or cfg["device_embed"]
    path = cfg.results_dir / "index_stats.json"
    all_stats = json.loads(path.read_text()) if path.exists() else {}
    for chunker in (["naive", "structured"] if args.chunker == "both" else [args.chunker]):
        all_stats[chunker] = build(cfg, chunker, args.workers, args.limit_docs, device)
        if not args.limit_docs:  # smoke runs must not overwrite real stats
            cfg.results_dir.mkdir(exist_ok=True)
            path.write_text(json.dumps(all_stats, indent=2))
        print(json.dumps({chunker: all_stats[chunker]}, indent=2))


if __name__ == "__main__":
    main()
