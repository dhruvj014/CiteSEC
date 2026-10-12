"""Phase 2 verification. Writes results/phase2_validation.json.

Checks, per chunker: schema, no empty text, token cap, index sizes. For the structured chunker on
FinanceBench docs: valid page on every chunk, no chunk spans pages (5-gram containment), table
integrity on 20 gold table pages. Round trip: a chunk's own text retrieves that chunk at rank 1.
"""
from __future__ import annotations

import json
import random
import re

from build_index import index_dir
from citesec.chunking.blocks import clean_markup, drop_empty_columns, parse_blocks, table_header
from citesec.chunking.tokens import HFTokenizer, model_len
from citesec.config import load_config
from citesec.index.bm25 import BM25Index
from citesec.index.dense import DenseIndex
from citesec.index.embed import Embedder
from citesec.index.store import ChunkStore
from citesec.ingest.financebench import load_questions
from citesec.ingest.pdf_parser import cache_path, read_pages

CAP = 512
NGRAM = 5
WORD = re.compile(r"[a-z0-9]+")  # alphanumerics only, so markdown underscores do not count
MAX_MISSING = 0.05


def words(text: str) -> list[str]:
    return WORD.findall(text.lower())


def ngrams(ws: list[str]) -> set[tuple[str, ...]]:
    return {tuple(ws[i : i + NGRAM]) for i in range(len(ws) - NGRAM + 1)}


class PageText:
    """A cleaned page as a word list, 5-gram set and joined string for short-chunk lookups."""

    def __init__(self, markdown: str) -> None:
        cleaned = "\n".join(clean_markup(x) for x in markdown.splitlines())
        self.ws = words(cleaned)
        self.grams = ngrams(self.ws)
        self.joined = " " + " ".join(self.ws) + " "

    def missing(self, chunk_words: list[str]) -> list[tuple[str, ...]]:
        """n-grams of the chunk absent from this page. Chunks under 5 words use a substring test."""
        if len(chunk_words) < NGRAM:
            ok = " " + " ".join(chunk_words) + " " in self.joined
            return [] if ok else [tuple(chunk_words)]
        return [g for g in ngrams(chunk_words) if g not in self.grams]


def basic_checks(chunks, tok) -> dict:
    over = [c.chunk_id for c in chunks if model_len(tok, c.embed_text) > CAP]
    return {
        "n_chunks": len(chunks),
        "empty_text": sum(1 for c in chunks if not c.text.strip()),
        "over_cap": len(over),
        "over_cap_examples": over[:5],
        "n_tokens_field_matches": sum(
            1 for c in chunks[:2000] if c.n_tokens == model_len(tok, c.embed_text)
        ),
        "max_tokens": max(c.n_tokens for c in chunks),
        "unique_ids": len({c.chunk_id for c in chunks}) == len(chunks),
    }


def page_checks(chunks, cfg) -> dict:
    """Structured chunks on FinanceBench docs: valid page, and text contained in that page.

    A chunk "spans pages" if more than 5% of its 5-grams are missing from its own page AND most of
    the missing ones appear on an adjacent page. Other misses (junctions where a table header is
    repeated, short chunks) are counted separately as not_contained.
    """
    pages: dict[str, list[PageText]] = {}
    bad_page, not_contained, spanning = [], [], []
    for c in chunks:
        if c.page is None:
            continue
        if c.doc_name not in pages:
            recs = read_pages(cache_path(cfg.processed_dir, c.doc_name))
            pages[c.doc_name] = [PageText(p.markdown) for p in recs]
        doc = pages[c.doc_name]
        if not (0 <= c.page < len(doc)):
            bad_page.append(c.chunk_id)
            continue
        cw = words(c.text)
        miss = doc[c.page].missing(cw)
        if len(miss) / max(len(cw) - NGRAM + 1, 1) <= MAX_MISSING:
            continue
        not_contained.append(c.chunk_id)
        near = [doc[j] for j in (c.page - 1, c.page + 1) if 0 <= j < len(doc)]
        if sum(any(g in p.grams for p in near) for g in miss) > len(miss) / 2:
            spanning.append(c.chunk_id)
    n_fb = sum(1 for c in chunks if c.page is not None)
    return {
        "fb_chunks": n_fb,
        "fb_chunks_with_valid_page": n_fb - len(bad_page),
        "pct_valid_page": round(100 * (n_fb - len(bad_page)) / max(n_fb, 1), 3),
        "chunks_spanning_pages": len(spanning),
        "spanning_examples": spanning[:5],
        "chunks_over_5pct_ngrams_missing_from_own_page": len(not_contained),
        "not_contained_examples": not_contained[:5],
    }


def table_checks(chunks, cfg, seed: int) -> dict:
    qs = load_questions(cfg.fb_dir)
    gold = sorted({(e.evidence_doc_name, e.evidence_page_num) for q in qs for e in q.evidence})
    by_page: dict[tuple[str, int], list] = {}
    for c in chunks:
        if c.page is not None and c.is_table:
            by_page.setdefault((c.doc_name, c.page), []).append(c)
    table_pages = []
    for doc, page in gold:
        md = read_pages(cache_path(cfg.processed_dir, doc))[page].markdown
        tables = [b for b in parse_blocks(md) if b.kind == "table"]
        if tables:
            table_pages.append((doc, page, tables))
    sample = random.Random(seed).sample(table_pages, min(20, len(table_pages)))
    intact = split_ok = broken = 0
    examples = []
    for doc, page, tables in sample:
        pieces = by_page.get((doc, page), [])
        for t in tables:
            rows = drop_empty_columns(t.text.splitlines())
            if not rows:
                continue
            head = table_header(rows)
            body = rows[len(head):]
            body_set = set(body)
            mine = [p for p in pieces if body_set & set(p.text.splitlines())]
            got = {ln for p in mine for ln in p.text.splitlines()}
            hdr_ok = all(head[0] in p.text.splitlines() for p in mine)
            if mine and body_set <= got and hdr_ok:
                intact += len(mine) == 1
                split_ok += len(mine) > 1
                if len(examples) < 5 and (len(mine) > 1 or len(examples) < 3):
                    examples.append({"doc": doc, "page": page, "pieces": len(mine),
                                     "first_piece": mine[0].text[:600]})
            else:
                broken += 1
    return {"pages_sampled": len(sample), "tables_intact_single_chunk": intact,
            "tables_split_with_repeated_header": split_ok, "tables_broken": broken,
            "examples": examples}


def round_trip(cfg, chunker: str, store: ChunkStore, n: int, seed: int, device: str) -> dict:
    d = index_dir(cfg, chunker)
    chunks = list(store.iter_all())
    sample = random.Random(seed).sample(chunks, min(n, len(chunks)))
    dense, bm25 = DenseIndex.load(d), BM25Index.load(d)
    slug = cfg["models"]["embed"].replace("/", "_")
    emb = Embedder(cfg["models"]["embed"], device=device, fp16=device.startswith("cuda"),
                   cache_path=cfg.processed_dir / "embcache" / f"{slug}.sqlite")
    vecs = emb.encode_passages([c.embed_text for c in sample])
    dense_hits = [dense.search(v[None], 1)[0][0][0] for v in vecs]
    d_ok = sum(1 for c, h in zip(sample, dense_hits, strict=True)
               if h == c.chunk_id or store.get(h).text == c.text)
    b_ok = 0
    for c in sample:
        top = bm25.search(c.embed_text, 1)
        b_ok += bool(top) and (top[0][0] == c.chunk_id or store.get(top[0][0]).text == c.text)
    return {"n": len(sample), "dense_rank1_pct": round(100 * d_ok / len(sample), 2),
            "bm25_rank1_pct": round(100 * b_ok / len(sample), 2)}


def main() -> None:
    cfg = load_config()
    tok = HFTokenizer(cfg["models"]["embed"])
    device = cfg["device_embed"]
    out: dict = {}
    for chunker in ("naive", "structured"):
        store = ChunkStore(index_dir(cfg, chunker) / "chunks.sqlite")
        chunks = list(store.iter_all())
        res = {"basic": basic_checks(chunks, tok)}
        if chunker == "structured":
            res["pages"] = page_checks(chunks, cfg)
            res["tables"] = table_checks(chunks, cfg, cfg.seed)
        res["round_trip"] = round_trip(cfg, chunker, store, 300, cfg.seed, device)
        out[chunker] = res
        print(chunker, json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "examples"}
                                   for k, v in res.items()}, indent=2))
    (cfg.results_dir / "phase2_validation.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
