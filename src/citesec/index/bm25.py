"""BM25 over chunks with a tokenizer that keeps financial numbers intact.

Tokens: lowercase words, plus numbers kept whole with their $ and % ("$4,200.5", "12.5%").
Each number also emits its bare form ("4200.5") so "$4,200" in a query can match "4,200" in a table
cell. The same function tokenizes documents and queries.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import bm25s

IDS_FILE = "bm25_ids.json"
_TOKEN_RE = re.compile(r"\(?\$?\d[\d,]*(?:\.\d+)?%?\)?|[a-z][a-z0-9'\-]*")
STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was were "
    "what which who will with".split()
)


def tokenize(text: str) -> list[str]:
    out: list[str] = []
    for tok in _TOKEN_RE.findall(text.lower()):
        if tok[0].isalpha():
            if tok not in STOPWORDS:
                out.append(tok)
            continue
        tok = tok.strip("()")
        out.append(tok)
        bare = tok.replace("$", "").replace(",", "").rstrip("%")
        if bare and bare != tok:
            out.append(bare)
    return out


class BM25Index:
    def __init__(self, retriever: bm25s.BM25, ids: list[str]) -> None:
        self.retriever, self.ids = retriever, ids

    @classmethod
    def build(cls, texts: list[str], ids: list[str]) -> BM25Index:
        if len(texts) != len(ids):
            raise ValueError("texts and ids must have the same length")
        retriever = bm25s.BM25()
        retriever.index([tokenize(t) or ["<empty>"] for t in texts], show_progress=False)
        return cls(retriever, list(ids))

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        toks = [t for t in tokenize(query) if t in self.retriever.vocab_dict]
        if not toks:
            return []
        docs, scores = self.retriever.retrieve([toks], k=min(k, len(self.ids)),
                                               show_progress=False)
        return [(self.ids[int(d)], float(s)) for d, s in zip(docs[0], scores[0], strict=True)
                if s > 0]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.retriever.save(str(directory / "bm25"))
        (directory / IDS_FILE).write_text(json.dumps(self.ids), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> BM25Index:
        retriever = bm25s.BM25.load(str(directory / "bm25"))
        ids = json.loads((directory / IDS_FILE).read_text(encoding="utf-8"))
        return cls(retriever, ids)
