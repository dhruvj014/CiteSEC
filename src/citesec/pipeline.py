"""The full question-answering pipeline used by the API, the UI and the latency profiler.

question -> retrieve (hybrid, rerank, optional company/year filter) -> confidence gate
         -> LLM with numbered context blocks -> citation parse and verification -> result

It is the same sequence the evaluation measures for system S4, with retrieval settings read from
configs/retrieval_selected.json and the gate threshold from configs/calibration.json.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Protocol

from citesec.config import Config
from citesec.generation.abstain import llm_abstained, load_threshold, should_abstain
from citesec.generation.citations import verify_citations
from citesec.generation.llm import OllamaClient
from citesec.generation.prompts import INSUFFICIENT, build_messages, page_label
from citesec.retrieval.filters import QueryFilter
from citesec.retrieval.hybrid import Retriever
from citesec.schemas import Chunk

SNIPPET_CHARS = 320


@dataclass
class Citation:
    doc_name: str
    page: int | None  # 1-indexed PDF page, None for EDGAR sections
    section: str | None
    snippet: str
    score: float
    supported: bool | None = None  # a figure or key phrase of the claim appears in the block


@dataclass
class EvidenceBlock:
    doc_name: str
    page: int | None
    section: str | None
    snippet: str
    score: float
    cited: bool


@dataclass
class AnswerResult:
    answer: str
    abstained: bool
    abstain_reason: str | None  # "gate" | "model" | None
    confidence: float  # top reranker score, 0 to 1
    citations: list[Citation] = field(default_factory=list)
    evidence: list[EvidenceBlock] = field(default_factory=list)
    latency_ms: dict[str, float] = field(default_factory=dict)
    invalid_citations_removed: int = 0
    tokens: dict[str, int] = field(default_factory=dict)


class PipelineLike(Protocol):
    def answer(self, question: str, top_k: int = 5, companies: list[str] | None = None,
               years: list[int] | None = None) -> AnswerResult: ...

    def index_size(self) -> int: ...

    def model_name(self) -> str: ...

    def llm_up(self) -> bool: ...


def snippet(chunk: Chunk) -> str:
    text = " ".join(chunk.text.split())
    return text if len(text) <= SNIPPET_CHARS else text[:SNIPPET_CHARS].rstrip() + "..."


def chunk_page(chunk: Chunk) -> int | None:
    return page_label(chunk.page) if chunk.page is not None else None


class RagPipeline:
    def __init__(self, retriever: Retriever, llm: OllamaClient, threshold: float,
                 mode: str = "hybrid", rerank: bool = True, use_filter: bool = True,
                 top_n: int = 50, rerank_n: int | None = None, n_context: int = 5) -> None:
        self.retriever, self.llm, self.threshold = retriever, llm, threshold
        self.mode, self.rerank, self.use_filter = mode, rerank, use_filter
        self.top_n, self.rerank_n, self.n_context = top_n, rerank_n, n_context
        self._lock = threading.Lock()  # one question at a time: GPU models and SQLite are shared

    @classmethod
    def from_config(cls, cfg: Config, rerank_n: int | None = None) -> RagPipeline:
        from citesec.index.embed import CachingEmbedder, Embedder
        from citesec.retrieval.rerank import Reranker

        sel = json.loads((cfg.root / "configs" / "retrieval_selected.json").read_text())
        ev, dr = cfg["device_embed"], cfg["device_rerank"]
        embedder = CachingEmbedder(
            Embedder(cfg["models"]["embed"], device=ev, fp16=ev.startswith("cuda")))
        reranker = Reranker(cfg["models"]["reranker"], device=dr, fp16=dr.startswith("cuda"))
        retriever = Retriever(cfg.processed_dir / "index" / sel["chunker"], embedder, reranker,
                              cfg["retrieval"]["rrf_k"])
        return cls(retriever, OllamaClient.from_config(cfg),
                   load_threshold(cfg.root / "configs" / "calibration.json"),
                   mode=sel["mode"], rerank=sel["rerank"], use_filter=sel["use_filter"],
                   top_n=cfg["retrieval"]["top_n"], rerank_n=rerank_n,
                   n_context=cfg["generation"]["n_context_blocks"])

    def index_size(self) -> int:
        return len(self.retriever.dense.ids)

    def model_name(self) -> str:
        return self.llm.model

    def llm_up(self) -> bool:
        return self.llm.is_up()

    def _explicit_filter(self, companies: list[str] | None,
                         years: list[int] | None) -> QueryFilter | None:
        """A filter from the caller. Company names are matched against the index, case-blind."""
        if not companies and not years:
            return None
        names = None
        if companies:
            known = {n.lower(): n for n in self.retriever.meta.names}
            names = frozenset(known[c.lower()] for c in companies if c.lower() in known)
        return QueryFilter(companies=names if companies else None,
                           years=frozenset(years) if years else None)

    def answer(self, question: str, top_k: int = 5, companies: list[str] | None = None,
               years: list[int] | None = None) -> AnswerResult:
        with self._lock:
            return self._answer(question, top_k, companies, years)

    def _answer(self, question: str, top_k: int, companies: list[str] | None,
                years: list[int] | None) -> AnswerResult:
        t_start = time.perf_counter()
        res = self.retriever.search(
            question, self.mode, self.rerank, self.use_filter, self.top_n,
            query_filter=self._explicit_filter(companies, years), rerank_n=self.rerank_n)
        t = res.timings_ms
        lat = {"retrieve": round(sum(v for k, v in t.items() if k not in ("rerank", "total")), 1),
               "rerank": round(t.get("rerank", 0.0), 1), "generate": 0.0}
        k = min(top_k, self.n_context) if top_k else self.n_context
        chunks, scores = res.chunks[:k], res.scores[:k]
        confidence = float(scores[0]) if scores else 0.0

        def finish(result: AnswerResult) -> AnswerResult:
            result.latency_ms = {**lat, "total": round((time.perf_counter() - t_start) * 1000, 1)}
            return result

        evidence = [EvidenceBlock(c.doc_name, chunk_page(c), c.section, snippet(c), float(s), False)
                    for c, s in zip(chunks, scores, strict=True)]
        if should_abstain(scores, self.threshold):
            return finish(AnswerResult(INSUFFICIENT, True, "gate", confidence, [], evidence))

        llm_res = self.llm.chat(build_messages(question, chunks))
        lat["generate"] = round(llm_res.total_ms - llm_res.load_ms, 1)
        tokens = {"prompt": llm_res.prompt_tokens, "completion": llm_res.completion_tokens}
        if llm_abstained(llm_res.text):
            return finish(AnswerResult(INSUFFICIENT, True, "model", confidence, [], evidence,
                                       tokens=tokens))

        report = verify_citations(llm_res.text, chunks)
        by_id = {c.chunk_id: (c, s) for c, s in zip(chunks, scores, strict=True)}
        cited: dict[str, Citation] = {}
        for c in report.citations:
            if c.valid and c.chunk_id and c.chunk_id not in cited:
                chunk, score = by_id[c.chunk_id]
                cited[c.chunk_id] = Citation(chunk.doc_name, chunk_page(chunk), chunk.section,
                                             snippet(chunk), float(score), c.supported)
        for block, chunk in zip(evidence, chunks, strict=True):
            block.cited = chunk.chunk_id in cited
        return finish(AnswerResult(report.clean_text, False, None, confidence,
                                   list(cited.values()), evidence,
                                   invalid_citations_removed=report.n_invalid, tokens=tokens))
