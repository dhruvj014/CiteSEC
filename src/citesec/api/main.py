"""FastAPI service.

    uvicorn citesec.api.main:app --port 8000

POST /query answers a question with citations, GET /health, GET /metrics (Prometheus), and the
OpenAPI docs at /docs. Indexes and models are loaded once at startup.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST
from pydantic import BaseModel, Field

from citesec.api.metrics import Metrics
from citesec.config import load_config
from citesec.pipeline import AnswerResult, PipelineLike, RagPipeline

log = logging.getLogger("citesec.api")


class Filters(BaseModel):
    companies: list[str] | None = Field(None, description="Restrict to these companies")
    years: list[int] | None = Field(None, description="Restrict to filings of these years")


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(5, ge=1, le=10, description="Context blocks given to the model (max 5)")
    filters: Filters | None = None


class CitationOut(BaseModel):
    doc_name: str
    page: int | None = Field(None, description="1-indexed PDF page, null for EDGAR sections")
    section: str | None = None
    snippet: str
    score: float
    supported: bool | None = Field(
        None, description="A figure or key phrase of the claim appears in the block")


class EvidenceOut(BaseModel):
    doc_name: str
    page: int | None
    section: str | None
    snippet: str
    score: float
    cited: bool


class LatencyOut(BaseModel):
    retrieve: float
    rerank: float
    generate: float
    total: float


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    abstained: bool
    abstain_reason: str | None = Field(None, description="gate (low retrieval confidence) or model")
    confidence: float = Field(description="Top reranker score, 0 to 1")
    latency_ms: LatencyOut
    evidence: list[EvidenceOut]
    invalid_citations_removed: int
    model: str


class HealthResponse(BaseModel):
    status: str
    index_chunks: int
    model: str
    llm_reachable: bool


def to_response(r: AnswerResult, model: str) -> QueryResponse:
    return QueryResponse(
        answer=r.answer, abstained=r.abstained, abstain_reason=r.abstain_reason,
        confidence=r.confidence, model=model,
        citations=[CitationOut(**c.__dict__) for c in r.citations],
        evidence=[EvidenceOut(**e.__dict__) for e in r.evidence],
        latency_ms=LatencyOut(**{k: r.latency_ms.get(k, 0.0)
                                 for k in ("retrieve", "rerank", "generate", "total")}),
        invalid_citations_removed=r.invalid_citations_removed)


def create_app(pipeline: PipelineLike | None = None) -> FastAPI:
    """`pipeline` is injected in tests. In production it is built at startup from the configs."""
    metrics = Metrics()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pipeline = pipeline or RagPipeline.from_config(load_config())
        metrics.index_chunks.set(app.state.pipeline.index_size())
        log.info("pipeline ready: %d chunks, model %s", app.state.pipeline.index_size(),
                 app.state.pipeline.model_name())
        yield

    app = FastAPI(title="CiteSEC", version="0.1.0", lifespan=lifespan,
                  description="Citation-grounded question answering over SEC filings.")
    app.state.metrics = metrics

    @app.post("/query", response_model=QueryResponse)
    def query(req: QueryRequest, request: Request) -> QueryResponse:
        pipe: PipelineLike = request.app.state.pipeline
        f = req.filters
        try:
            result = pipe.answer(req.question, req.top_k, f.companies if f else None,
                                 f.years if f else None)
        except httpx.HTTPError as exc:  # Ollama down or timed out
            metrics.errors.labels("generator_unavailable").inc()
            raise HTTPException(503, "The local language model is not reachable. "
                                     "Start Ollama and try again.") from exc
        except Exception:
            metrics.errors.labels("internal").inc()
            log.exception("query failed")
            raise
        metrics.observe(result)
        return to_response(result, pipe.model_name())

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        pipe: PipelineLike = request.app.state.pipeline
        up = pipe.llm_up()
        return HealthResponse(status="ok" if up else "degraded", index_chunks=pipe.index_size(),
                              model=pipe.model_name(), llm_reachable=up)

    @app.get("/metrics")
    def prometheus() -> Response:
        return Response(metrics.render(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
