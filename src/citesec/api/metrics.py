"""Prometheus metrics for the API. One registry per app instance so tests do not collide."""
from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

from citesec.pipeline import AnswerResult

BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64)
STAGES = ("retrieve", "rerank", "generate", "total")


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.stage = Histogram("citesec_stage_latency_seconds", "Latency per pipeline stage",
                               ["stage"], buckets=BUCKETS, registry=self.registry)
        self.queries = Counter("citesec_queries_total", "Queries by outcome", ["outcome"],
                               registry=self.registry)
        self.abstentions = Counter("citesec_abstentions_total", "Abstentions by reason",
                                   ["reason"], registry=self.registry)
        self.invalid_citations = Counter("citesec_invalid_citations_total",
                                         "Citations removed because they named no retrieved block",
                                         registry=self.registry)
        self.errors = Counter("citesec_errors_total", "Failed queries by kind", ["kind"],
                              registry=self.registry)
        self.index_chunks = Gauge("citesec_index_chunks", "Chunks in the loaded index",
                                  registry=self.registry)

    def observe(self, result: AnswerResult) -> None:
        for stage in STAGES:
            if stage in result.latency_ms:
                self.stage.labels(stage).observe(result.latency_ms[stage] / 1000)
        if result.abstained:
            self.abstentions.labels(result.abstain_reason or "unknown").inc()
            self.queries.labels(f"abstained_{result.abstain_reason}").inc()
        else:
            self.queries.labels("answered").inc()
        if result.invalid_citations_removed:
            self.invalid_citations.inc(result.invalid_citations_removed)

    def render(self) -> bytes:
        return generate_latest(self.registry)
