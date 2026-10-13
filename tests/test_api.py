import httpx
import pytest
from fastapi.testclient import TestClient

from citesec.api.main import create_app
from citesec.pipeline import AnswerResult, Citation, EvidenceBlock

LAT = {"retrieve": 60.0, "rerank": 250.0, "generate": 2000.0, "total": 2330.0}


class FakePipeline:
    def __init__(self, result: AnswerResult | None = None, error: Exception | None = None) -> None:
        self.result, self.error, self.calls = result, error, []

    def answer(self, question, top_k=5, companies=None, years=None):
        self.calls.append((question, top_k, companies, years))
        if self.error:
            raise self.error
        return self.result

    def index_size(self) -> int:
        return 1234

    def model_name(self) -> str:
        return "fake-model"

    def llm_up(self) -> bool:
        return True


def answered() -> AnswerResult:
    ev = [EvidenceBlock("ACME_2022_10K", 51, None, "Net sales were $5,577 million", 0.97, True),
          EvidenceBlock("ACME_2022_10K", 7, None, "Other text", 0.4, False)]
    cit = [Citation("ACME_2022_10K", 51, None, "Net sales were $5,577 million", 0.97, True)]
    return AnswerResult("Net sales were $5,577 million [ACME_2022_10K, p.51].", False, None, 0.97,
                        cit, ev, dict(LAT), invalid_citations_removed=1,
                        tokens={"prompt": 10, "completion": 5})


def abstained(reason: str) -> AnswerResult:
    return AnswerResult("INSUFFICIENT_EVIDENCE", True, reason, 0.01, [], [],
                        {**LAT, "generate": 0.0})


def client(pipe: FakePipeline) -> TestClient:
    return TestClient(create_app(pipe))


def test_query_response_schema() -> None:
    pipe = FakePipeline(answered())
    with client(pipe) as c:
        r = c.post("/query", json={"question": "What were net sales?"})
    assert r.status_code == 200
    body = r.json()
    assert body["abstained"] is False and body["confidence"] == 0.97
    assert body["model"] == "fake-model"
    assert body["citations"][0] == {"doc_name": "ACME_2022_10K", "page": 51, "section": None,
                                    "snippet": "Net sales were $5,577 million", "score": 0.97,
                                    "supported": True}
    assert body["latency_ms"] == LAT and body["invalid_citations_removed"] == 1
    assert [e["cited"] for e in body["evidence"]] == [True, False]
    assert pipe.calls == [("What were net sales?", 5, None, None)]


def test_filters_and_top_k_are_passed_through() -> None:
    pipe = FakePipeline(answered())
    with client(pipe) as c:
        c.post("/query", json={"question": "net sales?", "top_k": 3,
                               "filters": {"companies": ["Acme"], "years": [2022]}})
    assert pipe.calls == [("net sales?", 3, ["Acme"], [2022])]


@pytest.mark.parametrize("payload", [
    {"question": "ab"}, {"question": "x" * 1001}, {"question": "valid question", "top_k": 0},
    {"question": "valid question", "top_k": 11}, {}, {"question": "valid question", "filters": 3},
])
def test_invalid_requests_are_rejected(payload: dict) -> None:
    pipe = FakePipeline(answered())
    with client(pipe) as c:
        assert c.post("/query", json=payload).status_code == 422
    assert pipe.calls == []


def test_abstention_response() -> None:
    with client(FakePipeline(abstained("gate"))) as c:
        body = c.post("/query", json={"question": "What was Tesla revenue?"}).json()
    assert body["abstained"] and body["abstain_reason"] == "gate" and body["citations"] == []


def test_generator_down_is_a_503_not_a_500() -> None:
    pipe = FakePipeline(error=httpx.ConnectError("refused"))
    with client(pipe) as c:
        r = c.post("/query", json={"question": "net sales?"})
        assert r.status_code == 503 and "Ollama" in r.json()["detail"]
        assert 'citesec_errors_total{kind="generator_unavailable"} 1.0' in c.get("/metrics").text


def test_health() -> None:
    with client(FakePipeline(answered())) as c:
        assert c.get("/health").json() == {"status": "ok", "index_chunks": 1234,
                                           "model": "fake-model", "llm_reachable": True}


def test_prometheus_metrics_after_queries() -> None:
    pipe = FakePipeline(answered())
    with client(pipe) as c:
        c.post("/query", json={"question": "net sales?"})
        pipe.result = abstained("gate")
        c.post("/query", json={"question": "tesla revenue?"})
        pipe.result = abstained("model")
        c.post("/query", json={"question": "other thing?"})
        text = c.get("/metrics").text
    assert 'citesec_stage_latency_seconds_bucket{le="4.0",stage="generate"}' in text
    assert 'citesec_stage_latency_seconds_count{stage="total"} 3.0' in text
    assert 'citesec_queries_total{outcome="answered"} 1.0' in text
    assert 'citesec_abstentions_total{reason="gate"} 1.0' in text
    assert 'citesec_abstentions_total{reason="model"} 1.0' in text
    assert "citesec_invalid_citations_total 1.0" in text
    assert "citesec_index_chunks 1234.0" in text


def test_openapi_docs_available() -> None:
    with client(FakePipeline(answered())) as c:
        assert c.get("/docs").status_code == 200
        spec = c.get("/openapi.json").json()
    assert {"/query", "/health", "/metrics"} <= set(spec["paths"])
    assert "QueryResponse" in spec["components"]["schemas"]
