from pathlib import Path

import httpx
import pytest

from citesec import ui

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")

ANSWER = {
    "answer": "Net sales were $5,577 million [ACME_2022_10K, p.51]. Final answer: $5,577 million",
    "citations": [{"doc_name": "ACME_2022_10K", "page": 51, "section": None, "score": 0.97,
                   "snippet": "Net sales were $5,577 million", "supported": True}],
    "abstained": False, "abstain_reason": None, "confidence": 0.97, "model": "m",
    "latency_ms": {"retrieve": 60.0, "rerank": 250.0, "generate": 2000.0, "total": 2330.0},
    "evidence": [{"doc_name": "ACME_2022_10K", "page": 51, "section": None, "score": 0.97,
                  "snippet": "Net sales were $5,577 million", "cited": True}],
    "invalid_citations_removed": 1,
}
ABSTAINED = {**ANSWER, "answer": "INSUFFICIENT_EVIDENCE", "abstained": True, "citations": [],
             "abstain_reason": "gate", "confidence": 0.01, "invalid_citations_removed": 0}


def test_format_answer_badges_and_math() -> None:
    out = ui.format_answer("Sales [ACME_2022_10K, p.51] and capex [EDGAR_CVX_10K_2023-12-31, "
                           "Item 7]. \\[ \\text{x} = 2 \\] then \\( y \\)")
    assert ":blue-badge[ACME_2022_10K · p.51]" in out
    assert ":blue-badge[EDGAR_CVX_10K_2023-12-31 · Item 7]" in out
    assert "$$ \\text{x} = 2 $$" in out and "$y$" in out
    assert ui.format_answer("no citations [Note 5, p. 3]") == "no citations [Note 5, p. 3]"


def test_format_answer_escapes_currency_so_it_is_not_read_as_math() -> None:
    out = ui.format_answer("Up from $3,800,000,000 to $4,200,000,000 [ACME_2022_10K, p.2]")
    assert out.startswith("Up from \\$3,800,000,000 to \\$4,200,000,000")
    assert ":blue-badge[ACME_2022_10K · p.2]" in out
    math = ui.format_answer("\\[ x = \\$5 + 3 \\] costs $7")
    assert "\\\\$" not in math  # an already escaped \$ is not escaped twice
    assert "$$ x = \\$5 + 3 $$" in math and "costs \\$7" in math


def test_build_payload() -> None:
    assert ui.build_payload(" q? ", "", "", 3) == {"question": "q?", "top_k": 3}
    p = ui.build_payload("q", "Amcor, 3M", "FY2022 and 2023", 5)
    assert p["filters"] == {"companies": ["Amcor", "3M"], "years": [2022, 2023]}
    assert ui.build_payload("q", "", "2021", 5)["filters"] == {"companies": None, "years": [2021]}


def test_latency_rows_and_labels() -> None:
    assert ui.latency_rows(ANSWER["latency_ms"]) == [("retrieve", 0.06), ("rerank", 0.25),
                                                     ("generate", 2.0)]
    assert ui.citation_label("D", 4, None) == "D · p.4"
    assert ui.citation_label("D", None, "Item 1A") == "D · Item 1A"


def test_api_query_success_and_readable_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "post", lambda *a, **k: httpx.Response(200, json={"ok": 1}))
    assert ui.api_query("http://x/", {"question": "q"}) == {"ok": 1}

    monkeypatch.setattr(httpx, "post",
                        lambda *a, **k: httpx.Response(503, json={"detail": "Start Ollama"}))
    with pytest.raises(RuntimeError, match="Start Ollama"):
        ui.api_query("http://x", {})

    monkeypatch.setattr(httpx, "post", lambda *a, **k: httpx.Response(500, text="boom"))
    with pytest.raises(RuntimeError, match="HTTP 500"):
        ui.api_query("http://x", {})

    def refuse(*a, **k):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "post", refuse)
    with pytest.raises(RuntimeError, match="Cannot reach the API"):
        ui.api_query("http://x", {})


def test_api_health(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "get", lambda *a, **k: httpx.Response(200, json={"status": "ok"}))
    assert ui.api_health("http://x") == {"status": "ok"}

    def refuse(*a, **k):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "get", refuse)
    assert ui.api_health("http://x") is None


# ---- the app itself, run headless with the API mocked ----

def run_app(monkeypatch: pytest.MonkeyPatch, result: dict | Exception):
    from streamlit.testing.v1 import AppTest

    health = {"status": "ok", "index_chunks": 128020, "model": "m", "llm_reachable": True}
    monkeypatch.setattr(ui, "api_health", lambda base: health)

    def fake_query(base, payload, timeout=180.0):
        fake_query.payload = payload
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(ui, "api_query", fake_query)
    at = AppTest.from_file(APP, default_timeout=20).run()
    return at, fake_query


def click(at, label: str):
    return [b for b in at.button if b.label == label][0].click().run()


def test_app_answer_view(monkeypatch: pytest.MonkeyPatch) -> None:
    at, fake = run_app(monkeypatch, ANSWER)
    assert not at.exception and at.title[0].value == "CiteSEC"
    assert [b for b in at.button if b.label == "Ask"][0].disabled  # empty question
    at.text_area[0].set_value("What were net sales?").run()
    at = click(at, "Ask")
    assert not at.exception and not at.warning and not at.error
    assert any(":blue-badge[ACME_2022_10K · p.51]" in m.value for m in at.markdown)
    assert {m.label for m in at.metric} == {"Confidence", "Retrieve", "Rerank", "Generate", "Total"}
    assert any("1 citation(s) removed" in c.value for c in at.caption)
    assert fake.payload == {"question": "What were net sales?", "top_k": 5}


def test_app_abstention_banner(monkeypatch: pytest.MonkeyPatch) -> None:
    at, _ = run_app(monkeypatch, ABSTAINED)
    at.text_area[0].set_value("What was Tesla revenue?").run()
    at = click(at, "Ask")
    assert at.warning and "Not enough evidence" in at.warning[0].value
    assert not any("Answer" == s.value for s in at.subheader)


def test_app_shows_api_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    at, _ = run_app(monkeypatch, RuntimeError("Cannot reach the API at x"))
    at.text_area[0].set_value("What were net sales?").run()
    at = click(at, "Ask")
    assert at.error and "Cannot reach the API" in at.error[0].value


def test_app_example_buttons_fill_the_question(monkeypatch: pytest.MonkeyPatch) -> None:
    at, _ = run_app(monkeypatch, ANSWER)
    first = [b for b in at.button if b.label.startswith(ui.EXAMPLES[0][:30])][0]
    at = first.click().run()
    assert at.text_area[0].value == ui.EXAMPLES[0]
