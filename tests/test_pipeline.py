from tests.fakes import FakeLLM

from citesec.pipeline import RagPipeline
from citesec.retrieval.hybrid import Retriever

# c19 ("target passage about revenue") is doc D1 page index 19, so its citation tag is p.20


def make(retriever: Retriever, text: str, threshold: float = 0.5,
         use_filter: bool = False) -> tuple[RagPipeline, FakeLLM]:
    llm = FakeLLM(text)
    return RagPipeline(retriever, llm, threshold, use_filter=use_filter), llm  # type: ignore[arg-type]


def test_grounded_answer_with_citation_and_evidence(retriever: Retriever) -> None:
    pipe, llm = make(retriever, "The target passage is about revenue [D1, p.20].\n"
                                "Final answer: revenue")
    r = pipe.answer("what is the target passage about revenue")
    assert not r.abstained and r.abstain_reason is None
    assert r.confidence == 0.9
    assert len(r.citations) == 1
    c = r.citations[0]
    assert (c.doc_name, c.page, c.section) == ("D1", 20, None)
    assert "target passage" in c.snippet and c.supported is True
    assert len(r.evidence) == 5 and r.evidence[0].cited and not r.evidence[1].cited
    assert r.invalid_citations_removed == 0 and "[D1, p.20]" in r.answer
    assert r.tokens == {"prompt": 100, "completion": 20}
    # the model saw numbered blocks with citation tags and the question last
    user = llm.calls[0][1]["content"]
    assert "Block 1 [D1, p.20]" in user and user.rstrip().endswith(
        "Question: what is the target passage about revenue")


def test_invalid_citations_are_stripped_and_counted(retriever: Retriever) -> None:
    pipe, _ = make(retriever, "Revenue [D1, p.20] and made up [ZZZ_9, p.1].")
    r = pipe.answer("target revenue")
    assert r.invalid_citations_removed == 1
    assert "ZZZ_9" not in r.answer and len(r.citations) == 1


def test_model_refusal_is_reported(retriever: Retriever) -> None:
    pipe, _ = make(retriever, "INSUFFICIENT_EVIDENCE")
    r = pipe.answer("target revenue")
    assert r.abstained and r.abstain_reason == "model" and r.citations == []
    assert r.answer == "INSUFFICIENT_EVIDENCE" and len(r.evidence) == 5  # evidence still shown


def test_gate_skips_the_llm(retriever: Retriever) -> None:
    pipe, llm = make(retriever, "should never be used")
    # the Acme filter removes the only high-scoring chunk, so the best score is 0.01 < 0.5
    r = pipe.answer("target revenue", companies=["acme"])
    assert r.abstained and r.abstain_reason == "gate" and llm.calls == []
    assert r.latency_ms["generate"] == 0.0 and r.confidence == 0.01


def test_unknown_company_filter_abstains_with_zero_confidence(retriever: Retriever) -> None:
    pipe, llm = make(retriever, "x")
    r = pipe.answer("target revenue", companies=["Nobody Inc"])
    assert r.abstained and r.abstain_reason == "gate" and r.confidence == 0.0
    assert r.evidence == [] and llm.calls == []


def test_year_filter_and_top_k(retriever: Retriever) -> None:
    pipe, _ = make(retriever, "Revenue [D1, p.20].", threshold=0.0)
    r = pipe.answer("target revenue", top_k=2, years=[2021])
    assert len(r.evidence) == 2
    assert all(e.doc_name in ("D0", "D1") for e in r.evidence)


def test_latency_breakdown_is_consistent(retriever: Retriever) -> None:
    pipe, _ = make(retriever, "Revenue [D1, p.20].")
    lat = pipe.answer("target revenue").latency_ms
    assert set(lat) == {"retrieve", "rerank", "generate", "total"}
    assert lat["generate"] == 1000.0  # total_ms minus load_ms from the fake
    assert lat["total"] >= lat["retrieve"] + lat["rerank"]


def test_helpers(retriever: Retriever) -> None:
    pipe, _ = make(retriever, "x")
    assert pipe.index_size() == 20 and pipe.model_name() == "fake-llm" and pipe.llm_up()
