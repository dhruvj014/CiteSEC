import pytest

from citesec.eval.answer_metrics import abstained_final, reasoning_group, system_metrics


def row(i, kind, system="S2", qtype="novel-generated", reasoning=None, abstained=False,
        n_cit=2, n_valid=2, n_sup=1, removed=0, total=100.0):
    return {"id": i, "kind": kind, "system": system, "question_type": qtype,
            "question_reasoning": reasoning, "abstained": abstained, "n_citations": n_cit,
            "n_valid": n_valid, "n_supported": n_sup, "n_invalid_removed": removed,
            "latency_ms": {"generate": total, "total": total}}


def j(v, faithful=None, perr=False):
    return {"verdict": v, "faithful": faithful, "parse_error": perr}


def test_reasoning_group() -> None:
    assert reasoning_group("Numerical reasoning OR Logical reasoning") == "numerical reasoning"
    assert reasoning_group("Information extraction") == "information extraction"
    label = "Logical reasoning (based on numerical reasoning)"
    assert reasoning_group(label) == "logical reasoning"
    assert reasoning_group(None) == "unlabeled"


def test_abstained_final_uses_flag_or_judge() -> None:
    assert abstained_final({"abstained": True}, None)
    assert abstained_final({"abstained": False}, {"verdict": "REFUSED"})
    assert not abstained_final({"abstained": False}, {"verdict": "CORRECT"})


def test_accuracy_refusal_and_abstention_math() -> None:
    rows = [row("a1", "answerable"), row("a2", "answerable"), row("a3", "answerable"),
            row("a4", "answerable", abstained=True),
            row("u1", "unanswerable", abstained=True), row("u2", "unanswerable"),
            row("u3", "unanswerable", abstained=True)]
    judg = {"a1": j("CORRECT", True), "a2": j("INCORRECT", False), "a3": j("CORRECT", True),
            "a4": j("REFUSED"), "u1": j("REFUSED"), "u2": j("INCORRECT", False),
            "u3": j("REFUSED")}
    m = system_metrics(rows, judg)
    assert m["accuracy"] == 0.5  # 2 of 4 answerable
    assert m["false_refusal_rate"] == 0.25
    assert m["abstention_recall_unanswerable"] == pytest.approx(0.6667, abs=1e-4)
    assert m["abstention_precision"] == pytest.approx(0.6667, abs=1e-4)  # 2 of 3 abstentions
    assert m["answered_unanswerable_rate"] == pytest.approx(0.3333, abs=1e-4)
    assert m["faithfulness"] == pytest.approx(0.6667, abs=1e-4) and m["n_faithfulness_judged"] == 3


def test_accuracy_when_answered_and_refusal_causes() -> None:
    def r(i, gold, abstained=False):
        d = row(i, "answerable", abstained=abstained)
        d["gold_in_context"] = gold
        return d

    rows = [r("a1", True), r("a2", True), r("a3", False, True), r("a4", False, True),
            r("a5", True, True)]
    judg = {"a1": j("CORRECT"), "a2": j("INCORRECT"), "a3": j("REFUSED"), "a4": j("REFUSED"),
            "a5": j("REFUSED")}
    m = system_metrics(rows, judg)
    assert m["accuracy"] == pytest.approx(0.2) and m["accuracy_when_answered"] == 0.5
    assert m["n_answered"] == 2
    assert m["false_refusals_by_cause"] == {"gold_page_missing_from_context": 2,
                                            "gold_page_in_context": 1}
    assert m["gold_page_in_context"] == pytest.approx(0.6)
    assert m["outcomes_by_retrieval"] == {
        "gold_page_in_context": {"n": 3, "correct": 1, "incorrect": 1, "refused": 1},
        "gold_page_missing": {"n": 2, "correct": 0, "incorrect": 0, "refused": 2}}


def test_unanswerable_abstention_source() -> None:
    gate = row("u1", "unanswerable", abstained=True)
    gate["abstain_reason"] = "gate"
    tok = row("u2", "unanswerable", abstained=True)
    tok["abstain_reason"] = "llm"
    only_judge = row("u3", "unanswerable")
    judg = {"u1": j("REFUSED"), "u2": j("REFUSED"), "u3": j("REFUSED")}
    m = system_metrics([gate, tok, only_judge], judg)
    assert m["unanswerable_abstention_source"] == {"gate": 1, "model_token": 1, "judge_only": 1}


def test_judge_refusal_counts_as_abstention_without_flag() -> None:
    rows = [row("u1", "unanswerable"), row("a1", "answerable")]
    m = system_metrics(rows, {"u1": j("REFUSED"), "a1": j("CORRECT")})
    assert m["abstention_recall_unanswerable"] == 1.0 and m["abstention_precision"] == 1.0


def test_citation_metrics_and_cited_answer_rate() -> None:
    rows = [row("a1", "answerable", n_cit=4, n_valid=3, n_sup=2, removed=1),
            row("a2", "answerable", n_cit=2, n_valid=0, n_sup=0, removed=2),
            row("a3", "answerable", n_cit=0, n_valid=0, n_sup=0)]
    m = system_metrics(rows, {"a1": j("CORRECT"), "a2": j("INCORRECT"), "a3": j("CORRECT")})
    c = m["citations"]
    assert c["total"] == 6 and c["validity"] == 0.5
    assert c["support"] == pytest.approx(0.6667, abs=1e-4)
    assert c["invalid_removed"] == 3
    assert c["cited_answer_rate"] == pytest.approx(0.3333, abs=1e-4)  # only a1 has a valid citation


def test_breakdowns_and_latency() -> None:
    rows = [row("a1", "answerable", qtype="metrics-generated", reasoning="Numerical reasoning",
                total=100.0),
            row("a2", "answerable", qtype="metrics-generated", reasoning="Numerical reasoning",
                total=300.0),
            row("a3", "answerable", qtype="novel-generated", total=200.0)]
    m = system_metrics(rows, {"a1": j("CORRECT"), "a2": j("INCORRECT"), "a3": j("CORRECT")})
    assert m["accuracy_by_question_type"]["metrics-generated"] == {"n": 2, "accuracy": 0.5}
    assert m["accuracy_by_question_type"]["novel-generated"]["accuracy"] == 1.0
    assert m["accuracy_by_reasoning"]["numerical reasoning"]["n"] == 2
    assert m["latency_ms"]["total"]["p50"] == 200.0


def test_empty_denominators_are_none_not_zero() -> None:
    m = system_metrics([row("a1", "answerable")], {"a1": j("CORRECT")})
    assert m["abstention_recall_unanswerable"] is None and m["faithfulness"] is None
