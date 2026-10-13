import json
from pathlib import Path

import pytest

from citesec.eval.report import NOT_MEASURED, pct, render, table

METRICS = {"hit@1": 0.1, "hit@3": 0.2, "hit@5": 0.3, "hit@10": 0.4, "recall@5": 0.25, "mrr": 0.2,
           "doc_hit@5": 0.8, "n": 10}


def fake_results(split: str) -> dict:
    cfg = {"description": "x", "chunker": "structured", "mode": "hybrid", "rerank": True,
           "filter": False, "metrics": METRICS,
           "by_question_type": {"novel-generated": METRICS},
           "latency_ms": {"total": {"p50": 12.4, "p95": 40.0, "mean": 15.0}}}
    return {"split": split, "n_questions": 10, "seed": 42, "gpu": "TestGPU",
            "retrieval": {"top_n": 50, "rrf_k": 60},
            "models": {"embed": "e", "reranker": "r"}, "repeat_run_identical": True,
            "configs": {"F": cfg}}


def fake_answers(split: str) -> dict:
    sysm = {"n_answerable": 10, "n_unanswerable": 4, "accuracy": 0.4, "wrong_answer_rate": 0.3,
            "accuracy_when_answered": 0.6,
            "outcomes_by_retrieval": {
                "gold_page_in_context": {"n": 6, "correct": 4, "incorrect": 1, "refused": 1},
                "gold_page_missing": {"n": 4, "correct": 0, "incorrect": 1, "refused": 3}},
            "unanswerable_abstention_source": {"gate": 2, "model_token": 1},
            "false_refusal_rate": 0.2, "faithfulness": 0.8, "abstention_recall_unanswerable": 0.75,
            "abstention_precision": 0.6, "answered_unanswerable_rate": 0.25,
            "accuracy_by_question_type": {"novel-generated": {"n": 4, "accuracy": 0.5}},
            "accuracy_by_reasoning": {"numerical reasoning": {"n": 5, "accuracy": 0.2}},
            "citations": {"total": 51, "validity": 0.9, "support": 0.7, "cited_answer_rate": 0.85},
            "latency_ms": {"retrieve": {"p50": 50, "p95": 90},
                           "generate": {"p50": 9000, "p95": 20000},
                           "total": {"p50": 9500.0, "p95": 21000.0}}}
    return {"split": split, "judge": {"model": "j", "kind": "local (Ollama)"}, "generator": "g",
            "systems": {"S2": sysm, "S4": dict(sysm, accuracy=0.5)},
            "gate": {"threshold": 0.0414, "criterion": "max recall with false refusal <= 0.1",
                     "auc": 0.67, "precision": 0.7, "recall": 0.37, "false_refusal_rate": 0.06,
                     "f1_optimal": {"threshold": 0.91, "false_refusal_rate": 0.64}}}


def test_answer_sections_render(tmp_path: Path) -> None:
    rd = tmp_path / "results"
    rd.mkdir()
    (rd / "answers_metrics_test.json").write_text(json.dumps(fake_answers("test")))
    def jv(n, agree, kappa):
        pl = {k: {"n": 1, "judge_agrees": 1, "recall": 0.8} for k in
              ("CORRECT", "INCORRECT", "REFUSED")}
        return {"n": n, "agreement": agree, "cohen_kappa": kappa, "per_label": pl,
                "labeled_by": "tester, blind", "judge_model": "j", "judge_kind": "local (Ollama)"}

    (rd / "generation_determinism.json").write_text(json.dumps({
        "n": 5, "temperature": 0.0, "n_fresh_pairs_identical": 5, "n_identical_to_cached": 2}))
    (rd / "generation_variance_dev.json").write_text(json.dumps({
        "text_identical": 0.49, "verdict_agreement": 0.88, "n_unanswerable": 19,
        "unanswerable_abstained_run1": 19, "unanswerable_abstained_run2": 19,
        "answerable": {"accuracy_run1": 0.26, "accuracy_run2": 0.28, "refusal_rate_run1": 0.44,
                       "refusal_rate_run2": 0.46}}))
    (rd / "judge_validation_r1_v1.json").write_text(json.dumps(jv(40, 0.75, 0.63)))
    (rd / "judge_validation_r1_v3.json").write_text(json.dumps(jv(40, 0.90, 0.84)))
    (rd / "judge_validation_r3_v3.json").write_text(json.dumps(jv(20, 0.95, 0.90)))
    out = render(rd)
    row = "| S4 | S3 + confidence gate (full system) | 50.0 | 60.0 | 30.0 | 20.0 | 80.0 | 90.0 |"
    assert row in out
    assert "| Gold page in the 5 context blocks | 6 | 4 | 1 | 1 |" in out
    assert "refused by the confidence gate: 2" in out and "*Faithful is judged" in out
    assert "9.5 / 21.0" in out
    assert "| S2 | 75.0 | 60.0 | 25.0 | 20.0 |" in out
    assert "4 unanswerable questions (out-of-corpus" in out
    s0_first = fake_answers("test")
    s0_first["systems"] = {"S0": dict(s0_first["systems"]["S2"], n_unanswerable=0),
                           "S2": s0_first["systems"]["S2"]}
    (rd / "answers_metrics_test.json").write_text(json.dumps(s0_first))
    assert "4 unanswerable questions" in render(rd)  # S0 listed first must not zero the count
    (rd / "answers_metrics_test.json").write_text(json.dumps(fake_answers("test")))
    assert "would refuse 64.0% of answerable" in out and "0.0414" in out
    assert "Labels were written by tester" in out and "no human reviewer" not in out
    assert "| v1 | 1 | first look | 40 | 75.0 | 0.63 |" in out
    assert "| v3 | 3 | yes | 20 | 95.0 | 0.90 |" in out
    assert "pooled over 60 labeled answers: 91.7% agreement" in out  # (36 + 19) / 60
    assert "reasoning: numerical reasoning | 5 | 20.0" in out
    assert "only 2 of 5 matched the earlier cached run" in out
    assert "accuracy 26.0% then 28.0%" in out
    assert render(rd) == out


def fake_latency() -> dict:
    stat = lambda a, b: {"p50": a, "p95": b, "mean": a}  # noqa: E731
    stages = {"retrieve": stat(15, 28), "rerank": stat(208, 247), "generate": stat(3514, 12819),
              "total": stat(4051, 13323)}
    return {
        "gpu": "TestGPU", "generator": "gen", "n_queries": 100, "query_set": "test questions",
        "settings": {"top_n": 50, "rerank_n": 50, "n_context": 5, "num_predict": 700},
        "end_to_end": {"n": 100, "overall": stages,
                       "by_outcome": {"answered": {"n": 61, **stages}}},
        "optimization_rerank_candidates": {"dev_questions": 50, "by_rerank_n": {
            "50": {"rerank_ms": stat(213, 250), "retrieval_total_ms": stat(224, 263),
                   "dev_quality_percent": {"hit@1": 20.0, "hit@5": 40.0, "mrr": 30.7,
                                           "doc_hit@5": 90.0}},
            "30": {"rerank_ms": stat(129, 154), "retrieval_total_ms": stat(138, 165),
                   "dev_quality_percent": {"hit@1": 18.0, "hit@5": 40.0, "mrr": 29.7,
                                           "doc_hit@5": 88.0}}}},
        "optimization_query_embedding_cache": {
            "embed_query_ms_cold": stat(18, 40), "embed_query_ms_warm": stat(0, 0),
            "retrieval_total_ms_cold": stat(246, 300), "retrieval_total_ms_warm": stat(220, 263)},
    }


def test_serving_section_and_readme_block_match_the_report(tmp_path: Path) -> None:
    from citesec.eval.report import readme_block

    rd = tmp_path / "results"
    rd.mkdir()
    (rd / "latency.json").write_text(json.dumps(fake_latency()))
    (rd / "retrieval_test.json").write_text(json.dumps(fake_results("test") | {"configs": {
        "A": {**fake_results("test")["configs"]["F"], "chunker": "naive", "rerank": False,
              "metrics": {**METRICS, "hit@5": 0.24}},
        "F": fake_results("test")["configs"]["F"]}}))
    (rd / "answers_metrics_test.json").write_text(json.dumps(fake_answers("test")))
    out = render(rd)
    assert "| answered | 61 | 15 / 28 | 208 / 247 | 3514 / 12819 | 4051 / 13323 |" in out
    assert "| 30 | 129 / 154 | 138 / 165 | 18.0 | 40.0 | 29.7 | 88.0 |" in out
    assert "| warm (cached) | 0 / 0 | 220 / 263 |" in out
    block = readme_block(rd)
    # every headline number in the README block appears in the full report
    for needle in ("24.0", "30.0", "50.0", "9.5 / 21.0"):
        assert needle in block and needle in out
    assert "p50 4.1 s, p95 13.3 s" in block and "TestGPU" in block
    assert "refused quickly" in block


def test_update_readme_replaces_only_the_marked_block() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "make_report", Path(__file__).resolve().parents[1] / "scripts" / "make_report.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    text = "intro\n<!-- RESULTS:START -->\nold\n<!-- RESULTS:END -->\noutro"
    new = mod.update_readme(text, "NEW TABLE")
    assert new == "intro\n<!-- RESULTS:START -->\nNEW TABLE\n<!-- RESULTS:END -->\noutro"
    assert mod.update_readme(new, "NEW TABLE") == new  # idempotent
    with pytest.raises(SystemExit):
        mod.update_readme("no markers", "x")


def test_pct_and_table() -> None:
    assert pct(0.1234) == "12.3"
    assert table(["a", "b"], [["1", "2"]]) == "| a | b |\n|---|---|\n| 1 | 2 |"


def test_missing_results_say_not_measured(tmp_path: Path) -> None:
    out = render(tmp_path / "results")
    assert out.count(NOT_MEASURED) >= 5 and "Do not edit by hand" in out


def test_render_is_deterministic_and_complete(tmp_path: Path) -> None:
    rd = tmp_path / "results"
    rd.mkdir()
    for split in ("dev", "test"):
        (rd / f"retrieval_{split}.json").write_text(json.dumps(fake_results(split)))
    a, b = render(rd), render(rd)
    assert a == b
    row = "| F | structured | hybrid | yes | no | 10.0 | 20.0 | 30.0 | 40.0 | 25.0 | 20.0 | 80.0 |"
    assert row in a
    assert "12 / 40" in a and "TestGPU" in a and "novel-generated" in a
