import json
from pathlib import Path

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
