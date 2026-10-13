import pytest

from citesec.eval.judge_validation import agreement, cohen_kappa, sample_items


def rows(system, n, kind="answerable"):
    return [{"id": f"{system}_{kind}_{i}", "kind": kind, "question": "q", "gold_answer": "g",
             "answer": "a"} for i in range(n)]


def test_sample_is_seeded_stratified_and_unique() -> None:
    data = {"S0": rows("S0", 30), "S1": rows("S1", 30) + rows("S1", 10, "unanswerable"),
            "S2": rows("S2", 30) + rows("S2", 10, "unanswerable")}
    a, b = sample_items(data, 42), sample_items(data, 42)
    assert a == b and len(a) == 40
    assert sum(i["system"] == "S2" and i["kind"] == "answerable" for i in a) == 14
    assert sum(i["system"] == "S1" and i["kind"] == "answerable" for i in a) == 10
    assert sum(i["system"] == "S0" for i in a) == 8
    assert sum(i["kind"] == "unanswerable" for i in a) == 8
    assert len({(i["system"], i["id"]) for i in a}) == 40
    assert all(i["label"] is None for i in a)
    assert sample_items(data, 7) != a


def test_agreement_and_confusion() -> None:
    labels = ["CORRECT", "CORRECT", "INCORRECT", "REFUSED", "INCORRECT"]
    verdicts = ["CORRECT", "INCORRECT", "INCORRECT", "REFUSED", "CORRECT"]
    r = agreement(labels, verdicts)
    assert r["n"] == 5 and r["agreement"] == 0.6
    assert r["per_label"]["CORRECT"] == {"n": 2, "judge_agrees": 1, "recall": 0.5}
    assert r["per_label"]["REFUSED"]["recall"] == 1.0
    assert r["confusion"]["label=CORRECT judge=INCORRECT"] == 1


def test_kappa_extremes() -> None:
    same = ["CORRECT", "INCORRECT", "REFUSED", "CORRECT"]
    assert cohen_kappa(same, same) == pytest.approx(1.0)
    assert cohen_kappa(["CORRECT"] * 4, ["CORRECT"] * 4) == 1.0  # no variation, perfect agreement
    flipped = ["INCORRECT", "CORRECT", "CORRECT", "INCORRECT"]
    assert cohen_kappa(["CORRECT", "INCORRECT", "INCORRECT", "CORRECT"], flipped) < 0
