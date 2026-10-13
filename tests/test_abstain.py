import pytest

from citesec.eval.unanswerable import build, split_ids
from citesec.generation.abstain import auc, calibrate, llm_abstained, prf, should_abstain


def test_should_abstain() -> None:
    assert should_abstain([], 0.5)
    assert should_abstain([0.2, 0.9], 0.5)  # only the top score matters
    assert not should_abstain([0.7, 0.1], 0.5)
    assert not should_abstain([0.5], 0.5)  # equal is not below


def test_llm_abstained() -> None:
    assert llm_abstained("INSUFFICIENT_EVIDENCE")
    assert llm_abstained("  INSUFFICIENT_EVIDENCE\n")
    assert llm_abstained("The blocks are about Intel.\n\nFinal answer: INSUFFICIENT_EVIDENCE")
    assert not llm_abstained("The answer is 5.")
    assert not llm_abstained("We have insufficient evidence")  # prose is not the token


def test_prf() -> None:
    assert prf(8, 2, 2) == pytest.approx((0.8, 0.8, 0.8))
    assert prf(0, 0, 0) == (0.0, 0.0, 0.0)


def test_calibrate_perfectly_separable() -> None:
    cal = calibrate([0.8, 0.9, 0.95], [0.1, 0.2, 0.3])
    assert cal["f1"] == 1.0 and cal["precision"] == 1.0 and cal["recall"] == 1.0
    assert cal["false_refusal_rate"] == 0.0
    assert 0.3 < cal["threshold"] < 0.8 and cal["auc"] == 1.0


def test_calibrate_overlapping_picks_best_f1_and_lowest_threshold_on_tie() -> None:
    answerable = [0.9, 0.8, 0.6, 0.3]
    unanswerable = [0.7, 0.4, 0.2, 0.1]
    cal = calibrate(answerable, unanswerable)
    t = cal["threshold"]
    tp = sum(s < t for s in unanswerable)
    fp = sum(s < t for s in answerable)
    assert cal["precision"] == pytest.approx(tp / (tp + fp))
    assert cal["recall"] == pytest.approx(tp / 4)
    # no other threshold beats it
    for cand in [0.05, 0.15, 0.25, 0.35, 0.5, 0.65, 0.75, 0.85, 0.95]:
        ptp, pfp = sum(s < cand for s in unanswerable), sum(s < cand for s in answerable)
        assert prf(ptp, pfp, 4 - ptp)[2] <= cal["f1"] + 1e-9


def test_calibrate_false_refusal_cap_beats_f1_pick() -> None:
    answerable = [0.9, 0.8, 0.7, 0.35, 0.3, 0.25]
    unanswerable = [0.5, 0.45, 0.4, 0.2]
    free = calibrate(answerable, unanswerable)
    capped = calibrate(answerable, unanswerable, max_false_refusal=0.1)
    assert free["recall"] == 1.0 and free["false_refusal_rate"] == pytest.approx(0.5)
    assert capped["false_refusal_rate"] == 0.0
    assert capped["recall"] == pytest.approx(1 / 4)  # only the score below every answerable one
    assert capped["f1_optimal"]["threshold"] == free["threshold"]
    assert "false refusal" in capped["criterion"]


def test_auc() -> None:
    assert auc([0.9], [0.1]) == 1.0 and auc([0.1], [0.9]) == 0.0 and auc([0.5], [0.5]) == 0.5


def test_calibrate_needs_both_classes() -> None:
    with pytest.raises(ValueError):
        calibrate([0.5], [])


def test_split_ids_stratified_and_deterministic() -> None:
    items = [{"id": f"u{i}", "type": "A" if i < 40 else "B"} for i in range(60)]
    s = split_ids(items, 42)
    assert s == split_ids(items, 42)
    assert len(s["dev"]) == 13 + 6 and len(s["test"]) == 27 + 14
    assert not set(s["dev"]) & set(s["test"])


def test_build_rejects_holdout_company_in_index() -> None:
    class Cfg:
        seed = 42
        fb_dir = None

    with pytest.raises(SystemExit):
        build(Cfg(), {"Tesla, Inc."})  # type: ignore[arg-type]
