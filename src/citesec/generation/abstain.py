"""Confidence gate: abstain when the best reranker score is below a threshold chosen on dev.

Positive class for calibration is "should abstain" (the unanswerable questions). The threshold is
the one with the best F1 for that class on dev. It is never chosen on test.
"""
from __future__ import annotations

import json
from pathlib import Path

from citesec.generation.prompts import INSUFFICIENT


def should_abstain(scores: list[float], threshold: float) -> bool:
    """True when nothing was retrieved or the top reranker score is below the threshold."""
    return not scores or scores[0] < threshold


def llm_abstained(answer: str) -> bool:
    """The model replied INSUFFICIENT_EVIDENCE. The token is rare, so it counts anywhere in the
    reply, which also catches answers that reason first and then refuse."""
    return INSUFFICIENT in answer


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def calibrate(answerable: list[float], unanswerable: list[float],
              max_false_refusal: float | None = None) -> dict:
    """Sweep thresholds and pick one, using dev scores only.

    Positive class is "abstain" (unanswerable). Without a cap the pick is the max-F1 threshold.
    With `max_false_refusal` the pick is the threshold with the highest recall on unanswerable
    questions among those refusing at most that share of answerable ones (ties: higher F1, then
    lower threshold). The max-F1 point is always kept under "f1_optimal" for transparency.
    Each input is the top reranker score per question (0.0 if nothing was retrieved).
    """
    if not answerable or not unanswerable:
        raise ValueError("need both answerable and unanswerable scores")
    values = sorted(set(answerable) | set(unanswerable))
    cands = [values[0] - 1e-6] + [(a + b) / 2 for a, b in zip(values, values[1:], strict=False)]
    cands.append(values[-1] + 1e-6)
    rows = []
    for t in cands:
        tp = sum(s < t for s in unanswerable)
        fp = sum(s < t for s in answerable)
        p, r, f1 = prf(tp, fp, len(unanswerable) - tp)
        rows.append({"threshold": t, "f1": f1, "precision": p, "recall": r,
                     "false_refusal_rate": fp / len(answerable)})
    f1_best = max(rows, key=lambda x: (x["f1"], -x["threshold"]))  # lowest threshold on ties
    if max_false_refusal is None:
        best = dict(f1_best)
    else:
        ok = [x for x in rows if x["false_refusal_rate"] <= max_false_refusal + 1e-12]
        best = dict(max(ok, key=lambda x: (x["recall"], x["f1"], -x["threshold"])))
    best.update({"n_answerable": len(answerable), "n_unanswerable": len(unanswerable),
                 "auc": auc(answerable, unanswerable), "selected_on": "dev",
                 "criterion": ("max F1" if max_false_refusal is None else
                               f"max recall with false refusal rate <= {max_false_refusal}"),
                 "f1_optimal": f1_best,
                 "signal": "top reranker score (sigmoid, 0 to 1)"})
    return best


def auc(answerable: list[float], unanswerable: list[float]) -> float:
    """Chance a random answerable question outscores a random unanswerable one."""
    wins = sum((a > u) + 0.5 * (a == u) for a in answerable for u in unanswerable)
    return wins / (len(answerable) * len(unanswerable))


def save_calibration(path: Path, cal: dict) -> None:
    path.write_text(json.dumps(cal, indent=2), encoding="utf-8")


def load_threshold(path: Path) -> float:
    return float(json.loads(path.read_text(encoding="utf-8"))["threshold"])
