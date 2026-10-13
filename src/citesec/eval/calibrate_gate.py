"""Calibrate the abstention threshold on DEV only. Writes configs/calibration.json.

Reads the cached S2 retrieval scores for the dev answerable questions and the dev half of the
unanswerable set. Refuses to run on anything but dev.
"""
from __future__ import annotations

import json
import statistics

from citesec.config import load_config
from citesec.eval.answer_data import load_items, read_jsonl, retrieval_cache_path
from citesec.generation.abstain import calibrate, save_calibration

# The brief says max F1. On dev that point refuses 64% of answerable questions (the reranker score
# is entity-blind, AUC 0.67), so the operating point is capped instead. See docs/phase4-notes.md.
MAX_FALSE_REFUSAL = 0.10


def quantiles(xs: list[float]) -> dict:
    s = sorted(xs)
    q = lambda p: round(s[min(len(s) - 1, int(p * len(s)))], 4)  # noqa: E731
    return {"min": round(s[0], 4), "p25": q(0.25), "median": q(0.5), "p75": q(0.75),
            "max": round(s[-1], 4), "mean": round(statistics.fmean(s), 4)}


def main() -> None:
    cfg = load_config()
    items = load_items(cfg, "dev")
    cache = {(r["id"], r["system"]): r for r in read_jsonl(retrieval_cache_path(cfg, "dev"))}
    top = {it.id: (cache[(it.id, "S2")]["scores"] or [0.0])[0] for it in items}
    ans = [top[i.id] for i in items if i.kind == "answerable"]
    una = [top[i.id] for i in items if i.kind == "unanswerable"]
    cal = calibrate(ans, una, max_false_refusal=MAX_FALSE_REFUSAL)
    cal["score_distribution"] = {"answerable": quantiles(ans), "unanswerable": quantiles(una)}
    save_calibration(cfg.root / "configs" / "calibration.json", cal)
    print(json.dumps(cal, indent=2))


if __name__ == "__main__":
    main()
