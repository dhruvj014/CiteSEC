"""Answer, citation, abstention and latency metrics -> results/answers_metrics_{split}.json.

Answerable items are scored with the judge verdict. Unanswerable items are scored on whether the
system abstained. A system abstains if its own flag is set (INSUFFICIENT_EVIDENCE token or the
gate) or the judge calls the answer REFUSED.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict

from citesec.config import Config
from citesec.eval.answer_data import answers_path, judgments_path, read_jsonl
from citesec.eval.retrieval_metrics import percentiles

SYSTEMS = ("S0", "S1", "S2", "S3", "S4")
REASONING_GROUPS = ("numerical reasoning", "information extraction", "logical reasoning")


def reasoning_group(label: str | None) -> str:
    """FinanceBench labels are free text like 'Numerical reasoning OR Logical reasoning'."""
    low = (label or "").lower()
    hits = [g for g in REASONING_GROUPS if g in low]
    return min(hits, key=low.index) if hits else "unlabeled"  # the earliest phrase is the main type


def abstained_final(row: dict, judgment: dict | None) -> bool:
    return bool(row.get("abstained")) or (judgment or {}).get("verdict") == "REFUSED"


def rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def system_metrics(rows: list[dict], judgments: dict[str, dict]) -> dict:
    ans = [r for r in rows if r["kind"] == "answerable"]
    una = [r for r in rows if r["kind"] == "unanswerable"]
    verdict = lambda r: judgments.get(r["id"], {}).get("verdict")  # noqa: E731
    correct = [r for r in ans if verdict(r) == "CORRECT"]
    refused_ans = [r for r in ans if abstained_final(r, judgments.get(r["id"]))]
    abst_una = [r for r in una if abstained_final(r, judgments.get(r["id"]))]
    total_abst = len(refused_ans) + len(abst_una)

    by_type: dict[str, list[dict]] = defaultdict(list)
    by_reason: dict[str, list[dict]] = defaultdict(list)
    for r in ans:
        by_type[r["question_type"]].append(r)
        by_reason[reasoning_group(r["question_reasoning"])].append(r)
    def acc(group: dict[str, list[dict]]) -> dict:
        return {k: {"n": len(v), "accuracy": rate(sum(verdict(r) == "CORRECT" for r in v), len(v))}
                for k, v in sorted(group.items())}

    cited = [r for r in rows if r.get("n_citations") is not None and r["system"] != "S0"]
    n_cit = sum(r["n_citations"] for r in cited)
    n_valid = sum(r["n_valid"] for r in cited)
    n_sup = sum(r["n_supported"] for r in cited)
    answered = [r for r in ans if not abstained_final(r, judgments.get(r["id"])) and r in cited]
    faith = [judgments[r["id"]]["faithful"] for r in ans
             if r["id"] in judgments and judgments[r["id"]].get("faithful") is not None]

    lat: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        for k, v in r["latency_ms"].items():
            lat[k].append(v)

    answered_ans = [r for r in ans if not abstained_final(r, judgments.get(r["id"]))]
    has_ctx = [r for r in ans if r.get("gold_in_context") is not None]

    def outcome(r: dict) -> str:
        if abstained_final(r, judgments.get(r["id"])):
            return "refused"
        return "correct" if verdict(r) == "CORRECT" else "incorrect"

    by_retrieval = {}
    for name, flag in (("gold_page_in_context", True), ("gold_page_missing", False)):
        grp = [r for r in has_ctx if r["gold_in_context"] is flag]
        cnt = Counter(outcome(r) for r in grp)
        by_retrieval[name] = {"n": len(grp), "correct": cnt["correct"],
                              "incorrect": cnt["incorrect"], "refused": cnt["refused"]}
    sources = Counter(
        ("gate" if r.get("abstain_reason") == "gate" else "model_token" if r.get("abstained")
         else "judge_only") for r in abst_una)
    return {
        "n_answerable": len(ans), "n_unanswerable": len(una),
        "accuracy": rate(len(correct), len(ans)),
        "accuracy_when_answered": rate(len(correct), len(answered_ans)),
        "n_answered": len(answered_ans),
        "gold_page_in_context": rate(sum(bool(r["gold_in_context"]) for r in has_ctx),
                                     len(has_ctx)),
        "outcomes_by_retrieval": by_retrieval,
        "unanswerable_abstention_source": dict(sources),
        "false_refusals_by_cause": {
            "gold_page_missing_from_context": sum(
                1 for r in refused_ans if r.get("gold_in_context") is False),
            "gold_page_in_context": sum(1 for r in refused_ans if r.get("gold_in_context") is True),
        },
        "accuracy_by_question_type": acc(by_type),
        "accuracy_by_reasoning": acc(by_reason),
        "false_refusal_rate": rate(len(refused_ans), len(ans)),
        "wrong_answer_rate": rate(sum(verdict(r) == "INCORRECT" for r in ans), len(ans)),
        "abstention_recall_unanswerable": rate(len(abst_una), len(una)),
        "abstention_precision": rate(len(abst_una), total_abst),
        "answered_unanswerable_rate": rate(len(una) - len(abst_una), len(una)),
        "faithfulness": rate(sum(faith), len(faith)),
        "n_faithfulness_judged": len(faith),
        "citations": {
            "total": n_cit, "validity": rate(n_valid, n_cit), "support": rate(n_sup, n_valid),
            "invalid_removed": sum(r.get("n_invalid_removed", 0) for r in cited),
            "cited_answer_rate": rate(sum(r["n_valid"] > 0 for r in answered), len(answered)),
            "n_answered": len(answered),
        },
        "latency_ms": {k: percentiles(v) for k, v in lat.items()},
        "judge_parse_errors": sum(j.get("parse_error", False) for j in judgments.values()),
    }


def run(cfg: Config, split: str) -> dict:
    judge_model = cfg["models"].get("judge", cfg["models"]["generator"])
    out: dict = {"split": split, "judge": {"model": judge_model, "kind": "local (Ollama)"},
                 "generator": cfg["models"]["generator"], "systems": {}}
    for s in SYSTEMS:
        rows = read_jsonl(answers_path(cfg, s, split))
        if not rows:
            continue
        judgments = {j["id"]: j for j in read_jsonl(judgments_path(cfg, s, split))}
        out["systems"][s] = system_metrics(rows, judgments)
    cal = cfg.root / "configs" / "calibration.json"
    if cal.exists():
        c = json.loads(cal.read_text())
        out["gate"] = {k: c[k] for k in ("threshold", "criterion", "auc", "precision", "recall",
                                         "false_refusal_rate", "f1_optimal") if k in c}
    path = cfg.results_dir / f"answers_metrics_{split}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out
