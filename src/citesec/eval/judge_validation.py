"""Validate the LLM judge against hand labels before trusting any accuracy number.

    python -m citesec.eval.judge_validation sample     # pick 40 dev answers -> items file
    (fill in the "label" field of each item by reading question, reference and answer)
    python -m citesec.eval.judge_validation evaluate   # -> results/judge_validation.json

Items come from the DEV split only. The sample mixes systems and includes unanswerable questions.
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

from citesec.config import Config, load_config
from citesec.eval.answer_data import answers_path, judgments_path, read_jsonl
from citesec.eval.judge import JUDGE_VERSION, VERDICTS

N_ITEMS = 40
PLAN = {"S2": 14, "S1": 10, "S0": 8, "unanswerable": 8}  # unanswerable drawn from S1 and S2
ITEMS_FILE = "judge_validation_items.jsonl"


PLAN_R2 = {"S2": 10, "S1": 10, "S0": 10}  # fresh answers that are not refusal tokens
PLAN_R3 = {"S2": 12, "S1": 12, "S0": 12}  # judge v3, drawn after the numeric rule was designed
PLANS = {2: PLAN_R2, 3: PLAN_R3}


def items_path(cfg: Config, round_no: int = 1) -> Path:
    name = ITEMS_FILE if round_no == 1 else ITEMS_FILE.replace(".jsonl", f"_r{round_no}.jsonl")
    return cfg.root / "data" / "eval" / name


def sample_round2(rows: dict[str, list[dict]], used: set[tuple[str, str]], seed: int,
                  round_no: int = 2) -> list[dict]:
    """Fresh dev answers for validating a later judge version: not used in earlier rounds and not
    refusal tokens. Refusals are decided by rule, so the judge only separates CORRECT from
    INCORRECT."""
    rng = random.Random(seed + round_no)
    picked: list[tuple[str, dict]] = []
    for system, n in PLANS[round_no].items():
        pool = [r for r in rows.get(system, [])
                if r["kind"] == "answerable" and (system, r["id"]) not in used
                and "INSUFFICIENT_EVIDENCE" not in r["answer"]]
        picked += [(system, r) for r in rng.sample(pool, min(n, len(pool)))]
    prefix = f"jv{round_no}"
    return [{"val_id": f"{prefix}_{i:02d}", "id": r["id"], "system": s, "kind": r["kind"],
             "question": r["question"], "reference": r["gold_answer"], "answer": r["answer"],
             "label": None} for i, (s, r) in enumerate(picked)]


def sample_items(rows: dict[str, list[dict]], seed: int, plan: dict[str, int] = PLAN) -> list[dict]:
    """Seeded stratified sample. `rows` maps system name to its dev answer rows."""
    rng = random.Random(seed)
    picked: list[tuple[str, dict]] = []
    for system in ("S2", "S1", "S0"):
        pool = [r for r in rows.get(system, []) if r["kind"] == "answerable"]
        picked += [(system, r) for r in rng.sample(pool, min(plan[system], len(pool)))]
    una = [(s, r) for s in ("S1", "S2") for r in rows.get(s, []) if r["kind"] == "unanswerable"]
    picked += rng.sample(una, min(plan["unanswerable"], len(una)))
    return [{"val_id": f"jv_{i:02d}", "id": r["id"], "system": s, "kind": r["kind"],
             "question": r["question"], "reference": r["gold_answer"], "answer": r["answer"],
             "label": None} for i, (s, r) in enumerate(picked)]


def cohen_kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def agreement(labels: list[str], verdicts: list[str]) -> dict:
    n = len(labels)
    conf = Counter(zip(labels, verdicts, strict=True))
    per_label = {}
    for lab in VERDICTS:
        total = sum(v for (a, _), v in conf.items() if a == lab)
        per_label[lab] = {"n": total, "judge_agrees": conf[(lab, lab)],
                          "recall": round(conf[(lab, lab)] / total, 4) if total else None}
    return {
        "n": n,
        "agreement": round(sum(a == b for a, b in zip(labels, verdicts, strict=True)) / n, 4),
        "cohen_kappa": round(cohen_kappa(labels, verdicts), 4),
        "per_label": per_label,
        "confusion": {f"label={a} judge={b}": v for (a, b), v in sorted(conf.items())},
    }


def evaluate(cfg: Config, round_no: int = 1, tag: str | None = None) -> dict:
    """Judge verdicts vs labels for one round. Writes results/judge_validation_{tag}.json."""
    items = read_jsonl(items_path(cfg, round_no))
    missing = [i["val_id"] for i in items if i["label"] not in VERDICTS]
    if missing:
        raise SystemExit(f"{len(missing)} items have no valid label: {missing[:5]}...")
    judg: dict[tuple[str, str], str] = {}
    for system in {i["system"] for i in items}:
        for j in read_jsonl(judgments_path(cfg, system, "dev")):
            judg[(system, j["id"])] = j["verdict"]
    labels, verdicts = [], []
    for i in items:
        v = judg.get((i["system"], i["id"]))
        if v is None:
            raise SystemExit(f"no judgment for {i['system']} {i['id']}, run the judge stage first")
        labels.append(i["label"])
        verdicts.append(v)
    out = agreement(labels, verdicts)
    out.update({
        "split": "dev",
        "round": round_no,
        "judge_version": JUDGE_VERSION,
        "judge_model": cfg["models"].get("judge", cfg["models"]["generator"]),
        "judge_kind": "local (Ollama)",
        "labeled_by": items[0].get("labeled_by", "unknown"),
        "label_counts": dict(Counter(labels)),
        "disagreements": [{"val_id": i["val_id"], "label": lab, "judge": v}
                          for i, lab, v in zip(items, labels, verdicts, strict=True) if lab != v],
    })
    path = cfg.results_dir / f"judge_validation_{tag or f'r{round_no}_{JUDGE_VERSION}'}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def main() -> None:
    cfg = load_config()
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    round_no = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    if cmd == "sample":
        path = items_path(cfg, round_no)
        if path.exists():
            raise SystemExit("items file exists. Labels live there, do not resample.")
        rows = {s: read_jsonl(answers_path(cfg, s, "dev")) for s in ("S0", "S1", "S2")}
        if round_no == 1:
            items = sample_items(rows, cfg.seed)
        else:
            used = {(i["system"], i["id"]) for r in range(1, round_no)
                    for i in read_jsonl(items_path(cfg, r))}
            items = sample_round2(rows, used, cfg.seed, round_no)
        path.write_text("\n".join(json.dumps(i, ensure_ascii=False) for i in items) + "\n",
                        encoding="utf-8")
        print(f"wrote {len(items)} items to {path}")
    elif cmd == "evaluate":
        tag = sys.argv[3] if len(sys.argv) > 3 else None
        print(json.dumps(evaluate(cfg, round_no, tag), indent=2))
    else:
        raise SystemExit("usage: judge_validation.py sample|evaluate [round] [tag]")


if __name__ == "__main__":
    main()
