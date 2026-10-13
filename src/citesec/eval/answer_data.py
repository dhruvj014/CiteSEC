"""Shared helpers for the answer evaluation stages: question sets, JSONL IO, retrieval cache."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from citesec.config import Config
from citesec.eval.splits import load_split, select
from citesec.eval.unanswerable import load_unanswerable
from citesec.ingest.financebench import load_questions

SYSTEMS = ["S0", "S1", "S2", "S3", "S4"]


@dataclass
class EvalItem:
    id: str
    kind: str  # "answerable" | "unanswerable"
    question: str
    gold_answer: str | None = None
    question_type: str | None = None
    question_reasoning: str | None = None
    company: str | None = None
    gold_pages: tuple[tuple[str, int], ...] = ()  # (doc, 0-indexed page) of the gold evidence


def load_items(cfg: Config, split: str) -> list[EvalItem]:
    """Answerable FinanceBench questions of the split, then the unanswerable ones of that split."""
    items = [
        EvalItem(q.financebench_id, "answerable", q.question, q.answer, q.question_type,
                 q.question_reasoning, q.company,
                 tuple(sorted({(e.evidence_doc_name, e.evidence_page_num) for e in q.evidence})))
        for q in select(load_questions(cfg.fb_dir), load_split(cfg), split)
    ]
    items += [EvalItem(u["id"], "unanswerable", u["question"], None, "unanswerable",
                       None, u["company"])
              for u in load_unanswerable(cfg, split)]
    return items


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def retrieval_cache_path(cfg: Config, split: str) -> Path:
    return cfg.processed_dir / "answer_cache" / f"retrieval_{split}.jsonl"


def answers_path(cfg: Config, system: str, split: str) -> Path:
    return cfg.results_dir / f"answers_{system}_{split}.jsonl"


def judgments_path(cfg: Config, system: str, split: str) -> Path:
    return cfg.processed_dir / "answer_cache" / f"judgments_{system}_{split}.jsonl"
