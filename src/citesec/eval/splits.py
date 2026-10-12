"""Fixed, seeded, stratified dev/test split of the 150 FinanceBench questions.

dev (50) is for every choice (thresholds, retrieval config). test (100) is for final numbers only.
The split file is written once and must never change.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from citesec.config import Config
from citesec.schemas import Question

DEV_SIZE = 50


def make_split(questions: list[Question], seed: int, dev_size: int = DEV_SIZE) -> dict:
    by_type: dict[str, list[str]] = defaultdict(list)
    for q in questions:
        by_type[q.question_type].append(q.financebench_id)
    types = sorted(by_type)
    total = len(questions)
    quota = {t: dev_size * len(by_type[t]) // total for t in types}
    for t in types[: dev_size - sum(quota.values())]:  # distribute the remainder deterministically
        quota[t] += 1
    rng = random.Random(seed)
    dev: list[str] = []
    test: list[str] = []
    for t in types:
        ids = sorted(by_type[t])
        rng.shuffle(ids)
        dev += ids[: quota[t]]
        test += ids[quota[t] :]
    return {
        "seed": seed,
        "stratified_by": "question_type",
        "dev": sorted(dev),
        "test": sorted(test),
        "dev_by_type": {t: quota[t] for t in types},
    }


def split_path(cfg: Config) -> Path:
    return cfg.root / "data" / "eval" / "split.json"


def load_split(cfg: Config) -> dict:
    return json.loads(split_path(cfg).read_text(encoding="utf-8"))


def ensure_split(cfg: Config, questions: list[Question]) -> dict:
    """Create the split file if absent. Never overwrites an existing one."""
    path = split_path(cfg)
    if path.exists():
        return load_split(cfg)
    split = make_split(questions, cfg.seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=2), encoding="utf-8")
    return split


def select(questions: list[Question], split: dict, name: str) -> list[Question]:
    ids = set(split[name])
    return [q for q in questions if q.financebench_id in ids]
