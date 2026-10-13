"""Build the committed unanswerable question set: 40 Type A and 20 Type B, seeded.

Type A (out of corpus): a real large cap that is deliberately excluded from the index.
Type B (wrong period): a company that is indexed, asked about a fiscal year the index cannot answer.
A filing also reports prior-year comparison columns (up to 2 earlier years), so Type B years are
either after the latest indexed filing or at least 3 years before the earliest one.

    python -m citesec.eval.unanswerable           # build, validate, write
"""
from __future__ import annotations

import json
import random
from collections import defaultdict

from citesec.config import Config, load_config
from citesec.index.store import ChunkStore
from citesec.ingest.financebench import load_docs

HOLDOUT_NAMES = {  # ticker -> name used in questions
    "UNH": "UnitedHealth Group", "XOM": "Exxon Mobil", "HD": "The Home Depot",
    "PG": "Procter & Gamble", "LLY": "Eli Lilly", "NVDA": "NVIDIA", "TSLA": "Tesla",
    "V": "Visa", "ABBV": "AbbVie", "CSCO": "Cisco Systems",
}
# Only metrics every large company reports, so no question is odd for its company
METRICS = ["total revenue", "net income", "operating income", "capital expenditures",
           "diluted earnings per share", "total assets", "cash and cash equivalents",
           "long-term debt", "total liabilities", "total operating expenses"]
EARLIEST_YEAR = 2013  # earlier years would predate some companies' first filing (e.g. Block)
TEMPLATES = ["What was {metric} for {company} in fiscal {year}?",
             "What was {company}'s {metric} for fiscal year {year}?",
             "How much {metric} did {company} report for FY{year}?"]
LOOKBACK_COVERAGE = 2  # a filing shows this many prior years of comparison data
N_A, N_B = 40, 20
DEV_FRACTION_NUM, DEV_FRACTION_DEN = 1, 3  # same dev share as the 50/150 question split


def build(cfg: Config, indexed_companies: set[str]) -> list[dict]:
    rng = random.Random(cfg.seed)
    items: list[dict] = []
    # Type A: each holdout company x 4 distinct metrics, year drawn from 2020-2023
    leaked = [n for n in HOLDOUT_NAMES.values()
              if any(n.lower() in c.lower() or c.lower() in n.lower() for c in indexed_companies)]
    if leaked:
        raise SystemExit(f"holdout companies present in the index: {leaked}")
    for _ticker, name in sorted(HOLDOUT_NAMES.items()):
        for metric in rng.sample(METRICS, 4):
            items.append(_item("A", name, metric, rng.choice([2020, 2021, 2022, 2023]), rng))
    assert len(items) == N_A
    # Type B: indexed FinanceBench companies with a year the index cannot answer
    years: dict[str, set[int]] = defaultdict(set)
    for d in load_docs(cfg.fb_dir).values():
        if (cfg.fb_dir / "pdfs" / f"{d.doc_name}.pdf").exists() and _indexed(cfg, d.doc_name):
            years[d.company].add(int(d.doc_period))
    companies = sorted(years)
    for company in rng.sample(companies, N_B):
        lo, hi = min(years[company]), max(years[company])
        later = [hi + 1, hi + 2]
        earlier = [y for y in (lo - 4, lo - 5) if y >= EARLIEST_YEAR]
        year = rng.choice(later + earlier)
        assert year not in years[company] and not (lo - LOOKBACK_COVERAGE <= year <= hi)
        items.append(_item("B", company, rng.choice(METRICS), year, rng))
    assert len(items) == N_A + N_B
    for i, it in enumerate(items):
        it["id"] = f"unans_{it['type']}_{i:03d}"
    return items


def _indexed(cfg: Config, doc_name: str) -> bool:
    return (cfg.processed_dir / "pages" / f"{doc_name}.jsonl").exists()


def _item(kind: str, company: str, metric: str, year: int, rng: random.Random) -> dict:
    q = rng.choice(TEMPLATES).format(metric=metric, company=company, year=year)
    return {"type": kind, "question": q, "company": company, "metric": metric,
            "fiscal_year": year, "gold_behavior": "INSUFFICIENT_EVIDENCE"}


def split_ids(items: list[dict], seed: int) -> dict:
    """Seeded split stratified by type, one third dev (like the 50/150 question split)."""
    rng = random.Random(seed)
    dev: list[str] = []
    test: list[str] = []
    for kind in ("A", "B"):
        ids = sorted(i["id"] for i in items if i["type"] == kind)
        rng.shuffle(ids)
        k = len(ids) * DEV_FRACTION_NUM // DEV_FRACTION_DEN
        dev += ids[:k]
        test += ids[k:]
    return {"seed": seed, "dev": sorted(dev), "test": sorted(test)}


def load_unanswerable(cfg: Config, split: str | None = None) -> list[dict]:
    path = cfg.root / "data" / "eval" / "unanswerable.jsonl"
    items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if split is None:
        return items
    ids = set(json.loads((path.parent / "unanswerable_split.json").read_text())[split])
    return [i for i in items if i["id"] in ids]


def main() -> None:
    cfg = load_config()
    out = cfg.root / "data" / "eval"
    if (out / "unanswerable.jsonl").exists():
        raise SystemExit("unanswerable.jsonl exists. It is committed and must not be regenerated.")
    store = ChunkStore(cfg.processed_dir / "index" / "structured" / "chunks.sqlite")
    companies, _ = store.company_period()
    items = build(cfg, set(companies))
    out.mkdir(parents=True, exist_ok=True)
    (out / "unanswerable.jsonl").write_text(
        "\n".join(json.dumps(i) for i in items) + "\n", encoding="utf-8")
    split = split_ids(items, cfg.seed)
    (out / "unanswerable_split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    print(f"wrote {len(items)} questions, dev {len(split['dev'])} test {len(split['test'])}")
    for it in items:
        print(it["id"], "|", it["question"])


if __name__ == "__main__":
    main()
