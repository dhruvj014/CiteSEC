"""Render BENCHMARKS.md from results/*.json. Pure functions, no hand-edited numbers.

A section whose result file is missing says "not measured". Output is deterministic so the report
regenerates identically from the same results.
"""
from __future__ import annotations

import json
from pathlib import Path

NOT_MEASURED = "_not measured_"
GRID_COLS = [("hit@1", "Hit@1"), ("hit@3", "Hit@3"), ("hit@5", "Hit@5"), ("hit@10", "Hit@10"),
             ("recall@5", "Recall@5"), ("mrr", "MRR"), ("doc_hit@5", "Doc-Hit@5")]
LATENCY_STAGES = ["embed_query", "dense", "bm25", "fuse", "filter", "fetch", "rerank", "total"]


def load(results_dir: Path, name: str) -> dict | None:
    p = results_dir / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def section_data(stats: dict | None, index: dict | None) -> str:
    if not stats or not index:
        return NOT_MEASURED
    fb, ed = stats["financebench"], stats["edgar"]
    rows = [
        ["FinanceBench questions", str(fb["n_questions"])],
        ["FinanceBench gold documents (indexed)", str(fb["n_gold_docs_parsed"])],
        ["FinanceBench gold pages", f"{fb['gold_pages']:,}"],
        ["EDGAR filings (distractors)", f"{ed['n_filings']} from {ed['n_companies']} companies"],
        ["Total indexed documents", str(index["structured"]["n_docs"])],
        ["Chunks, naive", f"{index['naive']['n_chunks']:,}"],
        ["Chunks, structured", f"{index['structured']['n_chunks']:,}"],
        ["Table chunks, structured", f"{index['structured']['n_table_chunks']:,}"],
    ]
    return table(["Item", "Value"], rows)


def retrieval_rows(res: dict) -> list[list[str]]:
    rows = []
    for cid, c in res["configs"].items():
        m = c["metrics"]
        rows.append([cid, c["chunker"], c["mode"], "yes" if c["rerank"] else "no",
                     "yes" if c["filter"] else "no"] + [pct(m[k]) for k, _ in GRID_COLS])
    return rows


def section_retrieval(res: dict | None, title: str) -> str:
    if not res:
        return f"### {title}\n\n{NOT_MEASURED}"
    head = ["ID", "Chunker", "Retrieval", "Rerank", "Filter"] + [h for _, h in GRID_COLS]
    note = (f"{res['n_questions']} questions. Values are percent. Gold is the set of "
            "(document, page) pairs from FinanceBench evidence. Open retrieval over all "
            "indexed documents unless Filter is yes. Filter means the company and fiscal year "
            "are parsed from the question with a curated alias table, so it assumes the question "
            "names its company and is not open retrieval.")
    return f"### {title}\n\n{note}\n\n{table(head, retrieval_rows(res))}"


def section_latency(res: dict | None) -> str:
    if not res:
        return NOT_MEASURED
    stages = [s for s in LATENCY_STAGES
              if any(s in c["latency_ms"] for c in res["configs"].values())]
    rows = []
    for cid, c in res["configs"].items():
        cells = []
        for s in stages:
            lat = c["latency_ms"].get(s)
            cells.append(f"{lat['p50']:.0f} / {lat['p95']:.0f}" if lat else "-")
        rows.append([cid] + cells)
    return ("Milliseconds per query, p50 / p95, single query at a time, "
            f"{res['n_questions']} {res['split']} questions.\n\n"
            + table(["ID"] + stages, rows))


def section_by_type(res: dict | None, config_id: str) -> str:
    if not res or config_id not in res["configs"]:
        return NOT_MEASURED
    by = res["configs"][config_id]["by_question_type"]
    rows = [[t, str(m["n"]), pct(m["hit@5"]), pct(m["mrr"]), pct(m["doc_hit@5"])]
            for t, m in by.items()]
    return table(["Question type", "n", "Hit@5", "MRR", "Doc-Hit@5"], rows)


def section_config(dev: dict | None, selected: dict | None) -> str:
    lines = []
    ref = dev
    if ref:
        lines += [f"- Embedding model: `{ref['models']['embed']}`",
                  f"- Reranker: `{ref['models']['reranker']}`",
                  f"- GPU: {ref['gpu']}",
                  f"- Top-N per ranker: {ref['retrieval']['top_n']}, RRF k: "
                  f"{ref['retrieval']['rrf_k']}, seed: {ref['seed']}"]
        if ref.get("repeat_run_identical") is not None:
            lines.append(f"- Seeded rerun gave identical metrics: {ref['repeat_run_identical']}")
    if selected:
        lines.append(f"- Retrieval config selected on dev: **{selected['config_id']}** "
                     f"(criterion: {selected['criterion']})")
    return "\n".join(lines) if lines else NOT_MEASURED


def render(results_dir: Path) -> str:
    stats = load(results_dir, "data_stats.json")
    index = load(results_dir, "index_stats.json")
    dev = load(results_dir, "retrieval_dev.json")
    test = load(results_dir, "retrieval_test.json")
    root = results_dir.parent
    sel_path = root / "configs" / "retrieval_selected.json"
    selected = json.loads(sel_path.read_text()) if sel_path.exists() else None
    chosen = selected["config_id"] if selected else "F"
    parts = [
        "# Benchmarks",
        "Generated by `scripts/make_report.py` from `results/*.json`. Do not edit by hand.",
        "## Dataset and index\n\n" + section_data(stats, index),
        "## Retrieval ablation\n\nThe retrieval config was chosen on dev only. Test is the "
        "final, single run.",
        section_retrieval(dev, "Dev split"),
        section_retrieval(test, "Test split"),
        f"### Test split by question type (config {chosen})\n\n"
        + section_by_type(test, chosen),
        "### Retrieval latency (test split)\n\n" + section_latency(test),
        "## Configuration\n\n" + section_config(dev, selected),
        "## Answer quality, abstention and serving\n\n" + NOT_MEASURED
        + " (Phases 4 and 5).",
    ]
    return "\n\n".join(parts) + "\n"
