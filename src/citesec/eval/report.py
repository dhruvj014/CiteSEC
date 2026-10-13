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


SYSTEM_DESC = {
    "S0": "Closed-book LLM, no retrieval",
    "S1": "Naive chunks, dense top-5, no rerank, no gate",
    "S2": "Selected retrieval (structured, hybrid, rerank, filter), no gate",
    "S3": "S2 + citation verification (invalid citations removed)",
    "S4": "S3 + confidence gate (full system)",
}


def p(x: float | None) -> str:
    return "n/a" if x is None else pct(x)


def section_answers(m: dict | None) -> str:
    if not m:
        return NOT_MEASURED
    rows = []
    for s, d in m["systems"].items():
        c = d["citations"]
        lat = d["latency_ms"].get("total")
        rows.append([s, SYSTEM_DESC.get(s, ""), p(d["accuracy"]),
                     p(d["accuracy_when_answered"]), p(d["wrong_answer_rate"]),
                     p(d["false_refusal_rate"]), p(d["faithfulness"]),
                     p(c["validity"]), p(c["support"]), p(c["cited_answer_rate"]),
                     f"{lat['p50'] / 1000:.1f} / {lat['p95'] / 1000:.1f}" if lat else "-"])
    head = ["System", "Description", "Accuracy", "Accuracy when answered", "Wrong",
            "False refusal", "Faithful*", "Cite valid", "Cite support", "Cited answers",
            "Latency s p50 / p95"]
    first = next(iter(m["systems"].values()))
    note = (f"{first['n_answerable']} answerable questions per system. Percent. Accuracy is the "
            f"judge verdict CORRECT over all answerable questions, so refusals count against it. "
            f"Judge: {m['judge']['model']} ({m['judge']['kind']}), generator: {m['generator']}. "
            "Citation validity and support are measured on the raw model output, before invalid "
            "citations are removed, so S2, S3 and S4 share them. Cite support is a heuristic: a "
            "figure or key phrase from the claim appears in the cited block. Cited answers is the "
            "share of answered questions with at least one valid citation. Wrong is the share of "
            "answerable questions answered incorrectly, the hallucination rate. "
            "*Faithful is judged by the same local model and has not been validated against "
            "labels, treat it as indicative only.")
    return f"{note}\n\n{table(head, rows)}"


def section_outcomes(m: dict | None, system: str) -> str:
    if not m or system not in m["systems"]:
        return NOT_MEASURED
    d = m["systems"][system]
    rows = []
    for name, label in (("gold_page_in_context", "Gold page in the 5 context blocks"),
                        ("gold_page_missing", "Gold page missing from context")):
        o = d["outcomes_by_retrieval"][name]
        rows.append([label, str(o["n"]), str(o["correct"]), str(o["incorrect"]), str(o["refused"])])
    src = d.get("unanswerable_abstention_source", {})
    extra = ""
    if src:
        extra = (f"\n\nUnanswerable questions refused by the confidence gate: "
                 f"{src.get('gate', 0)}, "
                 f"by the model's own INSUFFICIENT_EVIDENCE reply: {src.get('model_token', 0)}, "
                 f"by judge ruling only: {src.get('judge_only', 0)} "
                 f"(of {d['n_unanswerable']} unanswerable).")
    return (f"Counts of answerable questions for {system}, split by whether retrieval put the gold "
            f"page in the context.\n\n"
            + table(["Retrieval outcome", "n", "Correct", "Incorrect", "Refused"], rows) + extra)


def section_variance(var: dict | None, det: dict | None) -> str:
    if not var and not det:
        return NOT_MEASURED
    parts = []
    if det:
        parts.append(
            f"Regenerating {det['n']} dev answers at temperature {det['temperature']} with a fixed "
            f"seed: {det['n_fresh_pairs_identical']} of {det['n']} back-to-back repeats were "
            f"identical, but only {det['n_identical_to_cached']} of {det['n']} matched the earlier "
            "cached run. Output is stable within a server session and not bit-reproducible across "
            "sessions (prompt-cache state).")
    if var:
        a = var["answerable"]
        parts.append(
            f"Effect on the metrics (dev, S2, every answer regenerated once): text identical for "
            f"{pct(var['text_identical'])}% of items, judge verdict identical for "
            f"{pct(var['verdict_agreement'])}%, accuracy {pct(a['accuracy_run1'])}% then "
            f"{pct(a['accuracy_run2'])}%, refusal rate {pct(a['refusal_rate_run1'])}% then "
            f"{pct(a['refusal_rate_run2'])}%, unanswerable abstained "
            f"{var['unanswerable_abstained_run1']} then {var['unanswerable_abstained_run2']} of "
            f"{var['n_unanswerable']}.")
    return "\n\n".join(parts)


def section_abstention(m: dict | None) -> str:
    if not m:
        return NOT_MEASURED
    rows = []
    for s, d in m["systems"].items():
        if d["n_unanswerable"] == 0:
            continue
        rows.append([s, p(d["abstention_recall_unanswerable"]), p(d["abstention_precision"]),
                     p(d["answered_unanswerable_rate"]), p(d["false_refusal_rate"])])
    n_una = max(d["n_unanswerable"] for d in m["systems"].values())  # S0 has none
    out = [f"{n_una} unanswerable questions (out-of-corpus company or "
           "unindexed fiscal year). Percent. Abstention counts the model replying "
           "INSUFFICIENT_EVIDENCE, the gate firing, or the judge ruling the answer a refusal.",
           table(["System", "Abstention recall", "Abstention precision",
                  "Answered anyway (hallucinated)", "False refusal on answerable"], rows)]
    g = m.get("gate")
    if g:
        f1 = g["f1_optimal"]
        out.append(
            f"Gate: top reranker score below {g['threshold']:.4f}, chosen on dev with "
            f"{g['criterion']} (dev precision {pct(g['precision'])}%, recall {pct(g['recall'])}%, "
            f"false refusal {pct(g['false_refusal_rate'])}%, score AUC {g['auc']:.2f}). The max-F1 "
            f"threshold {f1['threshold']:.4f} would refuse {pct(f1['false_refusal_rate'])}% of "
            "answerable dev questions, so it was not used.")
    return "\n\n".join(out)


def section_breakdown(m: dict | None, system: str) -> str:
    if not m or system not in m["systems"]:
        return NOT_MEASURED
    d = m["systems"][system]
    rows = [[f"type: {k}", str(v["n"]), p(v["accuracy"])]
            for k, v in d["accuracy_by_question_type"].items()]
    rows += [[f"reasoning: {k}", str(v["n"]), p(v["accuracy"])]
             for k, v in d["accuracy_by_reasoning"].items()]
    return table(["Group", "n", "Accuracy"], rows)


def section_answer_latency(m: dict | None) -> str:
    if not m:
        return NOT_MEASURED
    stages = ["retrieve", "rerank", "generate", "total"]
    rows = []
    for s, d in m["systems"].items():
        cells = []
        for st in stages:
            lat = d["latency_ms"].get(st)
            cells.append(f"{lat['p50']:.0f} / {lat['p95']:.0f}" if lat else "-")
        rows.append([s] + cells)
    return ("Milliseconds per question, p50 / p95, local Ollama generator on a single GPU.\n\n"
            + table(["System"] + stages, rows))


JUDGE_RUNS = [  # (file tag, judge version, round, fresh when the judge was designed?)
    ("r1_v1", "v1", 1, "first look"),
    ("r2_v2", "v2", 2, "yes"),
    ("r1_v3", "v3", 1, "no, seen"),
    ("r2_v3", "v3", 2, "no, seen"),
    ("r3_v3", "v3", 3, "yes"),
]
JUDGE_VERSIONS = {
    "v1": "LLM decides every verdict",
    "v2": "v1 + refusal token by rule + stricter prompt on fabricated detail",
    "v3": "v2 + single-number references graded by rule (final answer vs reference, rounding, "
          "unit scaling)",
}


def section_judge(runs: dict[str, dict]) -> str:
    """`runs` maps file tag (like r3_v3) to its validation result."""
    present = [(t, v, r, f) for t, v, r, f in JUDGE_RUNS if t in runs]
    if not present:
        return NOT_MEASURED
    rows = []
    for tag, ver, rnd, fresh in present:
        d = runs[tag]
        pl = d["per_label"]
        rows.append([ver, str(rnd), fresh, str(d["n"]), pct(d["agreement"]),
                     f"{d['cohen_kappa']:.2f}", p(pl["CORRECT"]["recall"]),
                     p(pl["INCORRECT"]["recall"]), p(pl["REFUSED"]["recall"])])
    v3 = [runs[t] for t, v, _r, _f in present if v == "v3"]
    pooled = ""
    if len(v3) > 1:
        agree = sum(round(d["agreement"] * d["n"]) for d in v3)
        total = sum(d["n"] for d in v3)
        pooled = f" Judge v3 pooled over {total} labeled answers: {pct(agree / total)}% agreement."
    first = runs[present[0][0]]
    note = (f"Judge model {first['judge_model']} ({first['judge_kind']}). Labels were written by "
            f"{first['labeled_by'].split(',')[0]}, blind to the judge verdicts, from dev answers "
            "only. They have not been audited by a human reviewer. The judge was changed after "
            "each round showed its mistakes, so only rows marked fresh are unbiased for that "
            "version. Recall columns are the share of items with that label that the judge "
            "reproduced." + pooled)
    ver_lines = "\n".join(f"- {k}: {v}" for k, v in JUDGE_VERSIONS.items())
    head = ["Judge", "Round", "Fresh sample", "Items", "Agreement %", "Kappa", "CORRECT recall",
            "INCORRECT recall", "REFUSED recall"]
    return f"{note}\n\n{ver_lines}\n\n{table(head, rows)}"


def render(results_dir: Path) -> str:
    stats = load(results_dir, "data_stats.json")
    index = load(results_dir, "index_stats.json")
    dev = load(results_dir, "retrieval_dev.json")
    test = load(results_dir, "retrieval_test.json")
    root = results_dir.parent
    sel_path = root / "configs" / "retrieval_selected.json"
    selected = json.loads(sel_path.read_text()) if sel_path.exists() else None
    chosen = selected["config_id"] if selected else "F"
    ans_test = load(results_dir, "answers_metrics_test.json")
    ans_dev = load(results_dir, "answers_metrics_dev.json")
    judge_runs = {t: d for t, *_ in JUDGE_RUNS
                  if (d := load(results_dir, f"judge_validation_{t}.json"))}
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
        "## Answer quality\n\n### Test split\n\n" + section_answers(ans_test),
        "### Test split accuracy by group (S4)\n\n" + section_breakdown(ans_test, "S4"),
        "### Outcomes by retrieval (test split, S4)\n\n" + section_outcomes(ans_test, "S4"),
        "### Answer latency (test split)\n\n" + section_answer_latency(ans_test),
        "### Dev split\n\n" + section_answers(ans_dev),
        "## Abstention\n\n### Test split\n\n" + section_abstention(ans_test),
        "### Dev split\n\n" + section_abstention(ans_dev),
        "## Judge validation\n\n" + section_judge(judge_runs),
        "## Generation variance\n\n" + section_variance(
            load(results_dir, "generation_variance_dev.json"),
            load(results_dir, "generation_determinism.json")),
        "## Serving\n\n" + NOT_MEASURED + " (Phase 5).",
    ]
    return "\n\n".join(parts) + "\n"
