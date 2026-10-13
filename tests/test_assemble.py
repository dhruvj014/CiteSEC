import json
from pathlib import Path

import pytest

from citesec.config import Config
from citesec.eval import stage_assemble
from citesec.eval.answer_data import EvalItem, answers_path, read_jsonl
from citesec.schemas import Chunk


def chunk(cid, page, text):
    return Chunk(chunk_id=cid, doc_name="ACME_2022_10K", company="Acme", doc_type="10K",
                 period="2022", page=page, text=text, n_tokens=5, header="Acme | 10K 2022")


class FakeStore:
    def __init__(self, chunks):
        self.by_id = {c.chunk_id: c for c in chunks}

    def get_many(self, ids):
        return [self.by_id[i] for i in ids if i in self.by_id]


@pytest.fixture
def env(tmp_path: Path, monkeypatch):
    cfg = Config({"seed": 42, "paths": {"raw_financebench": "fb", "raw_edgar": "edgar",
                                        "processed": "proc", "results": "results"}}, root=tmp_path)
    (tmp_path / "results").mkdir()
    items = [
        EvalItem("a1", "answerable", "What were net sales?", "$5,577 million", "novel-generated",
                 gold_pages=(("ACME_2022_10K", 50),)),
        EvalItem("a2", "answerable", "What was capex?", "$100", "metrics-generated",
                 "Numerical reasoning", gold_pages=(("ACME_2022_10K", 7),)),
        EvalItem("u1", "unanswerable", "What was Tesla revenue?"),
    ]
    store = FakeStore([chunk("c50", 50, "Net sales were $5,577 million."),
                       chunk("c51", 51, "Capex notes.")])
    monkeypatch.setattr(stage_assemble, "load_items", lambda cfg, split: items)
    monkeypatch.setattr(stage_assemble, "stores", lambda cfg: {"S1": store, "S2": store})

    def write(path: Path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    t = {"embed_query": 10.0, "dense": 5.0, "fetch": 2.0, "rerank": 100.0, "total": 117.0}
    ret = []
    for i, score in (("a1", 0.9), ("a2", 0.01), ("u1", 0.0)):
        for s in ("S1", "S2"):
            ret.append({"id": i, "system": s, "chunk_ids": ["c50", "c51"] if i != "u1" else [],
                        "scores": [score, 0.1] if i != "u1" else [], "timings_ms": t})
    write(tmp_path / "proc" / "answer_cache" / "retrieval_dev.jsonl", ret)

    def gen(i, s, raw):
        return {"id": i, "system": s, "raw": raw, "prompt_tokens": 100, "completion_tokens": 20,
                "gen_ms": 5000.0}

    good = ("Net sales were $5,577 million [ACME_2022_10K, p.51] and also "
            "[ACME_2022_10K, p.99]. Final answer: $5,577 million")
    write(tmp_path / "proc" / "answer_cache" / "llm_dev.jsonl", [
        gen("a1", "S0", "About $9 billion. Final answer: $9 billion"),
        gen("a2", "S0", "Final answer: $1"), gen("a1", "S1", good), gen("a2", "S1", "x"),
        gen("u1", "S1", "INSUFFICIENT_EVIDENCE"), gen("a1", "S2", good),
        gen("a2", "S2", "Capex was $100 [ACME_2022_10K, p.52]."),
        gen("u1", "S2", "Tesla revenue was $5 [ACME_2022_10K, p.51]")])
    return cfg


def test_assemble_derives_s3_and_s4_from_s2(env: Config) -> None:
    counts = stage_assemble.run(env, "dev", threshold=0.05)
    assert counts == {"S0": 2, "S1": 3, "S2": 3, "S3": 3, "S4": 3}
    rows = {s: {r["id"]: r for r in read_jsonl(answers_path(env, s, "dev"))}
            for s in ("S0", "S1", "S2", "S3", "S4")}

    # S0: closed book, no context, no citations, no unanswerable rows
    assert rows["S0"]["a1"]["context"] == [] and "u1" not in rows["S0"]

    # S2 keeps the raw text including the invalid p.99 tag, with counts computed on it
    s2 = rows["S2"]["a1"]
    assert s2["gold_in_context"] is True and rows["S2"]["a2"]["gold_in_context"] is False
    assert rows["S2"]["u1"]["gold_in_context"] is None
    assert "p.99" in s2["answer"] and s2["n_citations"] == 2 and s2["n_valid"] == 1
    assert s2["citations"][0]["chunk_id"] == "c50" and s2["citations"][0]["valid"] is True
    assert s2["top_score"] == 0.9

    # S3 strips the invalid citation but keeps the valid one, and records how many were removed
    s3 = rows["S3"]["a1"]
    assert "p.99" not in s3["answer"] and "[ACME_2022_10K, p.51]" in s3["answer"]
    assert s3["n_invalid_removed"] == 1 and s3["grounded"] is True

    # S4 = S3 when the top score clears the threshold
    s4 = rows["S4"]["a1"]
    assert s4["answer"] == s3["answer"] and s4["abstained"] is False

    # gate fires below the threshold: no generation time, answer is the refusal token
    g = rows["S4"]["a2"]
    assert g["abstained"] and g["abstain_reason"] == "gate"
    assert g["answer"] == "INSUFFICIENT_EVIDENCE"
    assert g["latency_ms"]["generate"] == 0.0 and g["tokens"] == {"prompt": 0, "completion": 0}
    assert g["latency_ms"]["total"] == g["latency_ms"]["retrieve"] + g["latency_ms"]["rerank"]
    # S2 and S3 still answer a2 (no gate), so the gate's effect is visible
    assert rows["S2"]["a2"]["abstained"] is False and rows["S3"]["a2"]["abstained"] is False

    # empty retrieval on the unanswerable item: gated even though S2 hallucinated an answer
    assert rows["S4"]["u1"]["abstain_reason"] == "gate"
    assert rows["S2"]["u1"]["abstained"] is False and rows["S1"]["u1"]["abstain_reason"] == "llm"


def test_latency_breakdown_fields(env: Config) -> None:
    stage_assemble.run(env, "dev", threshold=0.05)
    lat = read_jsonl(answers_path(env, "S2", "dev"))[0]["latency_ms"]
    assert lat["retrieve"] == 17.0 and lat["rerank"] == 100.0 and lat["generate"] == 5000.0
    assert lat["total"] == 5117.0
