from citesec.eval.judge import (
    build_faith_messages,
    build_judge_messages,
    parse_faith,
    parse_verdict,
)
from citesec.eval.stage_judge import pair_key


def test_parse_verdict_variants() -> None:
    assert parse_verdict("CORRECT") == ("CORRECT", False)
    assert parse_verdict("INCORRECT") == ("INCORRECT", False)  # not mistaken for CORRECT
    assert parse_verdict("  refused.\n") == ("REFUSED", False)
    assert parse_verdict("Verdict: Correct because the numbers match") == ("CORRECT", False)
    assert parse_verdict("The candidate is incorrect") == ("INCORRECT", False)


def test_parse_verdict_garbage_is_flagged() -> None:
    assert parse_verdict("") == ("INCORRECT", True)
    assert parse_verdict("I am not sure") == ("INCORRECT", True)


def test_refusal_token_is_decided_by_rule() -> None:
    from citesec.eval.judge import rule_refusal

    assert rule_refusal("INSUFFICIENT_EVIDENCE  Final answer: INSUFFICIENT_EVIDENCE")
    assert rule_refusal("The blocks are about Intel. INSUFFICIENT_EVIDENCE")
    assert not rule_refusal("Final answer: $5 million")
    assert not rule_refusal("The exact figure is not provided here")  # prose is left to the LLM


def test_prompt_penalizes_contradicting_details() -> None:
    text = build_judge_messages("q", "yes, 16%", "yes, 75%")[0]["content"]
    assert "contradict the reference" in text


def test_parse_faith() -> None:
    assert parse_faith("YES") == (True, False)
    assert parse_faith("No, the number is unsupported") == (False, False)
    assert parse_faith("maybe") == (None, True)


def test_judge_prompt_contains_all_parts() -> None:
    m = build_judge_messages("What was capex?", "$1,577 million", "Capex was $1.577 billion")
    text = m[0]["content"]
    assert "What was capex?" in text and "$1,577 million" in text and "1.577 billion" in text
    assert "REFUSED" in text and "INSUFFICIENT_EVIDENCE" in text


def test_unanswerable_reference_asks_for_refusal() -> None:
    text = build_judge_messages("What was Tesla revenue?", None, "It was 5")[0]["content"]
    assert "cannot be answered" in text and "refuse" in text


def test_long_answers_keep_the_end() -> None:
    long = "x" * 5000 + " Final answer: 42"
    text = build_judge_messages("q", "42", long)[0]["content"]
    assert "Final answer: 42" in text and len(text) < 4500
    assert "Final answer: 42" in build_faith_messages(long, "evidence")[0]["content"]


class FakeStore:
    def __init__(self, chunks):
        self.chunks = chunks

    def get_many(self, ids):
        return [self.chunks[i] for i in ids if i in self.chunks]


def test_evidence_prefers_cited_blocks_then_falls_back_to_all_context() -> None:
    from citesec.eval.stage_judge import evidence_text
    from citesec.schemas import Chunk

    mk = lambda i, t: Chunk(chunk_id=i, doc_name="D", company="A", doc_type="10K",  # noqa: E731
                            period="2022", page=0, text=t, n_tokens=3, header="H")
    store = FakeStore({"c1": mk("c1", "alpha"), "c2": mk("c2", "beta")})
    ctx = [{"chunk_id": "c1"}, {"chunk_id": "c2"}]
    cited = {"context": ctx, "citations": [{"valid": True, "chunk_id": "c2"},
                                           {"valid": False, "chunk_id": None}]}
    assert evidence_text(cited, store) == "H\nbeta"
    uncited = {"context": ctx, "citations": []}
    assert evidence_text(uncited, store) == "H\nalpha\n\nH\nbeta"


def test_pair_key_is_stable_and_distinguishes() -> None:
    assert pair_key("q1", "a") == pair_key("q1", "a")
    assert pair_key("q1", "a") != pair_key("q2", "a")
    assert pair_key("q1", "a") != pair_key("q1", "b")
