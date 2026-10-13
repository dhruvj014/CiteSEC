"""LLM-as-judge prompts and parsing. The default judge is the local Ollama model.

Verdicts: CORRECT, INCORRECT, REFUSED. A separate yes/no call checks faithfulness (is every claim
supported by the cited context). The judge is validated against hand labels before any accuracy
number is trusted, see eval/judge_validation.py.
"""
from __future__ import annotations

import re

from citesec.generation.prompts import INSUFFICIENT

VERDICTS = ("CORRECT", "INCORRECT", "REFUSED")
# v1: LLM decides every verdict (75% agreement with hand labels on dev round 1, see
# results/judge_validation_r1_v1.json). v2: the refusal token is detected by rule, and the prompt
# rejects answers whose supporting figures contradict the reference (77% on fresh round 2, with
# near-miss numbers and partial answers accepted). v3: references that are one number are graded
# by rule (eval/numeric.py), the LLM only judges the rest.
JUDGE_VERSION = "v3"
UNANSWERABLE_REFERENCE = ("This question cannot be answered from the provided filings. "
                          "The correct response is to refuse.")
MAX_ANSWER_CHARS = 2500

JUDGE_PROMPT = """You are grading a candidate answer to a financial question against a reference answer.

Question: {question}

Reference answer: {reference}

Candidate answer: {answer}

Rules:
- Reply REFUSED if the candidate declines to answer, says the information is not available or insufficient, or contains {insufficient}.
- Otherwise reply CORRECT if the candidate's final answer agrees with the reference on the key facts. Numeric answers are CORRECT when they match the reference within rounding or unit scaling, for example 1,577 million versus $1.577 billion, or 12.5% versus 0.125. Extra correct explanation does not make an answer incorrect.
- Otherwise reply INCORRECT. That includes a different number, a different conclusion (for example yes versus no), or missing the key fact. It also includes an answer whose overall yes or no matches the reference but whose supporting figures or facts contradict the reference, or that invents specifics the reference does not contain.

Reply with exactly one word: CORRECT, INCORRECT or REFUSED."""

FAITH_PROMPT = """You are checking whether an answer is faithful to the evidence it cites.

Evidence blocks:

{evidence}

Answer: {answer}

Is every factual claim and every number in the answer supported by the evidence blocks? A computed result counts as supported when it follows correctly from supported numbers. Reply with exactly one word: YES or NO."""


def build_judge_messages(question: str, reference: str | None, answer: str) -> list[dict[str, str]]:
    ref = reference if reference else UNANSWERABLE_REFERENCE
    content = JUDGE_PROMPT.format(question=question, reference=ref, insufficient=INSUFFICIENT,
                                  answer=answer[-MAX_ANSWER_CHARS:])
    return [{"role": "user", "content": content}]


def rule_refusal(answer: str) -> bool:
    """The model replied with the refusal token. Decided by rule, not by the LLM judge, because
    the token is defined in the generation prompt and the LLM judge confused refusals with
    correct 'not available' reference answers."""
    return INSUFFICIENT in answer


def parse_verdict(text: str) -> tuple[str, bool]:
    """(verdict, parse_error). INCORRECT is checked before CORRECT because it contains it."""
    m = re.search(r"\b(INCORRECT|CORRECT|REFUSED)\b", text.upper())
    if m:
        return m.group(1), False
    return "INCORRECT", True


def build_faith_messages(answer: str, evidence: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": FAITH_PROMPT.format(
        evidence=evidence, answer=answer[-MAX_ANSWER_CHARS:])}]


def parse_faith(text: str) -> tuple[bool | None, bool]:
    """(faithful, parse_error)."""
    m = re.search(r"\b(YES|NO)\b", text.upper())
    if m:
        return m.group(1) == "YES", False
    return None, True
