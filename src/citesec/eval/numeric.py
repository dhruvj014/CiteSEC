"""Deterministic numeric grading for references that are a single number.

A candidate is CORRECT when one of the figures in its final answer equals the reference after
rounding to the reference's own precision, allowing unit scaling (thousand, million, billion) and
percent versus fraction. Anything else is decided by the LLM judge.
"""
from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal

NUM_RE = re.compile(r"(?<![\w.])\(?-?\$?\s?\(?-?\d[\d,]*(?:\.\d+)?\)?\s?%?")
UNIT_RE = re.compile(r"\s*(thousand|million|billion|trillion|mn|bn|m|b|k)\b", re.IGNORECASE)
UNIT = {"thousand": 1e3, "k": 1e3, "million": 1e6, "mn": 1e6, "m": 1e6,
        "billion": 1e9, "bn": 1e9, "b": 1e9, "trillion": 1e12}
SCALES = (1.0, 1e3, 1e-3, 1e6, 1e-6, 1e9, 1e-9, 100.0, 0.01)
MAX_REFERENCE_WORDS = 12
FINAL_RE = re.compile(r"final answer\s*:?\s*(.*)", re.IGNORECASE | re.DOTALL)


def _to_float(raw: str) -> float | None:
    s = raw.strip()
    neg = s.startswith("(") and s.endswith(")") or "-" in s
    digits = re.sub(r"[^\d.]", "", s)
    if not digits or digits == ".":
        return None
    try:
        v = float(digits)
    except ValueError:
        return None
    return -v if neg else v


def decimals(raw: str) -> int:
    """Decimal places the reference actually specifies: 1577.00 has 0, 30.8 has 1, 3.46 has 2."""
    num = re.sub(r"[^\d.]", "", raw)
    if "." not in num:
        return 0
    frac = num.split(".")[1].rstrip("0")
    return len(frac)


def extract(text: str) -> list[tuple[str, float, float]]:
    """(raw, value, unit multiplier) for each number in the text."""
    out = []
    for m in NUM_RE.finditer(text):
        v = _to_float(m.group(0))
        if v is None:
            continue
        unit = UNIT_RE.match(text, m.end())
        out.append((m.group(0).strip(), v, UNIT[unit.group(1).lower()] if unit else 1.0))
    return out


def _same(ref: float, cand: float, dec: int) -> bool:
    """Equal after rounding both to `dec` decimals, financial (half-up) rounding."""
    q = Decimal(1).scaleb(-dec)
    a = Decimal(repr(ref)).quantize(q, rounding=ROUND_HALF_UP)
    b = Decimal(repr(round(cand, dec + 6))).quantize(q, rounding=ROUND_HALF_UP)
    return a == b


def numeric_reference(reference: str) -> tuple[float, int] | None:
    """(value, decimals) if the reference holds exactly one non-year number in a short sentence."""
    # years like "FY 2022" are context, not answer figures, so they do not count
    nums = [n for n in extract(reference) if not re.fullmatch(r"(19|20)\d{2}", n[0].strip("$ "))]
    words = re.findall(r"[A-Za-z]+", reference)
    if len(nums) != 1 or len(words) > MAX_REFERENCE_WORDS:
        return None
    raw, value, _unit = nums[0]
    return value, decimals(raw)


def final_text(answer: str) -> str:
    m = FINAL_RE.search(answer)
    return (m.group(1) if m else answer[-300:]).strip()


def numeric_verdict(reference: str | None, answer: str) -> str | None:
    """CORRECT or INCORRECT for single-number references, None when the LLM must decide."""
    if not reference:
        return None
    ref = numeric_reference(reference)
    if ref is None:
        return None
    ref_val, dec = ref
    cands = extract(final_text(answer))
    if not cands:
        return None
    for _raw, v, unit in cands:
        for value in {v, v * unit}:
            for k in SCALES:
                if _same(ref_val, value * k, dec):
                    return "CORRECT"
    return "INCORRECT"
