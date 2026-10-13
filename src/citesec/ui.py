"""Pure helpers for the Streamlit demo, kept out of the app file so they can be unit tested."""
from __future__ import annotations

import os
import re

import httpx

from citesec.generation.citations import CITE_RE, parse_citations

DEFAULT_API = os.environ.get("CITESEC_API", "http://127.0.0.1:8765")
EXAMPLES = [
    "By how much did Pepsico increase its unsecured five year revolving credit agreement "
    "on May 26, 2023?",
    "What is Amcor's year end FY2020 net AR (in USD millions)?",
    "Did AMD report customer concentration in FY22?",
    "What was Tesla's net income for fiscal year 2023?",
]
ABSTAIN_TEXT = {
    "gate": "Retrieval confidence was too low, so no answer was generated. Nothing in the indexed "
            "filings matched this question well enough.",
    "model": "The retrieved passages do not contain what is needed to answer for that company and "
             "period, so the model declined to answer.",
}
_BARE_DOLLAR = re.compile(r"(?<!\\)\$")  # a $ that is not already escaped
_DISPLAY_MATH = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)
_INLINE_MATH = re.compile(r"\\\((.+?)\\\)", re.DOTALL)


def api_query(base: str, payload: dict, timeout: float = 180.0) -> dict:
    """POST /query. Raises RuntimeError with a readable message on any failure."""
    try:
        r = httpx.post(f"{base.rstrip('/')}/query", json=payload, timeout=timeout)
    except httpx.HTTPError as exc:
        raise RuntimeError(f"Cannot reach the API at {base}. Is it running? ({exc})") from exc
    if r.status_code == 503:
        raise RuntimeError(r.json().get("detail", "The language model is not reachable."))
    if r.status_code != 200:
        raise RuntimeError(f"The API returned HTTP {r.status_code}: {r.text[:200]}")
    return r.json()


def api_health(base: str) -> dict | None:
    try:
        r = httpx.get(f"{base.rstrip('/')}/health", timeout=3.0)
        return r.json() if r.status_code == 200 else None
    except httpx.HTTPError:
        return None


def citation_label(doc_name: str, page: int | None, section: str | None) -> str:
    where = f"p.{page}" if page is not None else (section or "section")
    return f"{doc_name} · {where}"


def format_answer(text: str) -> str:
    """Markdown for display: citation tags become badges, LaTeX delimiters become $ and $$."""

    def badge(m: re.Match) -> str:
        cites = parse_citations(m.group(0))
        if not cites:
            return m.group(0)
        c = cites[0]
        return f":blue-badge[{citation_label(c.doc, c.page, c.section)}]"

    # Streamlit reads a pair of $ as LaTeX, so currency must be escaped before math is converted
    out = CITE_RE.sub(badge, _BARE_DOLLAR.sub(r"\\$", text))
    out = _DISPLAY_MATH.sub(lambda m: f"\n$$ {m.group(1).strip()} $$\n", out)
    return _INLINE_MATH.sub(lambda m: f"${m.group(1).strip()}$", out)


def latency_rows(lat: dict[str, float]) -> list[tuple[str, float]]:
    """Stages in pipeline order, in seconds."""
    return [(k, lat.get(k, 0.0) / 1000) for k in ("retrieve", "rerank", "generate")]


def build_payload(question: str, companies: str, years: str, top_k: int) -> dict:
    """Request body from the sidebar fields. Companies are comma separated, years too."""
    payload: dict = {"question": question.strip(), "top_k": top_k}
    comp = [c.strip() for c in companies.split(",") if c.strip()]
    yrs = [int(y) for y in re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", years)]  # also "FY2022"
    if comp or yrs:
        payload["filters"] = {"companies": comp or None, "years": yrs or None}
    return payload
