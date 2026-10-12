"""Optional query analysis: detect the company and fiscal years in a question.

Used only in the "filtered" mode. The primary evaluation is open retrieval with no filter, because
detecting the company from the question text is itself an assumption (it fails on questions that
omit the company or use an unusual name).
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from citesec.ingest.edgar import norm_name

# Extra names for FinanceBench companies, keyed by the exact `company` string in the dataset.
# Lowercase entries match case-insensitively. Uppercase tickers (3+ letters) match case-sensitively.
FB_ALIASES: dict[str, list[str]] = {
    "3M": ["3m", "MMM"],
    "AES Corporation": ["aes"],
    "AMD": ["amd", "advanced micro devices"],
    "Activision Blizzard": ["activision", "blizzard", "ATVI"],
    "Adobe": ["adobe", "ADBE"],
    "Amazon": ["amazon", "AMZN"],
    "Amcor": ["amcor"],
    "American Express": ["american express", "amex", "AXP"],
    "American Water Works": ["american water works", "american water"],
    "Best Buy": ["best buy", "BBY"],
    "Block": ["block, inc", "block inc", "square"],
    "Boeing": ["boeing"],
    "CVS Health": ["cvs health", "CVS"],
    "Coca-Cola": ["coca-cola", "coca cola", "coke"],
    "Corning": ["corning", "GLW"],
    "Costco": ["costco", "COST"],
    "Foot Locker": ["foot locker"],
    "General Mills": ["general mills", "GIS"],
    "JPMorgan": ["jpmorgan", "jp morgan", "JPM"],
    "Johnson & Johnson": ["johnson & johnson", "j&j", "jnj", "JNJ"],
    "Kraft Heinz": ["kraft heinz", "KHC"],
    "Lockheed Martin": ["lockheed martin", "lockheed", "LMT"],
    "MGM Resorts": ["mgm resorts", "mgm"],
    "Microsoft": ["microsoft", "MSFT"],
    "Netflix": ["netflix", "NFLX"],
    "Nike": ["nike", "NKE"],
    "Paypal": ["paypal", "PYPL"],
    "PepsiCo": ["pepsico", "pepsi"],
    "Pfizer": ["pfizer", "PFE"],
    "Ulta Beauty": ["ulta beauty", "ulta"],
    "Verizon": ["verizon"],
    "Walmart": ["walmart", "WMT"],
}
YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
FY2_RE = re.compile(r"\bFY\s?'?(\d{2})\b", re.IGNORECASE)


@dataclass(frozen=True)
class QueryFilter:
    companies: frozenset[str] | None = None  # exact chunk.company values, None means no filter
    years: frozenset[int] | None = None  # allowed doc years, None means no filter

    @property
    def active(self) -> bool:
        return self.companies is not None or self.years is not None


class AliasTable:
    """alias -> company names, from the FinanceBench aliases plus every company in the index."""

    def __init__(self, companies: dict[str, str | None]) -> None:
        """`companies` maps a company name to its ticker (or None)."""
        self.lower: dict[str, set[str]] = defaultdict(set)
        self.tickers: dict[str, set[str]] = defaultdict(set)
        for name, ticker in companies.items():
            self._add(name, name)
            if ticker and len(ticker) >= 3:
                self.tickers[ticker].add(name)
        for name, aliases in FB_ALIASES.items():
            if name in companies:
                for a in aliases:
                    self._add(a, name)

    def _add(self, alias: str, name: str) -> None:
        if alias.isupper() and alias.isalpha() and len(alias) >= 3 and alias != name.upper():
            self.tickers[alias].add(name)
            return
        a = norm_name(alias) if alias == name else alias.lower()
        if len(a) >= 2:  # keeps "3m"; the alias lists are curated and matched on word boundaries
            self.lower[a].add(name)

    def find_companies(self, question: str) -> set[str]:
        q = " " + re.sub(r"[^a-z0-9&,\-\. ]", " ", question.lower()) + " "
        found: set[str] = set()
        for alias, names in self.lower.items():
            if re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", q):
                found |= names
        for tick, names in self.tickers.items():
            if re.search(rf"\b{re.escape(tick)}\b", question):
                found |= names
        return found


def find_years(question: str) -> set[int]:
    """Four-digit years plus two-digit fiscal years such as FY22 or FY'21 (read as 20yy)."""
    years = {int(m) for m in YEAR_RE.findall(question)}
    years |= {2000 + int(m) for m in FY2_RE.findall(question)}
    return years


def parse_query(question: str, aliases: AliasTable) -> QueryFilter:
    """Company filter if a known company is named. Year filter if years are mentioned.

    A mentioned year y also admits y+1, because a comparison question about FY y can be answered
    from the filing for y+1, which reports both.
    """
    companies = aliases.find_companies(question)
    years = find_years(question)
    return QueryFilter(
        companies=frozenset(companies) if companies else None,
        years=frozenset(years | {y + 1 for y in years}) if years else None,
    )


class ChunkMeta:
    """Per-chunk company and year arrays aligned with the index rows, for fast filtering."""

    def __init__(self, companies: list[str], periods: list[str]) -> None:
        self.names = sorted(set(companies))
        lookup = {n: i for i, n in enumerate(self.names)}
        self.company_idx = np.array([lookup[c] for c in companies], dtype=np.int32)
        self.lookup = lookup
        self.years = np.array([_year(p) for p in periods], dtype=np.int32)

    def allowed(self, f: QueryFilter) -> np.ndarray | None:
        """Row positions passing the filter, or None when the filter is inactive."""
        if not f.active:
            return None
        mask = np.ones(len(self.years), dtype=bool)
        if f.companies is not None:
            ids = [self.lookup[c] for c in f.companies if c in self.lookup]
            mask &= np.isin(self.company_idx, ids)
        if f.years is not None:
            mask &= np.isin(self.years, list(f.years))
        return np.flatnonzero(mask)


def _year(period: str) -> int:
    m = YEAR_RE.search(period or "")
    return int(m.group(1)) if m else -1
