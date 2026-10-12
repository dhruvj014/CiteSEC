"""Token counting with character offsets.

Real runs use the embedding model's own tokenizer so the 512-token cap is exact. Unit tests use
RegexTokenizer so no model download is needed.
"""
from __future__ import annotations

import re
from typing import Protocol


class Tokenizer(Protocol):
    n_special: int  # special tokens the model adds around every input ([CLS], [SEP])

    def offsets(self, text: str) -> list[tuple[int, int]]:
        """Character span of every token, in order."""
        ...


_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


class RegexTokenizer:
    """Words and single punctuation marks. Roughly tracks WordPiece counts without a model."""

    n_special = 0

    def offsets(self, text: str) -> list[tuple[int, int]]:
        return [m.span() for m in _TOKEN_RE.finditer(text)]


class HFTokenizer:
    """Wraps a Hugging Face fast tokenizer (used for the real 512-token cap)."""

    def __init__(self, name: str) -> None:
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(name, use_fast=True)
        self.tok.model_max_length = 10**9  # we window ourselves, silence the length warning
        self.n_special = self.tok.num_special_tokens_to_add(pair=False)

    def offsets(self, text: str) -> list[tuple[int, int]]:
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        return [(a, b) for a, b in enc["offset_mapping"] if b > a]


def count_tokens(tok: Tokenizer, text: str) -> int:
    """Content tokens only, without the model's special tokens."""
    return len(tok.offsets(text)) if text else 0


def model_len(tok: Tokenizer, text: str) -> int:
    """Length the model actually sees, including special tokens. This is what the cap limits."""
    return count_tokens(tok, text) + tok.n_special
