"""Cross-encoder reranker (bge-reranker-base)."""
from __future__ import annotations

from typing import Protocol


class RerankerLike(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]: ...


class Reranker:
    """Scores (query, passage) pairs. Higher is more relevant. Scores are sigmoid values in [0, 1].

    Pairs are truncated to `max_length` tokens, so the tail of a very long passage is not seen.
    """

    def __init__(self, model_name: str, device: str = "cuda", fp16: bool = True,
                 batch_size: int = 16, max_length: int = 512) -> None:
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, device=device, max_length=max_length)
        if fp16 and device.startswith("cuda"):
            self.model.model.half()
        self.batch_size = batch_size

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        pairs = [(query, t) for t in texts]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        return [float(s) for s in scores]
