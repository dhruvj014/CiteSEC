"""Minimal Ollama chat client. Deterministic by default: temperature 0 and a fixed seed."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

import httpx

DEFAULT_HOST = "http://localhost:11434"


def normalize_host(host: str) -> str:
    """Ollama's own OLLAMA_HOST convention allows 'host:port' or '0.0.0.0:port' without a scheme."""
    host = host.strip().rstrip("/")
    if "://" not in host:
        host = "http://" + host
    if not re.search(r":\d+$", host):
        host += ":11434"
    return host.replace("://0.0.0.0", "://localhost")


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_ms: float
    load_ms: float


class OllamaClient:
    def __init__(self, model: str, host: str | None = None, temperature: float = 0.0,
                 seed: int = 42, num_ctx: int = 6144, num_predict: int = 700,
                 keep_alive: str = "30m", timeout: float = 300.0) -> None:
        self.model = model
        self.host = normalize_host(host or os.environ.get("OLLAMA_HOST") or DEFAULT_HOST)
        self.options = {"temperature": temperature, "seed": seed, "num_ctx": num_ctx,
                        "num_predict": num_predict}
        self.keep_alive = keep_alive
        self.client = httpx.Client(timeout=timeout)

    def chat(self, messages: list[dict[str, str]]) -> LLMResult:
        r = self.client.post(f"{self.host}/api/chat", json={
            "model": self.model, "messages": messages, "stream": False,
            "options": self.options, "keep_alive": self.keep_alive})
        r.raise_for_status()
        d = r.json()
        return LLMResult(
            text=d["message"]["content"].strip(),
            prompt_tokens=int(d.get("prompt_eval_count", 0)),
            completion_tokens=int(d.get("eval_count", 0)),
            total_ms=d.get("total_duration", 0) / 1e6,
            load_ms=d.get("load_duration", 0) / 1e6,
        )

    def is_up(self) -> bool:
        try:
            return self.client.get(f"{self.host}/api/tags", timeout=5).status_code == 200
        except httpx.HTTPError:
            return False

    def unload(self) -> None:
        """Free VRAM, e.g. before building indexes on the GPU."""
        self.client.post(f"{self.host}/api/generate",
                         json={"model": self.model, "keep_alive": 0}, timeout=30)

    @classmethod
    def from_config(cls, cfg) -> OllamaClient:
        g = cfg["generation"]
        return cls(cfg["models"]["generator"], temperature=g["temperature"], seed=g["seed"],
                   num_ctx=g["num_ctx"], num_predict=g["num_predict"],
                   keep_alive=g["keep_alive"])
