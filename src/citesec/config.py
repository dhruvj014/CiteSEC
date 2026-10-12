"""Typed access to configs/default.yaml and repo paths."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


class Config:
    """Thin wrapper over the YAML dict with resolved paths."""

    def __init__(self, raw: dict[str, Any], root: Path = REPO_ROOT) -> None:
        self.raw = raw
        self.root = root

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    @property
    def seed(self) -> int:
        return int(self.raw["seed"])

    def path(self, name: str) -> Path:
        return self.root / self.raw["paths"][name]

    @property
    def fb_dir(self) -> Path:
        return self.path("raw_financebench")

    @property
    def edgar_dir(self) -> Path:
        return self.path("raw_edgar")

    @property
    def processed_dir(self) -> Path:
        return self.path("processed")

    @property
    def results_dir(self) -> Path:
        return self.path("results")


@lru_cache(maxsize=1)
def load_config(path: Path = DEFAULT_CONFIG) -> Config:
    with open(path, encoding="utf-8") as f:
        return Config(yaml.safe_load(f))
