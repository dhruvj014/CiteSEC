from pathlib import Path

import pytest
from tests.fakes import FakeEmbedder, FakeReranker, build_tiny_index

from citesec.retrieval.hybrid import Retriever


@pytest.fixture
def retriever(tmp_path: Path) -> Retriever:
    build_tiny_index(tmp_path)
    return Retriever(tmp_path, FakeEmbedder(), FakeReranker())
