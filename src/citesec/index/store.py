"""SQLite chunk store: fast chunk_id to chunk lookup and ordered iteration."""
from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Iterator
from pathlib import Path

from citesec.schemas import Chunk

COLUMNS = [
    "chunk_id", "doc_name", "company", "doc_type", "period", "page", "section",
    "text", "is_table", "n_tokens", "header",
]


class ChunkStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.db = sqlite3.connect(path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS chunks ("
            "pos INTEGER PRIMARY KEY, chunk_id TEXT UNIQUE, doc_name TEXT, company TEXT, "
            "doc_type TEXT, period TEXT, page INTEGER, section TEXT, text TEXT, "
            "is_table INTEGER, n_tokens INTEGER, header TEXT)"
        )
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_doc ON chunks(doc_name)")

    def reset(self) -> None:
        self.db.execute("DELETE FROM chunks")
        self.db.commit()

    def add(self, chunks: Iterable[Chunk]) -> None:
        start = self.count()
        rows = [
            (start + i, *[int(getattr(c, k)) if k == "is_table" else getattr(c, k)
                          for k in COLUMNS])
            for i, c in enumerate(chunks)
        ]
        self.db.executemany(
            f"INSERT INTO chunks (pos, {', '.join(COLUMNS)}) VALUES (?{',?' * len(COLUMNS)})",
            rows,
        )
        self.db.commit()

    def count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    @staticmethod
    def _row(r: tuple) -> Chunk:
        d = dict(zip(COLUMNS, r, strict=True))
        d["is_table"] = bool(d["is_table"])
        return Chunk(**d)

    def iter_all(self) -> Iterator[Chunk]:
        """Chunks in insertion order, which is the row order of the vector indexes."""
        cur = self.db.execute(f"SELECT {', '.join(COLUMNS)} FROM chunks ORDER BY pos")
        for r in cur:
            yield self._row(r)

    def company_period(self) -> tuple[list[str], list[str]]:
        """Company and period of every chunk in index row order, for metadata filtering."""
        rows = self.db.execute("SELECT company, period FROM chunks ORDER BY pos").fetchall()
        return [r[0] for r in rows], [r[1] for r in rows]

    def get(self, chunk_id: str) -> Chunk | None:
        r = self.db.execute(
            f"SELECT {', '.join(COLUMNS)} FROM chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        return self._row(r) if r else None

    def get_many(self, chunk_ids: list[str]) -> list[Chunk]:
        """Preserves the order of `chunk_ids`."""
        out: dict[str, Chunk] = {}
        for i in range(0, len(chunk_ids), 500):
            part = chunk_ids[i : i + 500]
            marks = ",".join("?" * len(part))
            q = f"SELECT {', '.join(COLUMNS)} FROM chunks WHERE chunk_id IN ({marks})"
            for r in self.db.execute(q, part):
                c = self._row(r)
                out[c.chunk_id] = c
        return [out[i] for i in chunk_ids if i in out]

    def close(self) -> None:
        self.db.close()
