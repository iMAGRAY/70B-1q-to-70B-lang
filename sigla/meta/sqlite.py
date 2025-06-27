from __future__ import annotations

"""SQLite backend for SIGLA meta storage."""

import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Sequence, cast

from .base import MetaStore


class SQLiteMetaStore(MetaStore):
    """Persistent meta-store using builtin sqlite3.

    Notes
    -----
    • The interface is *naïve* and optimised for simplicity rather than raw
      throughput.  For massive graphs a columnar database (DuckDB) is advised.
    • A single table ``capsules(id INTEGER PRIMARY KEY, meta TEXT NOT NULL)``
      stores JSON-serialised metadata.
    """

    def __init__(self, db_path: str | Path):
        self.path = Path(db_path)
        self.conn = sqlite3.connect(self.path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS capsules (id INTEGER PRIMARY KEY, meta TEXT NOT NULL)"
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    # MetaStore interface
    # ------------------------------------------------------------------

    def add_many(self, metas: List[Dict[str, Any]]) -> List[int]:
        cur = self.conn.cursor()
        ids: List[int] = []
        for meta in metas:
            meta_json = json.dumps(meta, ensure_ascii=False)
            cur.execute("INSERT INTO capsules(meta) VALUES (?)", (meta_json,))
            assigned_id = cast(int, cur.lastrowid)
            ids.append(assigned_id)
            meta["id"] = assigned_id  # reflect back to caller
        self.conn.commit()
        return ids

    def all(self) -> Sequence[Dict[str, Any]]:
        cur = self.conn.execute("SELECT id, meta FROM capsules ORDER BY id")
        return [self._row_to_meta(row) for row in cur.fetchall()]

    def get(self, idx: int) -> Dict[str, Any]:  # type: ignore[override]
        cur = self.conn.execute("SELECT id, meta FROM capsules WHERE id=?", (idx,))
        row = cur.fetchone()
        if row is None:
            raise IndexError(idx)
        return self._row_to_meta(row)

    def update(self, idx: int, meta: Dict[str, Any]) -> None:  # type: ignore[override]
        meta_json = json.dumps(meta, ensure_ascii=False)
        self.conn.execute("UPDATE capsules SET meta=? WHERE id=?", (meta_json, idx))
        self.conn.commit()

    def remove(self, ids: List[int]) -> int:  # type: ignore[override]
        if not ids:
            return 0
        q_marks = ",".join(["?"] * len(ids))
        self.conn.execute(f"DELETE FROM capsules WHERE id IN ({q_marks})", ids)
        self.conn.commit()
        # rebuild contiguous IDs (simple reindex)
        self._reindex()
        return len(ids)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_meta(row):
        idx, meta_json = row
        meta = json.loads(meta_json)
        meta["id"] = int(idx)
        return meta

    def _reindex(self):
        # SQLite doesn't support AUTO_INCREMENT reset easily; we rebuild table.
        cur = self.conn.execute("SELECT meta FROM capsules ORDER BY id")
        metas = [json.loads(r[0]) for r in cur.fetchall()]
        self.conn.execute("DELETE FROM capsules")
        self.conn.execute("DELETE FROM sqlite_sequence WHERE name='capsules'")
        self.conn.commit()
        self.add_many(metas)

    def _replace_all(self, new_data: List[Dict[str, Any]]):  # type: ignore[override]
        self.conn.execute("DELETE FROM capsules")
        self.conn.execute("DELETE FROM sqlite_sequence WHERE name='capsules'")
        self.conn.commit()
        self.add_many(new_data)

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def __del__(self):  # pragma: no cover
        try:
            self.conn.close()
        except Exception:
            pass
