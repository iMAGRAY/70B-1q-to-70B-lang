"""Module registry for SIGLA.

A *module* is a reusable capsulegraph or skill package that can be routed
based on *intent* / *tags*.

Registry stores minimal metadata: ``name`` (unique), ``path`` (file or URI),
``tags`` list, ``added`` timestamp.

The preferred backend is **Neo4j** via Bolt protocol (``neo4j-driver``).
When the driver is not available the registry transparently falls back to a
simple **SQLite** file in the workspace directory so that SIGLA remains
fully functional without external services.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Sequence

try:
    from neo4j import GraphDatabase  # type: ignore

    _NEO_AVAILABLE = True
except ImportError:  # pragma: no cover – optional dep
    GraphDatabase = None  # type: ignore
    _NEO_AVAILABLE = False

__all__ = [
    "RegistryEntry",
    "ModuleRegistry",
]


@dataclass
class RegistryEntry:
    name: str
    path: str
    tags: List[str]
    added: float

    @classmethod
    def from_row(cls, row):  # type: ignore[override]
        if isinstance(row, (list, tuple)):
            name, path, tags_json, added = row
        else:  # neo4j record
            name = row["name"]
            path = row["path"]
            tags_json = row["tags"]
            added = row["added"]
        tags = json.loads(tags_json) if isinstance(tags_json, str) else tags_json
        return cls(name, path, tags, float(added))


# ---------------------------------------------------------------------------
# SQLite fallback (file registry.db in workspace): convenient for local tests
# ---------------------------------------------------------------------------

class _SQLiteBackend:
    def __init__(self, db_path: str | Path = "registry.db"):
        self.path = Path(db_path)
        self.conn = sqlite3.connect(self.path)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS modules (
                name TEXT PRIMARY KEY,
                path TEXT NOT NULL,
                tags TEXT NOT NULL,
                added REAL NOT NULL
            )
            """
        )
        self.conn.commit()

    # --- CRUD -------------------------------------------------------------
    def add(self, entry: RegistryEntry):
        self.conn.execute(
            "INSERT OR REPLACE INTO modules(name, path, tags, added) VALUES (?,?,?,?)",
            (entry.name, entry.path, json.dumps(entry.tags, ensure_ascii=False), entry.added),
        )
        self.conn.commit()

    def remove(self, name: str) -> int:
        cur = self.conn.execute("DELETE FROM modules WHERE name=?", (name,))
        self.conn.commit()
        return cur.rowcount

    def list(self) -> List[RegistryEntry]:
        cur = self.conn.execute("SELECT name, path, tags, added FROM modules ORDER BY added DESC")
        return [RegistryEntry.from_row(r) for r in cur.fetchall()]


# ---------------------------------------------------------------------------
# Neo4j backend
# ---------------------------------------------------------------------------

class _Neo4jBackend:
    def __init__(self, uri: str, user: str, password: str):
        if GraphDatabase is None:
            raise RuntimeError("neo4j-driver not installed")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    # --- helpers ----------------------------------------------------------
    def _run(self, query: str, **params):
        with self.driver.session() as sess:
            return list(sess.run(query, **params))

    # --- CRUD -------------------------------------------------------------
    def add(self, entry: RegistryEntry):
        self._run(
            """
            MERGE (m:Module {name:$name})
            SET m.path=$path, m.tags=$tags, m.added=$added
            """,
            name=entry.name,
            path=entry.path,
            tags=entry.tags,
            added=entry.added,
        )

    def remove(self, name: str) -> int:
        result = self._run("MATCH (m:Module {name:$name}) DETACH DELETE m RETURN COUNT(*) AS c", name=name)
        return result[0]["c"] if result else 0

    def list(self) -> List[RegistryEntry]:
        records = self._run("MATCH (m:Module) RETURN m.name AS name, m.path AS path, m.tags AS tags, m.added AS added ORDER BY m.added DESC")
        return [RegistryEntry.from_row(rec) for rec in records]


# ---------------------------------------------------------------------------
# Public facade – chooses backend automatically
# ---------------------------------------------------------------------------

class ModuleRegistry:
    """Facade for registry backends.

    When the environment variable ``SIGLA_NEO4J_URI`` is set the registry will
    try to connect to Neo4j; otherwise a local SQLite DB is used.
    """

    def __init__(self):
        uri = os.getenv("SIGLA_NEO4J_URI")
        if uri and _NEO_AVAILABLE:
            user = os.getenv("SIGLA_NEO4J_USER", "neo4j")
            password = os.getenv("SIGLA_NEO4J_PASS", "neo4j")
            self._backend = _Neo4jBackend(uri, user, password)
        else:
            self._backend = _SQLiteBackend()

    # --- facade methods --------------------------------------------------

    def add_module(self, name: str, path: str, tags: Optional[Sequence[str]] = None):
        entry = RegistryEntry(name=name, path=path, tags=list(tags or []), added=time.time())
        self._backend.add(entry)

    def remove_module(self, name: str) -> int:
        return self._backend.remove(name)

    def list_modules(self) -> List[RegistryEntry]:
        return self._backend.list() 