from __future__ import annotations

"""Meta-storage backends for SIGLA.

A *meta-store* keeps arbitrary capsule metadata (dict) keyed by an integer
identifier.  The default implementation is purely in-memory but richer
backends (SQLite, DuckDB, etc.) can be plugged in transparently.

Only a subset of CRUD operations are currently required by the core code:
    • add_many – bulk insert returning assigned IDs  
    • all       – return live list/sequence of all meta dicts (ordered by id)

The remaining helpers (get, update, remove) are provided with sensible default
implementations but can be overridden for performance.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Sequence


class MetaStore(ABC):
    """Abstract base-class for capsule meta-data storage backends."""

    # ------------------------------------------------------------------
    # CRUD interface (minimal)
    # ------------------------------------------------------------------

    @abstractmethod
    def add_many(self, metas: List[Dict[str, Any]]) -> List[int]:
        """Insert *metas* and return the list of assigned integer IDs."""

    @abstractmethod
    def all(self) -> Sequence[Dict[str, Any]]:
        """Return **live** sequence of all stored meta dicts (ordered by id)."""

    # ------------------------------------------------------------------
    # Optional helpers (can be overridden)
    # ------------------------------------------------------------------

    def get(self, idx: int) -> Dict[str, Any]:  # pragma: no cover – trivial
        return self.all()[idx]

    def update(self, idx: int, meta: Dict[str, Any]) -> None:  # pragma: no cover
        coll = list(self.all())
        coll[idx] = meta

    def remove(self, ids: List[int]) -> int:  # pragma: no cover – slow fallback
        """Remove capsules by id.  Returns number of removed rows."""
        if not ids:
            return 0
        ids_set = set(ids)
        remaining = [m for m in self.all() if m.get("id") not in ids_set]
        # rebuild contiguous ids
        for new_id, meta in enumerate(remaining):
            meta["id"] = new_id
        # replace contents (works for in-memory list) – subclasses may override
        self._replace_all(remaining)
        return len(ids_set)

    # ------------------------------------------------------------------
    # Internal helpers – subclasses can override for efficiency
    # ------------------------------------------------------------------

    def _replace_all(self, new_data: List[Dict[str, Any]]):
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def count(self) -> int:
        return len(self.all())


# ---------------------------------------------------------------------------
# In-memory backend (default).  Very light wrapper around Python list so that
# CapsuleStore keeps backward compatibility with the old ``store.meta`` list.
# ---------------------------------------------------------------------------

class InMemoryMetaStore(MetaStore):
    """Simple list-based meta store (no persistence)."""

    def __init__(self, storage: List[Dict[str, Any]] | None = None):
        # Use external list if provided so references stay the same.
        self._meta: List[Dict[str, Any]] = storage if storage is not None else []

    # --- MetaStore interface ------------------------------------------------

    def add_many(self, metas: List[Dict[str, Any]]) -> List[int]:
        start_id = len(self._meta)
        for i, meta in enumerate(metas):
            # auto-assign id if missing
            meta.setdefault("id", start_id + i)
            self._meta.append(meta)
        return [m["id"] for m in metas]

    def all(self) -> List[Dict[str, Any]]:
        return self._meta

    # --- internal ---

    def _replace_all(self, new_data: List[Dict[str, Any]]):
        # replace in-place, preserving object identity where possible
        self._meta.clear()
        self._meta.extend(new_data) 