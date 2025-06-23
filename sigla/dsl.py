"""Minimal SIGLA DSL helpers."""

from typing import List

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    faiss = None

from .core import CapsuleStore, merge_capsules, MissingDependencyError
from .graph import expand_with_links


def INTENT(store: CapsuleStore, text: str):
    """Embed a query into a vector."""
    if faiss is None:
        raise MissingDependencyError("faiss package is required for INTENT")
    return store.embed_query(text)


def RETRIEVE(store: CapsuleStore, vector, top_k: int = 5, tags: List[str] | None = None):
    """Retrieve top capsules given an embedded query."""
    if faiss is None:
        raise MissingDependencyError("faiss package is required for RETRIEVE")
    scores, indices = store.index.search(vector, top_k * 5)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        meta = store.meta[idx]
        if tags and not set(tags).intersection(meta.get("tags", [])):
            continue
        cap = meta.copy()
        cap["score"] = float(score)
        results.append(cap)
        if len(results) >= top_k:
            break
    return results


def MERGE(capsules: List[dict]):
    """Merge several capsules into a single text snippet."""
    return merge_capsules(capsules)


def INJECT(composite: str) -> str:
    """Return prompt fragment to inject into 1Q."""
    return f"[Контекст]: \"{composite}\""


def EXPAND(capsules: List[dict], store: CapsuleStore, depth: int = 1, limit: int = 10):
    """Expand capsules via their links."""
    return expand_with_links(capsules, store, depth=depth, limit=limit)
