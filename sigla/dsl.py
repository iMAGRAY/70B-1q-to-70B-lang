"""Minimal SIGLA DSL helpers."""

from typing import List, Dict, Any, Optional

try:
    import faiss  # type: ignore
except ImportError:
    faiss = None

from .core import CapsuleStore, merge_capsules, MissingDependencyError
from .graph import expand_with_links


def INTENT(store: CapsuleStore, text: str) -> str:
    """Extract intent from text and encode it (for test compatibility)."""
    # Simple intent extraction - return first few words
    words = text.split()[:3]
    return " ".join(words).lower()


def RETRIEVE(store: CapsuleStore, intent_vector: str, top_k: int = 5, tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Retrieve top capsules for a text query."""
    return store.query(intent_vector, top_k=top_k, tags=tags)


def MERGE(capsules: List[Dict[str, Any]], temperature: float = 1.0) -> str:
    """Merge several capsules into a single text snippet."""
    return merge_capsules(capsules, temperature=temperature)


def INJECT(text: str, store: Optional[CapsuleStore] = None) -> str:
    """Inject text as a new capsule and return its ID."""
    if store:
        store.add_capsule(text)
    return text


def EXPAND(capsule: Dict[str, Any], store: CapsuleStore, depth: int = 1, limit: int = 10) -> List[Dict[str, Any]]:
    """Expand capsule via its links."""
    return expand_with_links([capsule], store, depth=depth, limit=limit)
