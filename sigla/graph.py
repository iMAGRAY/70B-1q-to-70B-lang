from __future__ import annotations

from typing import List

from .core import CapsuleStore


def expand_with_links(capsules: List[dict], store: CapsuleStore, depth: int = 1, limit: int = 10) -> List[dict]:
    """Expand capsules by following their 'links' metadata."""
    visited = {c["id"] for c in capsules}
    queue = list(visited)
    results = list(capsules)
    for _ in range(depth):
        new_queue = []
        for cid in queue:
            meta = store.meta[cid]
            for link in meta.get("links", []):
                if link in visited or link < 0 or link >= len(store.meta):
                    continue
                visited.add(link)
                linked = store.meta[link].copy()
                linked["score"] = 0.0
                linked["id"] = link
                results.append(linked)
                new_queue.append(link)
                if len(results) >= limit:
                    return results
        queue = new_queue
    return results
