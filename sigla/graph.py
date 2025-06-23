from __future__ import annotations

3szrfh-codex/разработать-sigla-для-моделирования-мышления
from typing import List, Dict
import random
=======
xvy4pj-codex/разработать-sigla-для-моделирования-мышления
from typing import List
=======
from typing import List, Dict
import random
main
main

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
3szrfh-codex/разработать-sigla-для-моделирования-мышления
=======
xvy4pj-codex/разработать-sigla-для-моделирования-мышления
=======
main


def random_walk_links(
    capsules: List[dict],
    store: CapsuleStore,
    steps: int = 3,
    restart: float = 0.5,
    limit: int = 10,
) -> List[dict]:
    """Expand capsules via random walk with restart."""
    if not capsules:
        return []

    start = [c["id"] for c in capsules]
    visited: Dict[int, int] = {cid: 1 for cid in start}
    current = list(start)

    for _ in range(steps):
        next_nodes = []
        for cid in current:
            links = store.meta[cid].get("links", [])
            if links and random.random() > restart:
                next_nodes.append(random.choice(links))
            else:
                next_nodes.append(random.choice(start))
        current = next_nodes
        for cid in current:
            visited[cid] = visited.get(cid, 0) + 1

    results = []
    for cid, count in sorted(visited.items(), key=lambda x: -x[1])[:limit]:
        meta = store.meta[cid].copy()
        meta["score"] = float(count)
        meta["id"] = cid
        results.append(meta)
    return results
3szrfh-codex/разработать-sigla-для-моделирования-мышления
=======
main
main
