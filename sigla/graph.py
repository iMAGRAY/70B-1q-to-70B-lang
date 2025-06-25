from __future__ import annotations

from typing import Dict, List
import random

from .core import CapsuleStore


def expand_with_links(
    capsules: List[dict],
    store: CapsuleStore,
    depth: int = 1,
    limit: int = 10,
) -> List[dict]:
    """Breadth-first расширение капсул по их полю ``links``.

    - ``depth`` – сколько «слоёв» ссылок пройти.
    - ``limit`` – максимальное количество капсул в результате.
    """
    visited = {c["id"] for c in capsules}
    queue = list(visited)
    results: List[dict] = list(capsules)

    for _ in range(depth):
        new_queue: List[int] = []
        for cid in queue:
            meta = store.meta[cid]
            for link in meta.get("links", []):
                if (
                    link in visited
                    or link < 0
                    or link >= len(store.meta)
                    or len(results) >= limit
                ):
                    continue
                visited.add(link)
                linked = store.meta[link].copy()
                linked.update({"score": 0.0, "id": link})
                results.append(linked)
                new_queue.append(link)
                if len(results) >= limit:
                    break
        queue = new_queue
        if not queue or len(results) >= limit:
            break
    return results


def random_walk_links(
    capsules: List[dict],
    store: CapsuleStore,
    steps: int = 3,
    restart: float = 0.5,
    limit: int = 10,
) -> List[dict]:
    """Random-walk расширение капсул (walk with restart)."""
    if not capsules:
        return []

    start = [c["id"] for c in capsules]
    visited: Dict[int, int] = {cid: 1 for cid in start}
    current = list(start)

    for _ in range(steps):
        next_nodes: List[int] = []
        for cid in current:
            links = store.meta[cid].get("links", [])
            if links and random.random() > restart:
                next_nodes.append(random.choice(links))
            else:
                next_nodes.append(random.choice(start))
        current = next_nodes
        for cid in current:
            visited[cid] = visited.get(cid, 0) + 1

    # Отбираем наиболее часто посещённые узлы
    results: List[dict] = []
    for cid, count in sorted(visited.items(), key=lambda x: -x[1])[:limit]:
        meta = store.meta[cid].copy()
        meta.update({"score": float(count), "id": cid})
        results.append(meta)
    return results
