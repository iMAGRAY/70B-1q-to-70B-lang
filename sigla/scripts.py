import argparse
import json
import time
from pathlib import Path

from .core import (
    CapsuleStore,
    merge_capsules,
    MissingDependencyError,
    capsulate_text,
    clear_summarizer_cache,
)
from .dsl import INJECT
from .graph import expand_with_links
from . import log as siglog


def ingest(
    json_file: str,
    index_path: str,
    model: str,
    factory: str,
    link_neighbors: int,
    tags: list[str] | None = None,
    source: str | None = None,
    rating: float | None = None,
    dedup: bool = True,
):
    try:
        if Path(index_path + ".index").exists():
            store = CapsuleStore()
            store.load(index_path)
        else:
            store = CapsuleStore(model_name=model, index_factory=factory)
    except MissingDependencyError as e:
        print(f"error: {e}")
        return
    with open(json_file, "r", encoding="utf-8") as f:
        text = f.read()
        if text.lstrip().startswith("["):
            capsules = json.loads(text)
        else:
            capsules = [
                json.loads(line)
                for line in text.splitlines()
                if line.strip()
            ]
    if tags or source or rating is not None:
        for cap in capsules:
            if tags:
                cap_tags = cap.get("tags", [])
                cap["tags"] = sorted(set(cap_tags).union(tags))
            if source:
                cap.setdefault("metadata", {})["source"] = source
            if rating is not None and "rating" not in cap:
                cap.setdefault("metadata", {})["rating"] = rating
    start = time.time()
    added = store.add_capsules(capsules, link_neighbors=link_neighbors, dedup=dedup)
    duration = time.time() - start
    store.save(index_path)
    print(f"added {added} capsules")
    siglog.record("ingest", start, count=added)


def update_capsules(
    json_file: str,
    index_path: str,
    link_neighbors: int,
    tags: list[str] | None = None,
    source: str | None = None,
    rating: float | None = None,
    dedup: bool = True,
) -> None:
    """Append capsules to an existing index."""
    try:
        store = CapsuleStore()
        store.load(index_path)
    except MissingDependencyError as e:
        print(f"error: {e}")
        return
    with open(json_file, "r", encoding="utf-8") as f:
        text = f.read()
        if text.lstrip().startswith("["):
            capsules = json.loads(text)
        else:
            capsules = [
                json.loads(line)
                for line in text.splitlines()
                if line.strip()
            ]
    if tags or source or rating is not None:
        for cap in capsules:
            if tags:
                cap_tags = cap.get("tags", [])
                cap["tags"] = sorted(set(cap_tags).union(tags))
            if source:
                cap.setdefault("metadata", {})["source"] = source
            if rating is not None and "rating" not in cap:
                cap.setdefault("metadata", {})["rating"] = rating
    start = time.time()
    added = store.add_capsules(capsules, link_neighbors=link_neighbors, dedup=dedup)
    store.save(index_path)
    print(f"added {added} capsules")
    siglog.record("update", start, count=added)


def capsulate_file(
    input_file: str,
    output_file: str,
    tags: list[str] | None = None,
    source: str | None = None,
) -> None:
    """Convert raw text into capsule JSON.

    ``input_file`` or ``output_file`` can be ``-`` to denote standard
    input or output.
    """
    if input_file == "-":
        import sys

        text = sys.stdin.read()
    else:
        with open(input_file, "r", encoding="utf-8") as f:
            text = f.read()
    capsules = capsulate_text(text, tags, source)
    if output_file == "-":
        json.dump(capsules, sys.stdout, ensure_ascii=False, indent=2)
        if not sys.stdout.isatty():
            sys.stdout.write("\n")
    else:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(capsules, f, ensure_ascii=False, indent=2)
    print(f"wrote {len(capsules)} capsules")


def search(
    index_path: str,
    query: str,
    top_k: int,
    tags: list[str] | None = None,
    sources: list[str] | None = None,
    min_rating: float = 0.0,
):
    try:
        store = CapsuleStore()
        store.load(index_path)
    except MissingDependencyError as e:
        print(f"error: {e}")
        return
    start = time.time()
    results = store.query(
        query, top_k=top_k, tags=tags, sources=sources, min_rating=min_rating
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))
    siglog.record(
        "search",
        start,
        query=query,
        top_k=top_k,
        tags=tags,
        sources=sources,
        results=results,
        min_rating=min_rating,
    )


def inject_snippet(
    index_path: str,
    query: str,
    top_k: int,
    tags: list[str] | None = None,
    sources: list[str] | None = None,
    temperature: float = 1.0,
    min_rating: float = 0.0,
) -> None:
    try:
        store = CapsuleStore()
        store.load(index_path)
    except MissingDependencyError as e:
        print(f"error: {e}")
        return
    start = time.time()
    results = store.query(
        query, top_k=top_k, tags=tags, sources=sources, min_rating=min_rating
    )
    snippet = INJECT(merge_capsules(results, temperature=temperature))
    print(snippet)
    siglog.record(
        "inject",
        start,
        query=query,
        top_k=top_k,
        tags=tags,
        sources=sources,
        temperature=temperature,
        min_rating=min_rating,
    )


def compress_snippet(
    index_path: str,
    query: str,
    top_k: int,
    tags: list[str] | None = None,
    sources: list[str] | None = None,
    model: str = "sshleifer/distilbart-cnn-12-6",
    min_rating: float = 0.0,
    max_length: int = 60,
    min_length: int = 5,
):
    """Retrieve capsules and summarize them."""
    try:
        store = CapsuleStore()
        store.load(index_path)
    except MissingDependencyError as e:
        print(f"error: {e}")
        return
    start = time.time()
    results = store.query(
        query, top_k=top_k, tags=tags, sources=sources, min_rating=min_rating
    )
    try:
        from .core import compress_capsules

        summary = compress_capsules(
            results,
            model_name=model,
            max_length=max_length,
            min_length=min_length,
        )
        print(summary)
    except MissingDependencyError as e:
        print(f"error: {e}")
    siglog.record(
        "compress",
        start,
        query=query,
        top_k=top_k,
        tags=tags,
        sources=sources,
        model=model,
        min_rating=min_rating,
        max_length=max_length,
        min_length=min_length,
    )


def walk_search(
    index_path: str,
    query: str,
    top_k: int,
    depth: int,
    limit: int,
    tags: list[str] | None = None,
    sources: list[str] | None = None,
    algo: str = "bfs",
    restart: float = 0.5,
    min_rating: float = 0.0,
):
    try:
        store = CapsuleStore()
        store.load(index_path)
    except MissingDependencyError as e:
        print(f"error: {e}")
        return
    start = time.time()
    results = store.query(
        query, top_k=top_k, tags=tags, sources=sources, min_rating=min_rating
    )
    if algo == "random":
        from .graph import random_walk_links

        expanded = random_walk_links(
            results, store, steps=depth, restart=restart, limit=limit
        )
    else:
        expanded = expand_with_links(results, store, depth=depth, limit=limit)
    print(json.dumps(expanded, ensure_ascii=False, indent=2))
    siglog.record(
        "walk",
        start,
        query=query,
        top_k=top_k,
        depth=depth,
        limit=limit,
        algo=algo,
        restart=restart,
        tags=tags,
        sources=sources,
        min_rating=min_rating,
    )


def show_capsule(index_path: str, idx: int) -> None:
    """Print a capsule by its id."""
    try:
        store = CapsuleStore(lazy=True)
        store.load(index_path)
    except MissingDependencyError as e:
        print(f"error: {e}")
        return
    if idx < 0 or idx >= len(store.meta):
        print("error: invalid id")
        return
    print(json.dumps(store.meta[idx], ensure_ascii=False, indent=2))


def list_capsules(
    index_path: str,
    limit: int = 20,
    tags: list[str] | None = None,
    sources: list[str] | None = None,
    min_rating: float = 0.0,
) -> None:
    """List capsules optionally filtered by tags, source and rating."""
    try:
        store = CapsuleStore(lazy=True)
        store.load(index_path)
    except MissingDependencyError as e:
        print(f"error: {e}")
        return
    results = []
    for meta in store.meta:
        if tags and not set(tags).intersection(meta.get("tags", [])):
            continue
        if sources:
            src = meta.get("metadata", {}).get("source")
            if src not in sources:
                continue
        rating = meta.get("rating") or meta.get("metadata", {}).get("rating") or 1.0
        if rating < min_rating:
            continue
        results.append(meta)
        if len(results) >= limit:
            break
    print(json.dumps(results, ensure_ascii=False, indent=2))


def shell(
    index_path: str,
    top_k: int,
    tags: list[str] | None = None,
    sources: list[str] | None = None,
    temperature: float = 1.0,
    min_rating: float = 0.0,
) -> None:
    """Run an interactive search loop printing merged context."""
    try:
        store = CapsuleStore()
        store.load(index_path)
    except MissingDependencyError as e:
        print(f"error: {e}")
        return
    print("Enter empty line to exit.")
    while True:
        try:
            query = input("query> ").strip()
        except EOFError:
            break
        if not query:
            break
        start = time.time()
        results = store.query(
            query, top_k=top_k, tags=tags, sources=sources, min_rating=min_rating
        )
        snippet = INJECT(merge_capsules(results, temperature=temperature))
        print(snippet)
        siglog.record(
            "shell",
            start,
            query=query,
            top_k=top_k,
            temperature=temperature,
            sources=sources,
            min_rating=min_rating,
        )


def show_stats(log_file: str) -> None:
    """Print counts and average duration per event type."""
    stats: dict[str, dict[str, float]] = {}
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                event = json.loads(line)
                etype = event.get("type", "unknown")
                info = stats.setdefault(etype, {"count": 0, "total": 0.0})
                info["count"] += 1
                info["total"] += float(event.get("duration", 0.0))
    except FileNotFoundError:
        print("error: log file not found")
        return
    summary = {
        t: {
            "count": int(v["count"]),
            "avg_duration": (v["total"] / v["count"]) if v["count"] else 0,
        }
        for t, v in stats.items()
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def show_info(index_path: str) -> None:
    """Print summary information about an index."""
    try:
        store = CapsuleStore(lazy=True)
        store.load(index_path)
    except MissingDependencyError as e:
        print(f"error: {e}")
        return
    info = {
        "model": store.model_name,
        "dimension": store.dimension,
        "capsules": len(store.meta),
    }
    tag_counts: dict[str, int] = {}
    for m in store.meta:
        for t in m.get("tags", []):
            tag_counts[t] = tag_counts.get(t, 0) + 1
    if tag_counts:
        info["tags"] = tag_counts
    print(json.dumps(info, ensure_ascii=False, indent=2))


def export_capsules(
    index_path: str,
    output_file: str,
    tags: list[str] | None = None,
    sources: list[str] | None = None,
    min_rating: float = 0.0,
) -> None:
    """Export capsules to a JSON file with optional filters."""
    try:
        store = CapsuleStore(lazy=True)
        store.load(index_path)
    except MissingDependencyError as e:
        print(f"error: {e}")
        return
    items = []
    for meta in store.meta:
        if tags and not set(tags).intersection(meta.get("tags", [])):
            continue
        if sources:
            src = meta.get("metadata", {}).get("source")
            if src not in sources:
                continue
        rating = meta.get("rating") or meta.get("metadata", {}).get("rating") or 1.0
        if rating < min_rating:
            continue
        items.append(meta)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"exported {len(items)} capsules")


def export_graph(
    index_path: str,
    output_file: str,
    limit: int | None = None,
    tags: list[str] | None = None,
) -> None:
    """Export capsule links as a Graphviz DOT file."""
    try:
        store = CapsuleStore(lazy=True)
        store.load(index_path)
    except MissingDependencyError as e:
        print(f"error: {e}")
        return
    from .graph import export_dot

    export_dot(store, output_file, limit=limit, tags=tags)
    print(f"graph written to {output_file}")


def prune_capsules(
    index_path: str, ids: list[int] | None = None, tags: list[str] | None = None
) -> None:
    """Remove capsules by id or tags and rebuild the index."""
    try:
        store = CapsuleStore()
        store.load(index_path)
    except MissingDependencyError as e:
        print(f"error: {e}")
        return
    remove_set: set[int] = set(ids or [])
    if tags:
        for i, meta in enumerate(store.meta):
            if set(tags).intersection(meta.get("tags", [])):
                remove_set.add(i)
    if not remove_set:
        print("no matching capsules")
        return
    start = time.time()
    removed = store.remove_capsules(sorted(remove_set))
    store.save(index_path)
    siglog.record(
        "prune",
        start,
        removed=removed,
        ids=sorted(remove_set),
        tags=tags,
    )
    print(f"removed {removed} capsules")


def reindex_store(
    index_path: str, model: str | None = None, factory: str | None = None
) -> None:
    """Rebuild embeddings for all capsules, optionally with a new model or index type."""
    try:
        store = CapsuleStore()
        store.load(index_path)
    except MissingDependencyError as e:
        print(f"error: {e}")
        return
    start = time.time()
    store.rebuild_index(model, factory)
    store.save(index_path)
    siglog.record(
        "reindex",
        start,
        model=model or store.model_name,
        factory=factory or store.index_factory,
    )
    print("index rebuilt")


def rate_capsules(
    index_path: str,
    rating: float,
    ids: list[int] | None = None,
    tags: list[str] | None = None,
) -> None:
    """Update the rating of capsules by id or tags."""
    try:
        store = CapsuleStore(lazy=True)
        store.load(index_path)
    except MissingDependencyError as e:
        print(f"error: {e}")
        return
    id_set: set[int] = set(ids or [])
    if tags:
        for i, meta in enumerate(store.meta):
            if set(tags).intersection(meta.get("tags", [])):
                id_set.add(i)
    if not id_set:
        print("no matching capsules")
        return
    start = time.time()
    updated = store.update_metadata(sorted(id_set), rating=rating)
    store.save(index_path)
    siglog.record(
        "rate",
        start,
        updated=updated,
        rating=rating,
        ids=sorted(id_set),
        tags=tags,
    )
    print(f"updated {updated} capsules")


def update_meta(
    index_path: str,
    rating: float | None = None,
    add_tags: list[str] | None = None,
    remove_tags: list[str] | None = None,
    ids: list[int] | None = None,
    tags: list[str] | None = None,
) -> None:
    """Modify capsule metadata by id or tag selection."""
    try:
        store = CapsuleStore(lazy=True)
        store.load(index_path)
    except MissingDependencyError as e:
        print(f"error: {e}")
        return
    id_set: set[int] = set(ids or [])
    if tags:
        for i, meta in enumerate(store.meta):
            if set(tags).intersection(meta.get("tags", [])):
                id_set.add(i)
    if not id_set:
        print("no matching capsules")
        return
    start = time.time()
    updated = store.update_metadata(
        sorted(id_set),
        rating=rating,
        add_tags=add_tags,
        remove_tags=remove_tags,
    )
    store.save(index_path)
    siglog.record(
        "meta",
        start,
        updated=updated,
        rating=rating,
        add=add_tags,
        remove=remove_tags,
        ids=sorted(id_set),
        tags=tags,
    )
    print(f"updated {updated} capsules")


def embed_text(
    text: str,
    model: str | None = None,
    index_path: str | None = None,
) -> None:
    """Print the embedding vector for a text.

    If ``index_path`` is provided, the store's model and dimension are used.
    Otherwise ``model`` specifies the sentence-transformer (or ``hash``).
    """
    try:
        if index_path:
            store = CapsuleStore(lazy=True)
            store.load(index_path)
        else:
            store = CapsuleStore(model_name=model or "sentence-transformers/all-MiniLM-L6-v2")
    except MissingDependencyError as e:
        print(f"error: {e}")
        return

    vector = store.embed_query(text)[0].tolist()
    print(json.dumps(vector))


def similarity_cmd(text_a: str, text_b: str, model: str) -> None:
    """Print cosine similarity between two texts."""
    try:
        store = CapsuleStore(model_name=model)
    except MissingDependencyError as e:
        print(f"error: {e}")
        return
    vectors = store.embed_texts([text_a, text_b])
    import numpy as np

    sim = float(np.dot(vectors[0], vectors[1]))
    print(f"{sim:.6f}")


def clear_cache_cmd(index_path: str | None, summarizer: bool, embeddings: bool) -> None:
    """Clear cached summarizer and/or embedding vectors."""
    if summarizer:
        clear_summarizer_cache()
    if embeddings and index_path:
        try:
            store = CapsuleStore(lazy=True)
            store.load(index_path)
            store.clear_cache()
        except MissingDependencyError as e:
            print(f"error: {e}")
            return
    print("cache cleared")


def main():
    parser = argparse.ArgumentParser(description="SIGLA utility")
    subparsers = parser.add_subparsers(dest="cmd")
    parser.add_argument("--log-file")

    caps_p = subparsers.add_parser("capsulate", help="split text into capsules")
    caps_p.add_argument("input_file")
    caps_p.add_argument("output_file")
    caps_p.add_argument("--tags")
    caps_p.add_argument("--source")

    ingest_p = subparsers.add_parser("ingest")
    ingest_p.add_argument("json_file")
    ingest_p.add_argument("index_path")
    ingest_p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ingest_p.add_argument("--factory", default="Flat", help="FAISS index factory")
    ingest_p.add_argument(
        "--link", type=int, default=0, help="auto-link each new capsule to N neighbors"
    )
    ingest_p.add_argument("--tags")
    ingest_p.add_argument("--source")
    ingest_p.add_argument("--rating", type=float)
    ingest_p.add_argument(
        "--no-dedup",
        action="store_false",
        dest="dedup",
        default=True,
        help="allow duplicate texts",
    )

    update_p = subparsers.add_parser(
        "update", help="append capsules to an existing index"
    )
    update_p.add_argument("json_file")
    update_p.add_argument("index_path")
    update_p.add_argument(
        "--link", type=int, default=0, help="auto-link each new capsule to N neighbors"
    )
    update_p.add_argument("--tags")
    update_p.add_argument("--source")
    update_p.add_argument("--rating", type=float)
    update_p.add_argument(
        "--no-dedup",
        action="store_false",
        dest="dedup",
        default=True,
        help="allow duplicate texts",
    )

    search_p = subparsers.add_parser("search")
    search_p.add_argument("index_path")
    search_p.add_argument("query")
    search_p.add_argument("--top_k", type=int, default=5)
    search_p.add_argument("--tags")
    search_p.add_argument("--sources")
    search_p.add_argument("--min-rating", type=float, default=0.0)

    inject_p = subparsers.add_parser("inject")
    inject_p.add_argument("index_path")
    inject_p.add_argument("query")
    inject_p.add_argument("--top_k", type=int, default=5)
    inject_p.add_argument("--tags")
    inject_p.add_argument("--sources")
    inject_p.add_argument("--temperature", type=float, default=1.0)
    inject_p.add_argument("--min-rating", type=float, default=0.0)

    compress_p = subparsers.add_parser("compress", help="summarize retrieved capsules")
    compress_p.add_argument("index_path")
    compress_p.add_argument("query")
    compress_p.add_argument("--top_k", type=int, default=5)
    compress_p.add_argument("--model", default="sshleifer/distilbart-cnn-12-6")
    compress_p.add_argument("--tags")
    compress_p.add_argument("--sources")
    compress_p.add_argument("--min-rating", type=float, default=0.0)
    compress_p.add_argument("--max-length", type=int, default=60)
    compress_p.add_argument("--min-length", type=int, default=5)

    walk_p = subparsers.add_parser("walk")
    walk_p.add_argument("index_path")
    walk_p.add_argument("query")
    walk_p.add_argument("--top_k", type=int, default=5)
    walk_p.add_argument("--depth", type=int, default=1)
    walk_p.add_argument("--limit", type=int, default=10)
    walk_p.add_argument("--algo", choices=["bfs", "random"], default="bfs")
    walk_p.add_argument(
        "--restart", type=float, default=0.5, help="restart prob for random walk"
    )
    walk_p.add_argument("--tags")
    walk_p.add_argument("--sources")
    walk_p.add_argument("--min-rating", type=float, default=0.0)

    cap_p = subparsers.add_parser("capsule")
    cap_p.add_argument("index_path")
    cap_p.add_argument("id", type=int)

    list_p = subparsers.add_parser("list", help="list capsules")
    list_p.add_argument("index_path")
    list_p.add_argument("--limit", type=int, default=20)
    list_p.add_argument("--tags")
    list_p.add_argument("--sources")
    list_p.add_argument("--min-rating", type=float, default=0.0)

    export_p = subparsers.add_parser("export", help="dump capsules to JSON")
    export_p.add_argument("index_path")
    export_p.add_argument("output_file")
    export_p.add_argument("--tags")
    export_p.add_argument("--sources")
    export_p.add_argument("--min-rating", type=float, default=0.0)

    graph_p = subparsers.add_parser("graph", help="export graph in DOT format")
    graph_p.add_argument("index_path")
    graph_p.add_argument("output_file")
    graph_p.add_argument("--limit", type=int)
    graph_p.add_argument("--tags")

    prune_p = subparsers.add_parser("prune", help="remove capsules")
    prune_p.add_argument("index_path")
    prune_p.add_argument("--ids")
    prune_p.add_argument("--tags")

    reindex_p = subparsers.add_parser("reindex", help="rebuild embeddings")
    reindex_p.add_argument("index_path")
    reindex_p.add_argument("--model")
    reindex_p.add_argument("--factory")

    rate_p = subparsers.add_parser("rate", help="update capsule rating")
    rate_p.add_argument("index_path")
    rate_p.add_argument("--rating", type=float, required=True)
    rate_p.add_argument("--ids")
    rate_p.add_argument("--tags")

    meta_p = subparsers.add_parser("meta", help="modify capsule metadata")
    meta_p.add_argument("index_path")
    meta_p.add_argument("--rating", type=float)
    meta_p.add_argument("--add-tags")
    meta_p.add_argument("--remove-tags")
    meta_p.add_argument("--ids")
    meta_p.add_argument("--tags")

    info_p = subparsers.add_parser("info", help="show index summary")
    info_p.add_argument("index_path")

    embed_p = subparsers.add_parser("embed", help="embed text and print vector")
    embed_p.add_argument("text")
    embed_p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    embed_p.add_argument("--index", help="load model settings from an existing index")

    sim_p = subparsers.add_parser("similarity", help="compute text similarity")
    sim_p.add_argument("text_a")
    sim_p.add_argument("text_b")
    sim_p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")

    shell_p = subparsers.add_parser("shell")
    shell_p.add_argument("index_path")
    shell_p.add_argument("--top_k", type=int, default=5)
    shell_p.add_argument("--tags")
    shell_p.add_argument("--sources")
    shell_p.add_argument("--temperature", type=float, default=1.0)
    shell_p.add_argument("--min-rating", type=float, default=0.0)

    stats_p = subparsers.add_parser("stats", help="summarize a log file")
    stats_p.add_argument("log_file")

    cache_p = subparsers.add_parser("cache", help="clear caches")
    cache_p.add_argument("index_path", nargs="?")
    cache_p.add_argument("--summarizer", action="store_true")
    cache_p.add_argument("--embeddings", action="store_true")

    args = parser.parse_args()
    if args.log_file and args.cmd != "stats":
        siglog.start(args.log_file)
    tags = args.tags.split(",") if hasattr(args, "tags") and args.tags else None
    sources = (
        args.sources.split(",") if hasattr(args, "sources") and args.sources else None
    )
    if args.cmd == "ingest":
        ingest(
            args.json_file,
            args.index_path,
            args.model,
            args.factory,
            args.link,
            tags,
            args.source,
            args.rating,
            args.dedup,
        )
    elif args.cmd == "update":
        update_capsules(
            args.json_file,
            args.index_path,
            args.link,
            tags,
            args.source,
            args.rating,
            args.dedup,
        )
    elif args.cmd == "capsulate":
        cap_tags = args.tags.split(",") if args.tags else None
        capsulate_file(args.input_file, args.output_file, cap_tags, args.source)
    elif args.cmd == "search":
        search(args.index_path, args.query, args.top_k, tags, sources, args.min_rating)
    elif args.cmd == "inject":
        inject_snippet(
            args.index_path,
            args.query,
            args.top_k,
            tags,
            sources,
            args.temperature,
            args.min_rating,
        )
    elif args.cmd == "compress":
        compress_snippet(
            args.index_path,
            args.query,
            args.top_k,
            tags,
            sources,
            args.model,
            args.min_rating,
            args.max_length,
            args.min_length,
        )
    elif args.cmd == "walk":
        walk_search(
            args.index_path,
            args.query,
            args.top_k,
            args.depth,
            args.limit,
            tags,
            sources,
            args.algo,
            args.restart,
            args.min_rating,
        )
    elif args.cmd == "shell":
        shell(
            args.index_path,
            args.top_k,
            tags,
            sources,
            args.temperature,
            args.min_rating,
        )
    elif args.cmd == "capsule":
        show_capsule(args.index_path, args.id)
    elif args.cmd == "list":
        list_tags = args.tags.split(",") if args.tags else None
        list_sources = args.sources.split(",") if args.sources else None
        list_capsules(
            args.index_path,
            args.limit,
            list_tags,
            list_sources,
            args.min_rating,
        )
    elif args.cmd == "export":
        export_tags = args.tags.split(",") if args.tags else None
        export_sources = args.sources.split(",") if args.sources else None
        export_capsules(
            args.index_path,
            args.output_file,
            export_tags,
            export_sources,
            args.min_rating,
        )
    elif args.cmd == "graph":
        graph_tags = args.tags.split(",") if args.tags else None
        export_graph(args.index_path, args.output_file, args.limit, graph_tags)
    elif args.cmd == "prune":
        id_list = [int(x) for x in args.ids.split(",")] if args.ids else None
        prune_tags = args.tags.split(",") if args.tags else None
        prune_capsules(args.index_path, id_list, prune_tags)
    elif args.cmd == "reindex":
        reindex_store(args.index_path, args.model, args.factory)
    elif args.cmd == "rate":
        rate_ids = [int(x) for x in args.ids.split(",")] if args.ids else None
        rate_tags = args.tags.split(",") if args.tags else None
        rate_capsules(args.index_path, args.rating, rate_ids, rate_tags)
    elif args.cmd == "meta":
        meta_ids = [int(x) for x in args.ids.split(",")] if args.ids else None
        meta_tags = args.tags.split(",") if args.tags else None
        add_tags = args.add_tags.split(",") if args.add_tags else None
        remove_tags = args.remove_tags.split(",") if args.remove_tags else None
        update_meta(
            args.index_path,
            args.rating,
            add_tags,
            remove_tags,
            meta_ids,
            meta_tags,
        )
    elif args.cmd == "embed":
        embed_text(args.text, args.model, args.index)
    elif args.cmd == "similarity":
        similarity_cmd(args.text_a, args.text_b, args.model)
    elif args.cmd == "info":
        show_info(args.index_path)
    elif args.cmd == "stats":
        show_stats(args.log_file)
    elif args.cmd == "cache":
        clear_cache_cmd(args.index_path, args.summarizer, args.embeddings)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
