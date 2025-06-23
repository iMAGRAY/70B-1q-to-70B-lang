import argparse
import json
from pathlib import Path

from .core import CapsuleStore, merge_capsules, MissingDependencyError
from .dsl import INJECT
from .graph import expand_with_links
from . import log as siglog


3szrfh-codex/разработать-sigla-для-моделирования-мышления
def ingest(json_file: str, index_path: str, model: str, factory: str):
=======
xvy4pj-codex/разработать-sigla-для-моделирования-мышления
def ingest(json_file: str, index_path: str, model: str):
=======
def ingest(json_file: str, index_path: str, model: str, factory: str):
main
main
    try:
        if Path(index_path + ".index").exists():
            store = CapsuleStore()
            store.load(index_path)
        else:
3szrfh-codex/разработать-sigla-для-моделирования-мышления
            store = CapsuleStore(model_name=model, index_factory=factory)
=======
xvy4pj-codex/разработать-sigla-для-моделирования-мышления
            store = CapsuleStore(model_name=model)
=======
            store = CapsuleStore(model_name=model, index_factory=factory)
main
main
    except MissingDependencyError as e:
        print(f"error: {e}")
        return
    with open(json_file, 'r', encoding='utf-8') as f:
        capsules = json.load(f)
    store.add_capsules(capsules)
    store.save(index_path)
    siglog.log({"type": "ingest", "count": len(capsules)})


3szrfh-codex/разработать-sigla-для-моделирования-мышления
def update_capsules(json_file: str, index_path: str) -> None:
    """Append capsules to an existing index."""
    try:
        store = CapsuleStore()
        store.load(index_path)
    except MissingDependencyError as e:
        print(f"error: {e}")
        return
    with open(json_file, "r", encoding="utf-8") as f:
        capsules = json.load(f)
    store.add_capsules(capsules)
    store.save(index_path)
    siglog.log({"type": "update", "count": len(capsules)})


=======
main
def search(index_path: str, query: str, top_k: int, tags: list[str] | None = None):
    try:
        store = CapsuleStore()
        store.load(index_path)
    except MissingDependencyError as e:
        print(f"error: {e}")
        return
    results = store.query(query, top_k=top_k, tags=tags)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    siglog.log({"type": "search", "query": query, "top_k": top_k, "tags": tags, "results": results})


3szrfh-codex/разработать-sigla-для-моделирования-мышления
def inject_snippet(
    index_path: str,
    query: str,
    top_k: int,
    tags: list[str] | None = None,
    temperature: float = 1.0,
) -> None:
=======
def inject_snippet(index_path: str, query: str, top_k: int, tags: list[str] | None = None):
main
    try:
        store = CapsuleStore()
        store.load(index_path)
    except MissingDependencyError as e:
        print(f"error: {e}")
        return
    results = store.query(query, top_k=top_k, tags=tags)
    3szrfh-codex/разработать-sigla-для-моделирования-мышления
    snippet = INJECT(merge_capsules(results, temperature=temperature))
    print(snippet)
    siglog.log({
        "type": "inject",
        "query": query,
        "top_k": top_k,
        "tags": tags,
        "temperature": temperature,
    })
=======
    snippet = INJECT(merge_capsules(results))
    print(snippet)
    siglog.log({"type": "inject", "query": query, "top_k": top_k, "tags": tags})
main


def compress_snippet(index_path: str, query: str, top_k: int, tags: list[str] | None = None, model: str = "sshleifer/distilbart-cnn-12-6"):
    """Retrieve capsules and summarize them."""
    try:
        store = CapsuleStore()
        store.load(index_path)
    except MissingDependencyError as e:
        print(f"error: {e}")
        return
    results = store.query(query, top_k=top_k, tags=tags)
    try:
        from .core import compress_capsules
        summary = compress_capsules(results, model_name=model)
        print(summary)
    except MissingDependencyError as e:
        print(f"error: {e}")
    siglog.log({"type": "compress", "query": query, "top_k": top_k, "tags": tags})


 3szrfh-codex/разработать-sigla-для-моделирования-мышления
=======
xvy4pj-codex/разработать-sigla-для-моделирования-мышления
def walk_search(index_path: str, query: str, top_k: int, depth: int, limit: int, tags: list[str] | None = None):
=======
 main
def walk_search(
    index_path: str,
    query: str,
    top_k: int,
    depth: int,
    limit: int,
    tags: list[str] | None = None,
    algo: str = "bfs",
    restart: float = 0.5,
):
3szrfh-codex/разработать-sigla-для-моделирования-мышления
=======
main
main
    try:
        store = CapsuleStore()
        store.load(index_path)
    except MissingDependencyError as e:
        print(f"error: {e}")
        return
    results = store.query(query, top_k=top_k, tags=tags)
3szrfh-codex/разработать-sigla-для-моделирования-мышления
=======
xvy4pj-codex/разработать-sigla-для-моделирования-мышления
    expanded = expand_with_links(results, store, depth=depth, limit=limit)
    print(json.dumps(expanded, ensure_ascii=False, indent=2))
    siglog.log({"type": "walk", "query": query, "top_k": top_k, "depth": depth, "limit": limit, "tags": tags})
=======
main
    if algo == "random":
        from .graph import random_walk_links
        expanded = random_walk_links(results, store, steps=depth, restart=restart, limit=limit)
    else:
        expanded = expand_with_links(results, store, depth=depth, limit=limit)
    print(json.dumps(expanded, ensure_ascii=False, indent=2))
    siglog.log({"type": "walk", "query": query, "top_k": top_k, "depth": depth, "limit": limit, "algo": algo, "restart": restart, "tags": tags})
3szrfh-codex/разработать-sigla-для-моделирования-мышления
=======
main
main


def show_capsule(index_path: str, idx: int) -> None:
    """Print a capsule by its id."""
    try:
        store = CapsuleStore()
        store.load(index_path)
    except MissingDependencyError as e:
        print(f"error: {e}")
        return
    if idx < 0 or idx >= len(store.meta):
        print("error: invalid id")
        return
    print(json.dumps(store.meta[idx], ensure_ascii=False, indent=2))


def list_capsules(index_path: str, limit: int = 20, tags: list[str] | None = None) -> None:
    """List capsules optionally filtered by tags."""
    try:
        store = CapsuleStore()
        store.load(index_path)
    except MissingDependencyError as e:
        print(f"error: {e}")
        return
    results = []
    for meta in store.meta:
        if tags and not set(tags).intersection(meta.get("tags", [])):
            continue
        results.append(meta)
        if len(results) >= limit:
            break
    print(json.dumps(results, ensure_ascii=False, indent=2))

3szrfh-codex/разработать-sigla-для-моделирования-мышления
def shell(
    index_path: str,
    top_k: int,
    tags: list[str] | None = None,
    temperature: float = 1.0,
) -> None:
=======
def shell(index_path: str, top_k: int, tags: list[str] | None = None) -> None:
main
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
        results = store.query(query, top_k=top_k, tags=tags)
3szrfh-codex/разработать-sigla-для-моделирования-мышления
        snippet = INJECT(merge_capsules(results, temperature=temperature))
        print(snippet)
        siglog.log({
            "type": "shell",
            "query": query,
            "top_k": top_k,
            "temperature": temperature,
        })
=======
        snippet = INJECT(merge_capsules(results))
        print(snippet)
        siglog.log({"type": "shell", "query": query, "top_k": top_k})
main


def show_stats(log_file: str) -> None:
    """Print simple statistics from a log file."""
    counts = {}
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                event = json.loads(line)
                etype = event.get("type", "unknown")
                counts[etype] = counts.get(etype, 0) + 1
    except FileNotFoundError:
        print("error: log file not found")
        return
    print(json.dumps(counts, ensure_ascii=False, indent=2))


def show_info(index_path: str) -> None:
    """Print summary information about an index."""
    try:
        store = CapsuleStore()
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


def prune_capsules(index_path: str, ids: list[int] | None = None, tags: list[str] | None = None) -> None:
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
    removed = store.remove_capsules(sorted(remove_set))
    store.save(index_path)
    siglog.log({"type": "prune", "removed": removed, "ids": sorted(remove_set), "tags": tags})
    print(f"removed {removed} capsules")


 3szrfh-codex/разработать-sigla-для-моделирования-мышления
=======
xvy4pj-codex/разработать-sigla-для-моделирования-мышления
=======
main
def reindex_store(index_path: str, model: str | None = None, factory: str | None = None) -> None:
    """Rebuild embeddings for all capsules, optionally with a new model or index type."""
    try:
        store = CapsuleStore()
        store.load(index_path)
    except MissingDependencyError as e:
        print(f"error: {e}")
        return
    store.rebuild_index(model, factory)
    store.save(index_path)
    siglog.log({"type": "reindex", "model": model or store.model_name})
    print("index rebuilt")


3szrfh-codex/разработать-sigla-для-моделирования-мышления
=======
main
main
def main():
    parser = argparse.ArgumentParser(description="SIGLA utility")
    subparsers = parser.add_subparsers(dest="cmd")
    parser.add_argument("--log-file")

    ingest_p = subparsers.add_parser("ingest")
    ingest_p.add_argument("json_file")
    ingest_p.add_argument("index_path")
    ingest_p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
3szrfh-codex/разработать-sigla-для-моделирования-мышления
    ingest_p.add_argument("--factory", default="Flat", help="FAISS index factory")

    update_p = subparsers.add_parser("update", help="append capsules to an existing index")
    update_p.add_argument("json_file")
    update_p.add_argument("index_path")
=======
xvy4pj-codex/разработать-sigla-для-моделирования-мышления
=======
    ingest_p.add_argument("--factory", default="Flat", help="FAISS index factory")
main
main

    search_p = subparsers.add_parser("search")
    search_p.add_argument("index_path")
    search_p.add_argument("query")
    search_p.add_argument("--top_k", type=int, default=5)
    search_p.add_argument("--tags")

    inject_p = subparsers.add_parser("inject")
    inject_p.add_argument("index_path")
    inject_p.add_argument("query")
    inject_p.add_argument("--top_k", type=int, default=5)
    inject_p.add_argument("--tags")
3szrfh-codex/разработать-sigla-для-моделирования-мышления
    inject_p.add_argument("--temperature", type=float, default=1.0)
=======
 main

    compress_p = subparsers.add_parser("compress", help="summarize retrieved capsules")
    compress_p.add_argument("index_path")
    compress_p.add_argument("query")
    compress_p.add_argument("--top_k", type=int, default=5)
    compress_p.add_argument("--model", default="sshleifer/distilbart-cnn-12-6")
    compress_p.add_argument("--tags")

    walk_p = subparsers.add_parser("walk")
    walk_p.add_argument("index_path")
    walk_p.add_argument("query")
    walk_p.add_argument("--top_k", type=int, default=5)
    walk_p.add_argument("--depth", type=int, default=1)
    walk_p.add_argument("--limit", type=int, default=10)
3szrfh-codex/разработать-sigla-для-моделирования-мышления
    walk_p.add_argument("--algo", choices=["bfs", "random"], default="bfs")
    walk_p.add_argument("--restart", type=float, default=0.5, help="restart prob for random walk")
=======
xvy4pj-codex/разработать-sigla-для-моделирования-мышления
=======
    walk_p.add_argument("--algo", choices=["bfs", "random"], default="bfs")
    walk_p.add_argument("--restart", type=float, default=0.5, help="restart prob for random walk")
main
main
    walk_p.add_argument("--tags")

    cap_p = subparsers.add_parser("capsule")
    cap_p.add_argument("index_path")
    cap_p.add_argument("id", type=int)

    list_p = subparsers.add_parser("list", help="list capsules")
    list_p.add_argument("index_path")
    list_p.add_argument("--limit", type=int, default=20)
    list_p.add_argument("--tags")

    prune_p = subparsers.add_parser("prune", help="remove capsules")
    prune_p.add_argument("index_path")
    prune_p.add_argument("--ids")
    prune_p.add_argument("--tags")

3szrfh-codex/разработать-sigla-для-моделирования-мышления
=======
xvy4pj-codex/разработать-sigla-для-моделирования-мышления
=======
main
    reindex_p = subparsers.add_parser("reindex", help="rebuild embeddings")
    reindex_p.add_argument("index_path")
    reindex_p.add_argument("--model")
    reindex_p.add_argument("--factory")

3szrfh-codex/разработать-sigla-для-моделирования-мышления
=======
main
main
    info_p = subparsers.add_parser("info", help="show index summary")
    info_p.add_argument("index_path")

    shell_p = subparsers.add_parser("shell")
    shell_p.add_argument("index_path")
    shell_p.add_argument("--top_k", type=int, default=5)
    shell_p.add_argument("--tags")
3szrfh-codex/разработать-sigla-для-моделирования-мышления
    shell_p.add_argument("--temperature", type=float, default=1.0)
=======
main

    stats_p = subparsers.add_parser("stats", help="summarize a log file")
    stats_p.add_argument("log_file")

    args = parser.parse_args()
    if args.log_file and args.cmd != "stats":
        siglog.start(args.log_file)
    tags = args.tags.split(',') if hasattr(args, 'tags') and args.tags else None
    if args.cmd == "ingest":
3szrfh-codex/разработать-sigla-для-моделирования-мышления
        ingest(args.json_file, args.index_path, args.model, args.factory)
    elif args.cmd == "update":
        update_capsules(args.json_file, args.index_path)
    elif args.cmd == "search":
        search(args.index_path, args.query, args.top_k, tags)
    elif args.cmd == "inject":
        inject_snippet(args.index_path, args.query, args.top_k, tags, args.temperature)
    elif args.cmd == "compress":
        compress_snippet(args.index_path, args.query, args.top_k, tags, args.model)
    elif args.cmd == "walk":
=======
xvy4pj-codex/разработать-sigla-для-моделирования-мышления
        ingest(args.json_file, args.index_path, args.model)
=======
        ingest(args.json_file, args.index_path, args.model, args.factory)
main
    elif args.cmd == "search":
        search(args.index_path, args.query, args.top_k, tags)
    elif args.cmd == "inject":
        inject_snippet(args.index_path, args.query, args.top_k, tags)
    elif args.cmd == "compress":
        compress_snippet(args.index_path, args.query, args.top_k, tags, args.model)
    elif args.cmd == "walk":
xvy4pj-codex/разработать-sigla-для-моделирования-мышления
        walk_search(args.index_path, args.query, args.top_k, args.depth, args.limit, tags)
=======
 main
        walk_search(
            args.index_path,
            args.query,
            args.top_k,
            args.depth,
            args.limit,
            tags,
            args.algo,
            args.restart,
        )
3szrfh-codex/разработать-sigla-для-моделирования-мышления
    elif args.cmd == "shell":
        shell(args.index_path, args.top_k, tags, args.temperature)
=======
main
    elif args.cmd == "shell":
        shell(args.index_path, args.top_k, tags)
main
    elif args.cmd == "capsule":
        show_capsule(args.index_path, args.id)
    elif args.cmd == "list":
        list_tags = args.tags.split(',') if args.tags else None
        list_capsules(args.index_path, args.limit, list_tags)
    elif args.cmd == "prune":
        id_list = [int(x) for x in args.ids.split(',')] if args.ids else None
        prune_tags = args.tags.split(',') if args.tags else None
        prune_capsules(args.index_path, id_list, prune_tags)
3szrfh-codex/разработать-sigla-для-моделирования-мышления
    elif args.cmd == "reindex":
        reindex_store(args.index_path, args.model, args.factory)
=======
xvy4pj-codex/разработать-sigla-для-моделирования-мышления
=======
    elif args.cmd == "reindex":
        reindex_store(args.index_path, args.model, args.factory)
main
 main
    elif args.cmd == "info":
        show_info(args.index_path)
    elif args.cmd == "stats":
        show_stats(args.log_file)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
