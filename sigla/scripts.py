#!/usr/bin/env python3

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

from .core import (
    CapsuleStore, 
    get_available_local_models, 
    create_store_with_best_local_model,
    MissingDependencyError
)
from .dsl import INTENT, RETRIEVE, MERGE, INJECT, EXPAND


def cmd_ingest(args) -> None:
    """Ingest documents into the capsule store."""
    print(f"Ingesting from: {args.input}")
    
    # Create store
    if args.local_model:
        if not os.path.exists(args.local_model):
            print(f"Error: Local model path not found: {args.local_model}")
            sys.exit(1)
        store = CapsuleStore.with_local_model(args.local_model, device=args.device)
    elif args.auto_model:
        store = create_store_with_best_local_model(device=args.device)
    else:
        store = CapsuleStore(model_name=args.model, device=args.device)
    
    # Read input files
    capsules = []
    if os.path.isfile(args.input):
        # Single file
        if args.input.endswith('.json'):
            with open(args.input, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    capsules = data
                else:
                    capsules = [{"text": json.dumps(data)}]
        else:
            with open(args.input, 'r', encoding='utf-8') as f:
                content = f.read()
                capsules = [{"text": content, "source": args.input}]
    elif os.path.isdir(args.input):
        # Directory
        for file_path in Path(args.input).rglob("*"):
            if file_path.is_file() and file_path.suffix in ['.txt', '.md', '.json']:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        capsules.append({
                            "text": content,
                            "source": str(file_path),
                            "tags": [file_path.suffix[1:]]  # Add file extension as tag
                        })
                except Exception as e:
                    print(f"Warning: Failed to read {file_path}: {e}")
    else:
        print(f"Error: Input path not found: {args.input}")
        sys.exit(1)
    
    if not capsules:
        print("No content found to ingest")
        sys.exit(1)
    
    print(f"Adding {len(capsules)} capsules...")
    store.add_capsules(capsules)
    
    # Save the store
    store.save(args.output)
    print(f"Saved capsule store to {args.output}")


def cmd_search(args) -> None:
    """Search the capsule store."""
    if not os.path.exists(f"{args.store}.json"):
        print(f"Error: Store not found: {args.store}")
        sys.exit(1)
    
    # Load store
    store = CapsuleStore()
    store.load(args.store)
    
    # Perform search
    results = store.query(args.query, top_k=args.top_k, tags=args.tags)
    
    if not results:
        print("No results found")
        return
    
    print(f"Found {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"\n--- Result {i} (score: {result['score']:.3f}) ---")
        print(f"ID: {result['id']}")
        if result.get('source'):
            print(f"Source: {result['source']}")
        if result.get('tags'):
            print(f"Tags: {', '.join(result['tags'])}")
        print(f"Text: {result['text'][:200]}...")


def cmd_info(args) -> None:
    """Show information about the capsule store."""
    if not os.path.exists(f"{args.store}.json"):
        print(f"Error: Store not found: {args.store}")
        sys.exit(1)
    
    # Load store
    store = CapsuleStore()
    store.load(args.store)
    
    info = store.get_info()
    print("Capsule Store Information:")
    print(f"  Model: {info['model']}")
    print(f"  Dimension: {info['dimension']}")
    print(f"  Index Factory: {info['index_factory']}")
    print(f"  Capsules: {info['capsules']}")
    print(f"  Vectors: {info['vectors']}")
    print(f"  Device: {info.get('device', 'unknown')}")
    
    if info['tags']:
        print(f"  Tags:")
        for tag, count in sorted(info['tags'].items()):
            print(f"    {tag}: {count}")


def cmd_list_models(args) -> None:
    """List available local models."""
    models = get_available_local_models()
    
    if not models:
        print("No local models found in current directory")
        print("\nTo use local models:")
        print("1. Download model files to a directory")
        print("2. Ensure the directory contains config.json and model files")
        print("3. Use --local_model <path> or --auto_model")
        return
    
    print(f"Found {len(models)} local models:")
    for i, model in enumerate(models, 1):
        model_path = Path(model)
        size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
        size_mb = size / (1024 * 1024)
        print(f"  {i}. {model} ({size_mb:.1f} MB)")


def cmd_serve(args) -> None:
    """Start the web server."""
    try:
        import uvicorn
        from fastapi import FastAPI
    except ImportError:
        print("Error: fastapi and uvicorn are required for the server")
        print("Install with: pip install fastapi uvicorn")
        sys.exit(1)
    
    # Load or create store
    if args.store and os.path.exists(f"{args.store}.json"):
        print(f"Loading store from {args.store}")
        store = CapsuleStore()
        store.load(args.store)
    else:
        print("Creating new store")
        if args.auto_model:
            store = create_store_with_best_local_model(device=args.device)
        else:
            store = CapsuleStore(device=args.device)
    
    # Create simple FastAPI app
    app = FastAPI(title="SIGLA API", description="Semantic Information Graph API")
    
    @app.get("/")
    def root():
        return {"message": "SIGLA API is running", "store_info": store.get_info()}
    
    @app.post("/search")
    def search_endpoint(query: str, top_k: int = 5):
        return store.query(query, top_k=top_k)
    
    print(f"Starting server on http://localhost:{args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)


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


def main():
    parser = argparse.ArgumentParser(description="SIGLA utility")
    subparsers = parser.add_subparsers(dest="cmd", required=True)
    parser.add_argument("--log-file")

    # Ingest command
    ingest_p = subparsers.add_parser("ingest", help="ingest documents into a new store")
    ingest_p.add_argument("input", help="input file or directory")
    ingest_p.add_argument("output", help="output store path")
    ingest_p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", 
                         help="HuggingFace model name")
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
    
    if not args.command:
        parser.print_help()


if __name__ == "__main__":
    main()
