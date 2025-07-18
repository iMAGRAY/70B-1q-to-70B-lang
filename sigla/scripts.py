#!/usr/bin/env python3

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import time
import subprocess
from itertools import islice

from .core import (
    CapsuleStore, 
    get_available_local_models, 
    create_store_with_best_local_model,
    MissingDependencyError,
    compress_capsules,
)
from .abtest import run_ab_test

# logging helper
from . import log as siglog


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
    
    capsules = _read_capsules(args.input)
    if not capsules:
        print(f"Error: Input path not found or no readable files: {args.input}")
        sys.exit(1)
    
    # --------------------------- streaming ingest in batches for efficiency
    total = len(capsules)
    batch_size = max(1, args.batch_size)

    try:
        from tqdm import tqdm  # type: ignore
        pbar = tqdm(total=total, desc="Ingesting", unit="caps")
    except ImportError:
        pbar = None

    it = iter(capsules)
    while True:
        batch = list(islice(it, batch_size))
        if not batch:
            break
        store.add_capsules(batch)
        if pbar:
            pbar.update(len(batch))

    if pbar:
        pbar.close()
    
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
        print("  Tags:")
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
    """Запустить полноценный сервер FastAPI."""
    try:
        import uvicorn
        from . import server as sigserver
    except ImportError:
        print("Error: fastapi and uvicorn are required for the server")
        print("Install with: pip install fastapi uvicorn")
        sys.exit(1)

    # Load or create store
    if args.store and os.path.exists(f"{args.store}.json"):
        print(f"Loading store from {args.store}")
        sigserver.store = CapsuleStore(model_name="dummy")
        sigserver.store.load(args.store)
    else:
        print("Creating new store")
        sigserver.store = (
            create_store_with_best_local_model(device=args.device)
            if args.auto_model
            else CapsuleStore(model_name="dummy", device=args.device)
        )

    sigserver.index_path = args.store or "sigla"
    app = sigserver.create_server()

    print(f"Starting server on http://localhost:{args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)


def cmd_demo(args) -> None:
    """Запустить демонстрационный сервер с примерными капсулами."""
    store_name = args.store
    sample = Path(__file__).resolve().parent.parent / "sample_capsules.json"
    if not Path(f"{store_name}.json").is_file():
        print(f"Создаётся демо-индекс {store_name}")
        with open(sample, "r", encoding="utf-8") as f:
            data = json.load(f)
        demo_store = CapsuleStore(model_name="dummy", auto_link_k=2)
        demo_store.add_capsules(data)
        demo_store.save(store_name)

    demo_args = argparse.Namespace(
        store=store_name, port=args.port, auto_model=False, device="auto"
    )
    cmd_serve(demo_args)


def cmd_dsl(args) -> None:
    """Execute DSL commands."""
    from .dsl import INTENT, RETRIEVE, MERGE, INJECT, EXPAND
    
    # Load store
    store = CapsuleStore()
    try:
        store.load(args.store)
    except FileNotFoundError:
        print(f"Error: Store '{args.store}' not found")
        return
    except MissingDependencyError as e:
        print(f"Error: {e}")
        return
    
    if args.command == "intent":
        result = INTENT(store, args.text)
        print(f"Intent: {result}")
    elif args.command == "retrieve":
        results = RETRIEVE(store, args.text, top_k=args.top_k)
        print(f"Retrieved {len(results)} capsules:")
        for i, result in enumerate(results):
            print(f"  {i+1}. {result.get('text', '')[:100]}...")
    elif args.command == "merge":
        # For merge, we need to retrieve first
        capsules = RETRIEVE(store, args.text, top_k=args.top_k)
        result = MERGE(capsules)
        print(f"Merged result:\n{result}")
    elif args.command == "inject":
        result = INJECT(args.text, store, tags=["dsl", "injected"])
        print(f"Injected: {result}")
    elif args.command == "expand":
        # Get first result and expand it
        capsules = RETRIEVE(store, args.text, top_k=1)
        if capsules:
            expanded = EXPAND(capsules[0], store, depth=1, limit=5)
            print(f"Expanded to {len(expanded)} capsules:")
            for i, cap in enumerate(expanded):
                print(f"  {i+1}. {cap.get('text', '')[:100]}...")
        else:
            print("No capsules found to expand")


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


def cmd_module(args) -> None:
    """Handle registry commands (add/remove/list)."""
    from .registry import ModuleRegistry

    reg = ModuleRegistry()

    if args.action == "add":
        reg.add_module(args.name, args.path, tags=args.tags)
        print(f"Added module '{args.name}' -> {args.path}")
    elif args.action == "remove":
        removed = reg.remove_module(args.name)
        if removed:
            print(f"Removed module '{args.name}'")
        else:
            print("Nothing removed")
    elif args.action == "list" or args.action is None:
        modules = reg.list_modules()
        if not modules:
            print("Registry is empty")
            return
        print(f"{len(modules)} module(s):")
        for m in modules:
            ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(m.added))
            print(f"- {m.name:20} {m.path:40} [{', '.join(m.tags)}] {ts}")
    else:
        print("Unknown module action")


def cmd_list(args) -> None:
    """List stored capsules."""
    if not os.path.exists(f"{args.store}.json"):
        print(f"Error: Store not found: {args.store}")
        sys.exit(1)

    store = CapsuleStore()
    store.load(args.store)

    tag_filter = set(args.tags or [])
    count = 0
    for meta in store.meta:
        if tag_filter and not tag_filter.intersection(meta.get("tags", [])):
            continue
        print(f"[{meta['id']}] {meta.get('tags', [])} -> {meta.get('text', '')[:120]}…")
        count += 1
        if args.limit and count >= args.limit:
            break
    if count == 0:
        print("No capsules match the criteria")


def cmd_capsule(args) -> None:
    """Display a single capsule by id."""
    if not os.path.exists(f"{args.store}.json"):
        print(f"Error: Store not found: {args.store}")
        sys.exit(1)
    store = CapsuleStore()
    store.load(args.store)
    if args.id < 0 or args.id >= len(store.meta):
        print("Capsule id out of range")
        sys.exit(1)
    meta = store.meta[args.id]
    print(json.dumps(meta, ensure_ascii=False, indent=2))


def cmd_compress(args) -> None:
    """Summarize top-k capsules using an LLM summarizer."""
    if not os.path.exists(f"{args.store}.json"):
        print(f"Error: Store not found: {args.store}")
        sys.exit(1)
    store = CapsuleStore()
    store.load(args.store)
    results = store.query(args.query, top_k=args.top_k, tags=args.tags)
    if not results:
        print("Nothing found to compress")
        return
    summary = compress_capsules(results, model_name=args.model)
    print(summary)


def cmd_prune(args) -> None:
    """Remove capsules by id list or tag filter."""
    ids = [int(x) for x in args.ids.split(',')] if args.ids else None
    prune_capsules(args.store, ids=ids, tags=args.tags)


def cmd_reindex(args) -> None:
    """Rebuild embeddings for the store."""
    reindex_store(args.store, model=args.model, factory=args.factory)


def cmd_abtest(args) -> None:
    """Run A/B evaluation on a dataset JSON file."""
    import json
    from pathlib import Path

    path = Path(args.dataset)
    if not path.exists():
        print(f"Dataset not found: {path}")
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return
    if not isinstance(data, list):
        print("Dataset must be a list of objects with 'question' and 'answer'")
        return

    store = CapsuleStore()
    try:
        store.load(args.store)
    except FileNotFoundError:
        print(f"Store '{args.store}' not found")
        return
    except MissingDependencyError as e:
        print(f"Error: {e}")
        return

    summary = run_ab_test(
        store,
        data,
        top_k=args.top_k,
        temperature=args.temperature,
        baseline=args.baseline,
    )

    print("A/B test summary (average)")
    for side in ("sigla", "baseline"):
        print(f"  {side}:")
        for k, v in sorted(summary[side].items()):
            print(f"    {k:8}: {v:.3f}")


def _run_repl(store: "CapsuleStore", top_k: int = 5, tags: list[str] | None = None) -> None:
    """Very small interactive REPL for quick manual testing."""
    try:
        from prompt_toolkit import prompt  # type: ignore
    except ImportError:
        prompt = input  # fallback

    print("Entering SIGLA shell. Type :q or :exit to quit.")
    while True:
        try:
            query = prompt("SIGLA> ")  # type: ignore
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if query.strip() in {":q", ":exit"}:
            break
        if not query.strip():
            continue
        results = store.query(query, top_k=top_k, tags=tags)
        if not results:
            print("No results")
            continue
        for i, res in enumerate(results, 1):
            print(f"[{i}] (score {res['score']:.3f}) {res['text'][:120]}…")


def cmd_shell(args) -> None:
    """Run interactive shell on a capsule store."""
    if os.path.exists(f"{args.store}.json"):
        store = CapsuleStore()
        store.load(args.store)
    else:
        if args.auto_model:
            store = create_store_with_best_local_model(device=args.device)
        else:
            store = CapsuleStore(device=args.device)
            # Empty store; warn user
            print("Warning: created empty store – queries will return nothing until you ingest data.")
    _run_repl(store, top_k=args.top_k, tags=args.tags)


def cmd_stats(args) -> None:
    """Generate simple statistics from a JSONL log produced by sigla.log."""
    if not os.path.isfile(args.log):
        print(f"Log file not found: {args.log}")
        return
    counts: dict[str, int] = {}
    queries: dict[str, int] = {}
    with open(args.log, "r", encoding="utf-8") as f:
        for line in f:
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            etype = ev.get("type", "unknown")
            counts[etype] = counts.get(etype, 0) + 1
            if etype in {"search", "ask"}:
                q = ev.get("query", "")
                if q:
                    queries[q] = queries.get(q, 0) + 1

    print("Event counts:")
    for t, c in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {t:10}: {c}")

    if queries:
        print("\nTop queries:")
        for q, n in sorted(queries.items(), key=lambda x: -x[1])[:10]:
            print(f"  {n:4} × {q}")


# -----------------------------------------------------------------------------
# Clean CLI entry (rewritten – fixes previous merge conflicts)
# -----------------------------------------------------------------------------

def main() -> None:  # noqa: D401
    """SIGLA command-line utility (ingest, search, serve, convert, run-cg…)."""

    parser = argparse.ArgumentParser(description="SIGLA – Semantic Information Graph CLI")
    subparsers = parser.add_subparsers(dest="command")

    # ------------------------------------------------------------------ ingest
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents")
    ingest_parser.add_argument("input", help="Input file or directory")
    ingest_parser.add_argument("--output", "-o", default="capsules", help="Output store name")
    ingest_parser.add_argument("--model", "-m", default="sentence-transformers/all-MiniLM-L6-v2")
    ingest_parser.add_argument("--local-model")
    ingest_parser.add_argument("--auto-model", action="store_true")
    ingest_parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ingest_parser.add_argument("--batch-size", "-b", type=int, default=256, help="Number of capsules per add batch (streaming ingest)")
    ingest_parser.set_defaults(func=cmd_ingest)

    # ------------------------------------------------------ remote-ingest
    rem_parser = subparsers.add_parser("remote-ingest", help="Send documents to running SIGLA server via /update")
    rem_parser.add_argument("input", help="Input file or directory")
    rem_parser.add_argument("--url", default="http://localhost:8000", help="Base URL of SIGLA server")
    rem_parser.add_argument("--batch-size", "-b", type=int, default=256)
    rem_parser.add_argument("--token", help="X-API-Key header value")
    rem_parser.set_defaults(func=cmd_remote_ingest)

    # ----------------------------------------------------------------- search
    search_parser = subparsers.add_parser("search", help="Search capsules")
    search_parser.add_argument("query")
    search_parser.add_argument("--store", "-s", default="capsules")
    search_parser.add_argument("--top-k", "-k", type=int, default=5)
    search_parser.add_argument("--tags", nargs="*")
    search_parser.set_defaults(func=cmd_search)

    # ------------------------------------------------------------------- info
    info_parser = subparsers.add_parser("info", help="Show store info")
    info_parser.add_argument("--store", "-s", default="capsules")
    info_parser.set_defaults(func=cmd_info)

    # ------------------------------------------------------------- list-models
    list_parser = subparsers.add_parser("list-models", help="List local HF models")
    list_parser.set_defaults(func=cmd_list_models)

    # ------------------------------------------------------------------ serve
    serve_parser = subparsers.add_parser("serve", help="Run FastAPI server")
    serve_parser.add_argument("--store")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--auto-model", action="store_true")
    serve_parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    serve_parser.set_defaults(func=cmd_serve)

    # -------------------------------------------------------------- demo
    demo_parser = subparsers.add_parser("demo", help="Run demo server with sample capsules")
    demo_parser.add_argument("--store", "-s", default="demo_store")
    demo_parser.add_argument("--port", "-p", type=int, default=8000)
    demo_parser.set_defaults(func=cmd_demo)

    # -------------------------------------------------------------- convert cg
    conv_parser = subparsers.add_parser("convert", help="Convert HF model → .capsulegraph")
    conv_parser.add_argument("model_path")
    conv_parser.add_argument("output")
    conv_parser.add_argument("--bits", type=int, default=8, choices=[8, 16])
    conv_parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    conv_parser.set_defaults(func=_cmd_convert)

    # ---------------------------------------------------------- hf → gguf
    hfgguf_parser = subparsers.add_parser("hf-to-gguf", help="Convert HuggingFace checkpoint to GGUF via llama.cpp script")
    hfgguf_parser.add_argument("model_dir", help="Path to local HF model directory or repository ID")
    hfgguf_parser.add_argument("--outfile", "-o", help="Destination .gguf file (default: <model>-f16.gguf)")
    hfgguf_parser.add_argument("--outtype", default="f16", choices=["f32", "f16", "bf16", "q8_0", "auto"], help="Output tensor precision / quantization")
    hfgguf_parser.add_argument("--vocab-only", action="store_true", help="Export only tokenizer & vocab (tiny file)")
    hfgguf_parser.add_argument("--script", default="llama.cpp/convert_hf_to_gguf.py", help="Path to llama.cpp conversion script")
    hfgguf_parser.set_defaults(func=cmd_hf_to_gguf)

    # ----------------------------------------------------------- run capsuleg
    run_parser = subparsers.add_parser("run-cg", help="Generate text from .capsulegraph")
    run_parser.add_argument("archive")
    run_parser.add_argument("--prompt", default="Hello")
    run_parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    run_parser.add_argument("--max-tokens", type=int, default=128)
    run_parser.set_defaults(func=_cmd_run_cg)

    # ------------------------------------------------------------------- DSL
    dsl_parser = subparsers.add_parser("dsl", help="Run DSL helpers")
    dsl_parser.add_argument("command", choices=["intent", "retrieve", "merge", "inject", "expand"])
    dsl_parser.add_argument("text")
    dsl_parser.add_argument("--store", "-s", required=True)
    dsl_parser.add_argument("--top-k", "-k", type=int, default=5)
    dsl_parser.set_defaults(func=cmd_dsl)

    # ---------------------------------------------------------- module registry
    module_parser = subparsers.add_parser("module", help="Manage module registry")
    module_sub = module_parser.add_subparsers(dest="action")
    
    module_add = module_sub.add_parser("add", help="Add module to registry")
    module_add.add_argument("name")
    module_add.add_argument("path")
    module_add.add_argument("--tags", nargs="*")
    
    module_remove = module_sub.add_parser("remove", help="Remove module from registry")
    module_remove.add_argument("name")
    
    module_sub.add_parser("list", help="List modules")

    module_parser.set_defaults(func=cmd_module)

    # ----------------------------------------------------------------- list
    list_caps_parser = subparsers.add_parser("list", help="List capsules")
    list_caps_parser.add_argument("--store", "-s", default="capsules")
    list_caps_parser.add_argument("--limit", "-n", type=int, default=20)
    list_caps_parser.add_argument("--tags", nargs="*")
    list_caps_parser.set_defaults(func=cmd_list)

    # --------------------------------------------------------------- capsule
    cap_parser = subparsers.add_parser("capsule", help="Show capsule by id")
    cap_parser.add_argument("id", type=int)
    cap_parser.add_argument("--store", "-s", default="capsules")
    cap_parser.set_defaults(func=cmd_capsule)

    # -------------------------------------------------------------- compress
    comp_parser = subparsers.add_parser("compress", help="Summarize retrieved capsules")
    comp_parser.add_argument("query")
    comp_parser.add_argument("--store", "-s", default="capsules")
    comp_parser.add_argument("--top-k", "-k", type=int, default=5)
    comp_parser.add_argument("--tags", nargs="*")
    comp_parser.add_argument("--model", default="sshleifer/distilbart-cnn-12-6")
    comp_parser.set_defaults(func=cmd_compress)

    # ---------------------------------------------------------------- prune
    prune_parser = subparsers.add_parser("prune", help="Remove capsules by id or tag")
    prune_parser.add_argument("--store", "-s", default="capsules")
    prune_parser.add_argument("--ids", help="Comma-separated list of ids to remove")
    prune_parser.add_argument("--tags", nargs="*", help="Tags filter")
    prune_parser.set_defaults(func=cmd_prune)

    # -------------------------------------------------------------- reindex
    reidx_parser = subparsers.add_parser("reindex", help="Rebuild embeddings/index")
    reidx_parser.add_argument("--store", "-s", default="capsules")
    reidx_parser.add_argument("--model", help="New embedding model")
    reidx_parser.add_argument("--factory", help="FAISS index factory string")
    reidx_parser.set_defaults(func=cmd_reindex)

    # ------------------------------------------------------------------ shell
    shell_parser = subparsers.add_parser("shell", help="Interactive shell for quick tests")
    shell_parser.add_argument("--store", "-s", default="capsules")
    shell_parser.add_argument("--top-k", "-k", type=int, default=5)
    shell_parser.add_argument("--tags", nargs="*")
    shell_parser.add_argument("--auto-model", action="store_true", help="Pick best local model if store missing")
    shell_parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    shell_parser.set_defaults(func=cmd_shell)

    # ------------------------------------------------------------------ stats
    stats_parser = subparsers.add_parser("stats", help="Show statistics for a SIGLA JSONL log")
    stats_parser.add_argument("log", help="Path to JSONL log file")
    stats_parser.set_defaults(func=cmd_stats)

    # ----------------------------------------------------------------- abtest
    ab_parser = subparsers.add_parser("abtest", help="Run A/B evaluation")
    ab_parser.add_argument("dataset", help="JSON file with questions & answers")
    ab_parser.add_argument("--store", "-s", default="capsules")
    ab_parser.add_argument("--top-k", "-k", type=int, default=5)
    ab_parser.add_argument("--temperature", "-t", type=float, default=1.0)
    ab_parser.add_argument("--baseline", choices=["echo", "none"], default="echo")
    ab_parser.set_defaults(func=cmd_abtest)

    args = parser.parse_args()

    if not getattr(args, "func", None):  # pragma: no cover – safety
        parser.print_help()
        return

    try:
        args.func(args)
    except MissingDependencyError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


# -----------------------------------------------------------------------
# helper for convert
# -----------------------------------------------------------------------


def _cmd_convert(args):  # noqa: D401
    from .converter import convert_to_capsulegraph

    convert_to_capsulegraph(
        model_path=args.model_path,
        output_path=args.output,
        quant_bits=args.bits,
        device=args.device,
    )


def _cmd_run_cg(args):  # noqa: D401
    """Run generation from capsulegraph via runner."""
    from .runner import load_capsulegraph
    import torch

    model, tok, _ = load_capsulegraph(args.archive, device=args.device)

    input_ids = tok(args.prompt, return_tensors="pt").to(args.device)
    with torch.no_grad():
        output = model.generate(
            **input_ids,
            max_new_tokens=args.max_tokens,
            do_sample=True,
            temperature=0.9,
        )
    text = tok.decode(output[0], skip_special_tokens=True)
    print(text)


# -----------------------------------------------------------------------
# hf → gguf helper
# -----------------------------------------------------------------------


def cmd_hf_to_gguf(args):
    """Wrapper around llama.cpp's *convert_hf_to_gguf.py* for convenience."""

    script_path = Path(args.script)
    if not script_path.is_file():
        print(f"Conversion script not found: {script_path}")
        sys.exit(1)

    cmd: list[str] = [sys.executable, str(script_path), args.model_dir, "--outtype", args.outtype]
    if args.vocab_only:
        cmd.append("--vocab-only")
    if args.outfile:
        cmd.extend(["--outfile", args.outfile])

    print("Running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"Conversion failed (code {exc.returncode})")
        sys.exit(exc.returncode)


# -----------------------------------------------------------------------
# Remote ingest helper
# -----------------------------------------------------------------------


def _read_capsules(path: str) -> List[Dict[str, Any]]:
    """Utility: read files/dir into capsule dicts (same logic as cmd_ingest)."""
    caps: list[dict] = []
    if os.path.isfile(path):
        if path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                caps = data if isinstance(data, list) else [{"text": json.dumps(data)}]
        else:
            with open(path, "r", encoding="utf-8") as f:
                caps = [{"text": f.read(), "source": path}]
    elif os.path.isdir(path):
        for fp in Path(path).rglob("*"):
            if fp.is_file() and fp.suffix in {".txt", ".md", ".json"}:
                try:
                    with open(fp, "r", encoding="utf-8") as f:
                        caps.append({"text": f.read(), "source": str(fp), "tags": [fp.suffix[1:]]})
                except Exception:
                    pass
    return caps


def cmd_remote_ingest(args) -> None:  # noqa: D401
    """Stream documents to remote SIGLA server using /update endpoint."""

    try:
        import requests  # type: ignore
    except ImportError:
        print("Error: requests package required (pip install requests)")
        sys.exit(1)

    capsules = _read_capsules(args.input)
    if not capsules:
        print("No capsules found to send")
        sys.exit(1)

    headers = {"Content-Type": "application/json"}
    if args.token:
        headers["X-API-Key"] = args.token

    total = len(capsules)
    batch_size = max(1, args.batch_size)

    # send total once
    init_resp = requests.post(f"{args.url}/update?total={total}", json=[], headers=headers)
    if init_resp.status_code >= 300:
        print(f"Server error: {init_resp.text}")
        sys.exit(1)

    try:
        from tqdm import tqdm  # type: ignore
        pbar = tqdm(total=total, desc="Uploading", unit="caps")
    except ImportError:
        pbar = None

    it = iter(capsules)
    sent = 0
    while True:
        batch = list(islice(it, batch_size))
        if not batch:
            break
        resp = requests.post(f"{args.url}/update", json=batch, headers=headers)
        if resp.status_code >= 300:
            print(f"Server error: {resp.text}")
            sys.exit(1)
        sent += len(batch)
        if pbar:
            pbar.update(len(batch))

    if pbar:
        pbar.close()
    print(f"Uploaded {sent} capsules to {args.url}")


if __name__ == "__main__":
    main()
