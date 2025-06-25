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
        print("3. Use --local-model <path> or --auto-model")
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


def cmd_dsl(args) -> None:
    """Run DSL commands."""
    if not args.store or not os.path.exists(f"{args.store}.json"):
        print("Error: Store is required for DSL commands")
        sys.exit(1)
    
    store = CapsuleStore()
    store.load(args.store)
    
    if args.command == "intent":
        result = INTENT(args.text)
        print(f"Intent: {result}")
    
    elif args.command == "retrieve":
        results = RETRIEVE(args.text, store, top_k=args.top_k)
        print(f"Retrieved {len(results)} capsules:")
        for i, cap in enumerate(results, 1):
            print(f"  {i}. {cap['text'][:100]}...")
    
    elif args.command == "merge":
        results = RETRIEVE(args.text, store, top_k=args.top_k)
        merged = MERGE(results)
        print(f"Merged result:\n{merged}")
    
    elif args.command == "inject":
        result = INJECT(args.text, store)
        print(f"Injected capsule ID: {result}")
    
    elif args.command == "expand":
        results = RETRIEVE(args.text, store, top_k=3)
        if results:
            expanded = EXPAND(results[0], store)
            print(f"Expanded with {len(expanded)} related capsules")
            for cap in expanded:
                print(f"  - {cap['text'][:100]}...")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="SIGLA - Semantic Information Graph with Language Agents")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents")
    ingest_parser.add_argument("input", help="Input file or directory")
    ingest_parser.add_argument("-o", "--output", default="capsules", help="Output store name")
    ingest_parser.add_argument("-m", "--model", default="sentence-transformers/all-MiniLM-L6-v2", 
                               help="Model name")
    ingest_parser.add_argument("--local-model", help="Path to local model directory")
    ingest_parser.add_argument("--auto-model", action="store_true", 
                               help="Automatically use best available local model")
    ingest_parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"],
                               help="Device to use")
    ingest_parser.set_defaults(func=cmd_ingest)
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search capsules")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("-s", "--store", default="capsules", help="Store name")
    search_parser.add_argument("-k", "--top-k", type=int, default=5, help="Number of results")
    search_parser.add_argument("-t", "--tags", nargs="*", help="Filter by tags")
    search_parser.set_defaults(func=cmd_search)
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Show store information")
    info_parser.add_argument("-s", "--store", default="capsules", help="Store name")
    info_parser.set_defaults(func=cmd_info)
    
    # List models command
    list_parser = subparsers.add_parser("list-models", help="List available local models")
    list_parser.set_defaults(func=cmd_list_models)
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start web server")
    serve_parser.add_argument("-s", "--store", help="Store to load")
    serve_parser.add_argument("-p", "--port", type=int, default=8000, help="Port number")
    serve_parser.add_argument("--auto-model", action="store_true",
                              help="Use best available local model")
    serve_parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"],
                              help="Device to use")
    serve_parser.set_defaults(func=cmd_serve)
    
    # Convert command
    conv_parser = subparsers.add_parser("convert", help="Convert HF model to .capsulegraph")
    conv_parser.add_argument("model_path", help="Path or name of HF model")
    conv_parser.add_argument("output", help="Output .capsulegraph file")
    conv_parser.add_argument("--bits", type=int, default=8, choices=[8,16], help="Quantisation bits (8=int8,16=fp16)")
    conv_parser.add_argument("--device", default="cpu", choices=["cpu","cuda","mps"], help="Device for conversion")
    conv_parser.set_defaults(func=lambda a: _cmd_convert(a))
    
    # Run capsulegraph command
    run_parser = subparsers.add_parser("run-cg", help="Run a .capsulegraph archive and generate text")
    run_parser.add_argument("archive", help="Path to .capsulegraph")
    run_parser.add_argument("--prompt", default="Hello", help="Input prompt")
    run_parser.add_argument("--device", default="cpu", choices=["cpu","cuda","mps"], help="Device")
    run_parser.add_argument("--max-tokens", type=int, default=128, help="Max new tokens")
    run_parser.set_defaults(func=lambda a: _cmd_run_cg(a))
    
    # DSL command
    dsl_parser = subparsers.add_parser("dsl", help="Run DSL commands")
    dsl_parser.add_argument("command", choices=["intent", "retrieve", "merge", "inject", "expand"])
    dsl_parser.add_argument("text", help="Input text")
    dsl_parser.add_argument("-s", "--store", required=True, help="Store name")
    dsl_parser.add_argument("-k", "--top-k", type=int, default=5, help="Number of results")
    dsl_parser.set_defaults(func=cmd_dsl)

    args = parser.parse_args()
    
    if not args.command:
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


if __name__ == "__main__":
    main()
