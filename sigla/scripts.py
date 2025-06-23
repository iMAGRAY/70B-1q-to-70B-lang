import argparse
import json
from pathlib import Path

from .core import CapsuleStore


def ingest(json_file: str, index_path: str, model: str):
    store = CapsuleStore(model_name=model)
    with open(json_file, 'r', encoding='utf-8') as f:
        capsules = json.load(f)
    store.add_capsules(capsules)
    store.save(index_path)


def search(index_path: str, query: str, top_k: int):
    store = CapsuleStore()
    store.load(index_path)
    results = store.query(query, top_k=top_k)
    print(json.dumps(results, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="SIGLA utility")
    subparsers = parser.add_subparsers(dest="cmd")

    ingest_p = subparsers.add_parser("ingest")
    ingest_p.add_argument("json_file")
    ingest_p.add_argument("index_path")
    ingest_p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")

    search_p = subparsers.add_parser("search")
    search_p.add_argument("index_path")
    search_p.add_argument("query")
    search_p.add_argument("--top_k", type=int, default=5)

    args = parser.parse_args()
    if args.cmd == "ingest":
        ingest(args.json_file, args.index_path, args.model)
    elif args.cmd == "search":
        search(args.index_path, args.query, args.top_k)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
