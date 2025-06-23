try:
    from fastapi import FastAPI, HTTPException
except Exception:  # pragma: no cover - optional dependency
    FastAPI = None
    class HTTPException(Exception):
        pass

from typing import List
import argparse
import os
from . import log as siglog

from .core import CapsuleStore, merge_capsules

if FastAPI:
    app = FastAPI(title="SIGLA Server")
else:
    app = None

store: CapsuleStore | None = None
index_path: str = ""

if app:
    @app.on_event("startup")
    def _load_store():
        global store
        if not index_path:
            raise RuntimeError("Index path not set")
        s = CapsuleStore()
        if os.path.exists(index_path + ".index"):
            s.load(index_path)
        store = s

    @app.get("/search")
    def search(query: str, top_k: int = 5, tags: str | None = None):
        if store is None:
            raise HTTPException(status_code=500, detail="Store not loaded")
        tag_list = tags.split(',') if tags else None
        results = store.query(query, top_k=top_k, tags=tag_list)
        siglog.log({"type": "search", "query": query, "top_k": top_k, "tags": tag_list, "results": results})
        return results

    @app.get("/ask")
    def ask(query: str, top_k: int = 5, tags: str | None = None):
        if store is None:
            raise HTTPException(status_code=500, detail="Store not loaded")
        tag_list = tags.split(',') if tags else None
        results = store.query(query, top_k=top_k, tags=tag_list)
        merged = merge_capsules(results)
        siglog.log({"type": "ask", "query": query, "top_k": top_k, "tags": tag_list, "context": merged})
        return {"context": merged}

    @app.get("/capsule/{idx}")
    def get_capsule(idx: int):
        if store is None:
            raise HTTPException(status_code=500, detail="Store not loaded")
        if idx < 0 or idx >= len(store.meta):
            raise HTTPException(status_code=404, detail="Capsule not found")
        return store.meta[idx]

    @app.post("/update")
    def update_capsules(capsules: List[dict]):
        if store is None:
            raise HTTPException(status_code=500, detail="Store not loaded")
        store.add_capsules(capsules)
        if index_path:
            store.save(index_path)
        siglog.log({"type": "update", "added": len(capsules)})
        return {"added": len(capsules)}

def cli():
    parser = argparse.ArgumentParser(description="Run SIGLA API server")
    parser.add_argument("index_path", help="Path prefix of the FAISS index")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-file")
    args = parser.parse_args()
    global index_path
    index_path = args.index_path
    if args.log_file:
        siglog.start(args.log_file)

    if FastAPI is None:
        parser.error("fastapi is required to run the server")
    try:
        import uvicorn  # type: ignore
    except Exception:
        parser.error("uvicorn is required to run the server")

    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    cli()
