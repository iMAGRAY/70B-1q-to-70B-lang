from fastapi import FastAPI, HTTPException
from typing import List
import argparse
import os
import uvicorn

from .core import CapsuleStore, merge_capsules

app = FastAPI(title="SIGLA Server")
store: CapsuleStore | None = None
index_path: str = ""

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
def search(query: str, top_k: int = 5):
    if store is None:
        raise HTTPException(status_code=500, detail="Store not loaded")
    return store.query(query, top_k=top_k)

@app.get("/ask")
def ask(query: str, top_k: int = 5):
    if store is None:
        raise HTTPException(status_code=500, detail="Store not loaded")
    results = store.query(query, top_k=top_k)
    merged = merge_capsules(results)
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
    return {"added": len(capsules)}

def cli():
    parser = argparse.ArgumentParser(description="Run SIGLA API server")
    parser.add_argument("index_path", help="Path prefix of the FAISS index")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    global index_path
    index_path = args.index_path
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    cli()
