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

from .core import CapsuleStore, merge_capsules, compress_capsules, MissingDependencyError
from .graph import expand_with_links, random_walk_links

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
        s = CapsuleStore(lazy=True)
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
    def ask(query: str, top_k: int = 5, tags: str | None = None, temperature: float = 1.0):
        if store is None:
            raise HTTPException(status_code=500, detail="Store not loaded")
        tag_list = tags.split(',') if tags else None
        results = store.query(query, top_k=top_k, tags=tag_list)
        merged = merge_capsules(results, temperature=temperature)
        siglog.log({"type": "ask", "query": query, "top_k": top_k, "tags": tag_list, "temperature": temperature, "context": merged})
        return {"context": merged}

    @app.get("/capsule/{idx}")
    def get_capsule(idx: int):
        if store is None:
            raise HTTPException(status_code=500, detail="Store not loaded")
        if idx < 0 or idx >= len(store.meta):
            raise HTTPException(status_code=404, detail="Capsule not found")
        return store.meta[idx]

    @app.post("/update")
    def update_capsules(capsules: List[dict], link_neighbors: int = 0):
        if store is None:
            raise HTTPException(status_code=500, detail="Store not loaded")
        store.add_capsules(capsules, link_neighbors=link_neighbors)
        if index_path:
            store.save(index_path)
        siglog.log({"type": "update", "added": len(capsules), "link": link_neighbors})
        return {"added": len(capsules)}

    @app.get("/info")
    def info():
        """Return summary information about the index."""
        if store is None:
            raise HTTPException(status_code=500, detail="Store not loaded")
        tag_counts: dict[str, int] = {}
        for m in store.meta:
            for t in m.get("tags", []):
                tag_counts[t] = tag_counts.get(t, 0) + 1
        data = {
            "model": store.model_name,
            "dimension": store.dimension,
            "capsules": len(store.meta),
        }
        if tag_counts:
            data["tags"] = tag_counts
        return data

    @app.get("/list")
    def list_capsules(limit: int = 20, tags: str | None = None):
        if store is None:
            raise HTTPException(status_code=500, detail="Store not loaded")
        tag_list = tags.split(',') if tags else None
        results = []
        for meta in store.meta:
            if tag_list and not set(tag_list).intersection(meta.get("tags", [])):
                continue
            results.append(meta)
            if len(results) >= limit:
                break
        return results

    @app.get("/dump")
    def dump_capsules(limit: int = 0, tags: str | None = None):
        """Return capsules as JSON (optionally filtered and limited)."""
        if store is None:
            raise HTTPException(status_code=500, detail="Store not loaded")
        tag_list = tags.split(',') if tags else None
        results = []
        for meta in store.meta:
            if tag_list and not set(tag_list).intersection(meta.get("tags", [])):
                continue
            results.append(meta)
            if limit and len(results) >= limit:
                break
        return results

    @app.get("/graph")
    def graph(limit: int = 0, tags: str | None = None):
        """Return the capsule graph in Graphviz DOT format."""
        if store is None:
            raise HTTPException(status_code=500, detail="Store not loaded")
        tag_list = tags.split(',') if tags else None
        from .graph import to_dot
        dot = to_dot(store, limit=limit or None, tags=tag_list)
        return {
            "dot": dot
        }

    @app.get("/walk")
    def walk(
        query: str,
        top_k: int = 5,
        depth: int = 1,
        limit: int = 10,
        tags: str | None = None,
        algo: str = "bfs",
        restart: float = 0.5,
    ):
        """Expand results via capsule links."""
        if store is None:
            raise HTTPException(status_code=500, detail="Store not loaded")
        tag_list = tags.split(',') if tags else None
        results = store.query(query, top_k=top_k, tags=tag_list)
        if algo == "random":
            expanded = random_walk_links(
                results, store, steps=depth, restart=restart, limit=limit
            )
        else:
            expanded = expand_with_links(results, store, depth=depth, limit=limit)
        siglog.log(
            {
                "type": "walk",
                "query": query,
                "top_k": top_k,
                "depth": depth,
                "limit": limit,
                "algo": algo,
                "restart": restart,
                "tags": tag_list,
            }
        )
        return expanded

    @app.get("/compress")
    def compress(
        query: str,
        top_k: int = 5,
        tags: str | None = None,
        model: str = "sshleifer/distilbart-cnn-12-6",
    ):
        """Return a summary of retrieved capsules."""
        if store is None:
            raise HTTPException(status_code=500, detail="Store not loaded")
        tag_list = tags.split(',') if tags else None
        results = store.query(query, top_k=top_k, tags=tag_list)
        try:
            summary = compress_capsules(results, model_name=model)
        except MissingDependencyError as e:
            raise HTTPException(status_code=500, detail=str(e))
        siglog.log(
            {
                "type": "compress",
                "query": query,
                "top_k": top_k,
                "model": model,
                "tags": tag_list,
            }
        )
        return {"summary": summary}

    @app.post("/prune")
    def prune(ids: str = "", tags: str | None = None):
        """Remove capsules by id or tags and rebuild the index."""
        if store is None:
            raise HTTPException(status_code=500, detail="Store not loaded")
        id_list = [int(x) for x in ids.split(",") if x] if ids else []
        tag_list = tags.split(',') if tags else None
        remove_set = set(id_list)
        if tag_list:
            for i, meta in enumerate(store.meta):
                if set(tag_list).intersection(meta.get("tags", [])):
                    remove_set.add(i)
        if not remove_set:
            return {"removed": 0}
        removed = store.remove_capsules(sorted(remove_set))
        if index_path:
            store.save(index_path)
        siglog.log({"type": "prune", "removed": removed, "ids": sorted(remove_set), "tags": tag_list})
        return {"removed": removed}

    @app.post("/reindex")
    def reindex(model: str | None = None, factory: str | None = None):
        """Recompute all embeddings and rebuild the index."""
        if store is None:
            raise HTTPException(status_code=500, detail="Store not loaded")
        try:
            store.rebuild_index(model, factory)
        except MissingDependencyError as e:
            raise HTTPException(status_code=500, detail=str(e))
        if index_path:
            store.save(index_path)
        siglog.log({"type": "reindex", "model": model or store.model_name, "factory": factory or store.index_factory})
        return {"model": store.model_name, "factory": store.index_factory}

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
