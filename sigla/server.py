from __future__ import annotations

import argparse
import os
import time
from typing import List, Optional

# ---------------------------------------------------------------------------
# Optional FastAPI import – graceful fallback when dependency missing.
# ---------------------------------------------------------------------------

try:
    from fastapi import FastAPI, HTTPException, Depends, Header, Response
    from contextlib import asynccontextmanager
    FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover – FastAPI not installed
    FASTAPI_AVAILABLE = False

# Prometheus metrics (optional)
try:
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST  # type: ignore
    PROM_AVAILABLE = True
except ImportError:  # pragma: no cover – prometheus_client not installed
    PROM_AVAILABLE = False
    # define dummies
    def Counter(*_a, **_kw):  # type: ignore
        class _Dummy:
            def labels(self, *args, **kwargs):
                return self
            def inc(self, *_):
                pass
            def observe(self, *_):
                pass
        return _Dummy()
    def Histogram(*_a, **_kw):  # type: ignore
        return Counter()

# ---------------------------------------------------------------------------

from . import log as siglog
from .core import CapsuleStore, merge_capsules, compress_capsules, MissingDependencyError
from .graph import expand_with_links, random_walk_links

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

REQUEST_COUNT = Counter(
    "sigla_requests_total",
    "Total requests to SIGLA API",
    ["endpoint"],
) if PROM_AVAILABLE else Counter("noop", "", ["endpoint"])

REQUEST_LATENCY = Histogram(
    "sigla_request_latency_seconds",
    "Request latency",
    ["endpoint"],
) if PROM_AVAILABLE else Histogram("noop", "", ["endpoint"])

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

store: CapsuleStore | None = None
index_path: str = ""
API_KEY_ENV = "SIGLA_API_KEY"


# ---------------------------------------------------------------------------
# Server creation
# ---------------------------------------------------------------------------

def create_server():
    """Create and configure the FastAPI server."""
    if not FASTAPI_AVAILABLE:
        raise MissingDependencyError("fastapi", "server functionality")

    # Auth dependency
    def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> bool:
        """Check API key via X-API-Key header."""
        expected = os.getenv(API_KEY_ENV)
        if not expected:  # security off
            return True
        if x_api_key == expected:
            return True
        raise HTTPException(status_code=403, detail="Invalid API key")

    # Lifespan manager
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """FastAPI lifespan context manager."""
        global store
        if not index_path:
            raise RuntimeError("Index path not set")
        local_store = CapsuleStore()
        if os.path.exists(index_path + ".index"):
            local_store.load(index_path)
            store = local_store
        yield

    # Create app
    app = FastAPI(title="SIGLA Server", lifespan=lifespan)

    # Metrics helpers
    def _track(name):
        def decorator(func):
            def wrapped(*args, **kwargs):
                start = time.perf_counter()
                try:
                    return func(*args, **kwargs)
                finally:
                    duration = time.perf_counter() - start
                    REQUEST_COUNT.labels(endpoint=name).inc()
                    REQUEST_LATENCY.labels(endpoint=name).observe(duration)
            return wrapped
        return decorator

    # Routes
    @app.get("/metrics")
    def metrics():
        if not PROM_AVAILABLE:
            raise HTTPException(status_code=500, detail="prometheus_client not installed")
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/search")
    @_track("search")
    def search(query: str, top_k: int = 5, tags: str | None = None):
        if store is None:
            raise HTTPException(status_code=500, detail="Store not loaded")
        tag_list = tags.split(",") if tags else None
        results = store.query(query, top_k=top_k, tags=tag_list)
        siglog.log({
            "type": "search",
            "query": query,
            "top_k": top_k,
            "tags": tag_list,
            "results": results,
        })
        return results

    @app.get("/ask")
    def ask(query: str, top_k: int = 5, tags: str | None = None, temperature: float = 1.0):
        if store is None:
            raise HTTPException(status_code=500, detail="Store not loaded")
        tag_list = tags.split(",") if tags else None
        results = store.query(query, top_k=top_k, tags=tag_list)
        merged = merge_capsules(results, temperature=temperature)
        siglog.log({
            "type": "ask",
            "query": query,
            "top_k": top_k,
            "tags": tag_list,
            "temperature": temperature,
            "context": merged,
        })
        return {"context": merged}

    @app.get("/capsule/{idx}")
    def get_capsule(idx: int, _auth: bool = Depends(require_api_key)):
        if store is None:
            raise HTTPException(status_code=500, detail="Store not loaded")
        if idx < 0 or idx >= len(store.meta):
            raise HTTPException(status_code=404, detail="Capsule not found")
        return store.meta[idx]

    @app.post("/update")
    def update_capsules(capsules: List[dict], _auth: bool = Depends(require_api_key)):
        if store is None:
            raise HTTPException(status_code=500, detail="Store not loaded")
        store.add_capsules(capsules)
        if index_path:
            store.save(index_path)
        siglog.log({"type": "update", "added": len(capsules)})
        return {"added": len(capsules)}

    @app.get("/info")
    def info():
        """Return summary information about the index."""
        if store is None:
            raise HTTPException(status_code=500, detail="Store not loaded")
        return store.get_info()

    @app.get("/list")
    def list_capsules(limit: int = 20, tags: str | None = None, _auth: bool = Depends(require_api_key)):
        if store is None:
            raise HTTPException(status_code=500, detail="Store not loaded")
        tag_list = tags.split(",") if tags else None
        results: List[dict] = []
        for meta in store.meta:
            if tag_list and not set(tag_list).intersection(meta.get("tags", [])):
                continue
            results.append(meta)
            if len(results) >= limit:
                break
        return results

    @app.get("/walk")
    def walk(
        query: str,
        top_k: int = 5,
        depth: int = 1,
        limit: int = 10,
        tags: str | None = None,
        algo: str = "bfs",
        restart: float = 0.5,
        _auth: bool = Depends(require_api_key),
    ):
        if store is None:
            raise HTTPException(status_code=500, detail="Store not loaded")
        tag_list = tags.split(",") if tags else None
        results = store.query(query, top_k=top_k, tags=tag_list)
        if algo == "random":
            expanded = random_walk_links(
                results, store, steps=depth, restart=restart, limit=limit
            )
        else:
            expanded = expand_with_links(results, store, depth=depth, limit=limit)
        siglog.log({
            "type": "walk",
            "query": query,
            "top_k": top_k,
            "depth": depth,
            "limit": limit,
            "algo": algo,
            "restart": restart,
            "tags": tag_list,
        })
        return expanded

    @app.get("/compress")
    def compress(
        query: str,
        top_k: int = 5,
        tags: str | None = None,
        model: str = "sshleifer/distilbart-cnn-12-6",
        _auth: bool = Depends(require_api_key),
    ):
        if store is None:
            raise HTTPException(status_code=500, detail="Store not loaded")
        tag_list = tags.split(",") if tags else None
        results = store.query(query, top_k=top_k, tags=tag_list)
        compressed = compress_capsules(results, model_name=model)
        siglog.log({
            "type": "compress",
            "query": query,
            "top_k": top_k,
            "tags": tag_list,
            "model": model,
            "result": compressed,
        })
        return {"compressed": compressed}

    @app.post("/prune")
    def prune(ids: str = "", tags: str | None = None, _auth: bool = Depends(require_api_key)):
        if store is None:
            raise HTTPException(status_code=500, detail="Store not loaded")
        
        to_remove: List[int] = []
        if ids:
            try:
                to_remove = [int(x.strip()) for x in ids.split(",") if x.strip()]
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid ID format")
        
        if tags:
            tag_list = [t.strip() for t in tags.split(",")]
            for idx, meta in enumerate(store.meta):
                if set(tag_list).intersection(meta.get("tags", [])):
                    to_remove.append(idx)
        
        removed = store.remove_capsules(to_remove)
        if index_path:
            store.save(index_path)
        
        siglog.log({"type": "prune", "removed": removed})
        return {"removed": removed}

    @app.post("/reindex")
    def reindex(model: str | None = None, factory: str | None = None, _auth: bool = Depends(require_api_key)):
        if store is None:
            raise HTTPException(status_code=500, detail="Store not loaded")
        
        store.rebuild_index(model_name=model, index_factory=factory)
        if index_path:
            store.save(index_path)
        
        siglog.log({"type": "reindex", "model": model, "factory": factory})
        return {"message": "Index rebuilt successfully"}

    return app


# Create app instance if FastAPI is available
if FASTAPI_AVAILABLE:
    app = create_server()
else:
    app = None  # type: ignore


def cli() -> None:
    """Command-line interface for the SIGLA server."""
    global index_path
    
    if not FASTAPI_AVAILABLE:
        raise MissingDependencyError("fastapi uvicorn", "server CLI")
    
    parser = argparse.ArgumentParser(description="SIGLA server")
    parser.add_argument("--index", "-i", default="sigla.index", help="Path to index file")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", "-p", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()
    
    index_path = args.index
    
    # Import uvicorn only when needed
    try:
        import uvicorn
    except ImportError:
        raise MissingDependencyError("uvicorn", "running the server")
    
    uvicorn.run(
        "sigla.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    cli()
