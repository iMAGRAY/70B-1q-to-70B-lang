from __future__ import annotations

import argparse
import os
import time
from typing import List, Optional, Any
import asyncio

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
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST, Gauge  # type: ignore
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
            def set(self, *_):
                pass
        return _Dummy()
    def Histogram(*_a, **_kw):  # type: ignore
        return Counter()
    def Gauge(*_a, **_kw):  # type: ignore
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

CAPSULE_TOTAL = Gauge(
    "sigla_capsules_total",
    "Total number of capsules in store",
) if PROM_AVAILABLE else Gauge("noop", "")

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

store: CapsuleStore | None = None
index_path: str = ""
API_KEY_ENV = "SIGLA_API_KEY"
# auto-reindex settings
REINDEX_THRESHOLD = 500  # capsules added
_added_since_reindex = 0
_ingest_total = 0
_ingest_done = 0
_event_subscribers: list[Any] = []


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
        if store is not None:
            CAPSULE_TOTAL.set(len(store.meta))
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
    def ask(
        query: str,
        top_k: int = 5,
        tags: str | None = None,
        temperature: float = 1.0,
        min_score: float = 0.35,
        fallback: str = "echo",  # echo|none
        kv: bool = False,  # when true, include token ids for KV-cache style injection
    ):
        if store is None:
            raise HTTPException(status_code=500, detail="Store not loaded")
        tag_list = tags.split(",") if tags else None
        results = store.query(query, top_k=top_k, tags=tag_list)
        
        # Confidence check
        low_conf = not results or (results and results[0].get("score", 0.0) < min_score)

        if low_conf and fallback == "echo":
            merged = query  # simplest fallback – pass the user query itself
        else:
            merged = merge_capsules(results, temperature=temperature)

        # Prepare token ids if requested
        token_ids: list[int] | None = None
        if kv:
            try:
                from transformers import AutoTokenizer  # type: ignore
                tok = AutoTokenizer.from_pretrained(store.model_name.replace("local:", ""))
                token_ids = tok(merged, add_special_tokens=False)["input_ids"]  # type: ignore[index]
            except Exception:
                # Fallback – return UTF-8 bytes as ints if tokenizer unavailable
                token_ids = list(merged.encode("utf-8"))

        # KV flag currently informational – downstream client may use it
        context_payload = {
            "context": merged,
            "low_conf": low_conf,
            "kv": kv,
            "tokens": token_ids,
        }
        siglog.log({
            "type": "ask",
            "query": query,
            "top_k": top_k,
            "tags": tag_list,
            "temperature": temperature,
            "min_score": min_score,
            "low_conf": low_conf,
            "context": merged,
        })
        return context_payload

    @app.get("/capsule/{idx}")
    def get_capsule(idx: int, _auth: bool = Depends(require_api_key)):
        if store is None:
            raise HTTPException(status_code=500, detail="Store not loaded")
        if idx < 0 or idx >= len(store.meta):
            raise HTTPException(status_code=404, detail="Capsule not found")
        return store.meta[idx]

    @app.post("/update")
    def update_capsules(
        capsules: List[dict],
        total: int | None = None,
        _auth: bool = Depends(require_api_key),
    ):
        if store is None:
            raise HTTPException(status_code=500, detail="Store not loaded")
        global _added_since_reindex, _ingest_total, _ingest_done

        # If client sends total count once – reset counters
        if total is not None and total > 0:
            _ingest_total = total
            _ingest_done = 0

        store.add_capsules(capsules)
        _added_since_reindex += len(capsules)
        _ingest_done += len(capsules)

        # notify stream listeners – include progress if known
        payload = {"event": "update", "added": len(capsules)}
        if _ingest_total > 0:
            pct = int(100 * _ingest_done / _ingest_total)
            payload.update({"progress": pct})
        _broadcast(payload)

        if index_path:
            store.save(index_path)
        # auto reindex if threshold exceeded
        if _added_since_reindex >= REINDEX_THRESHOLD:
            import threading
            def _rebuild():
                if store is None:
                    return
                store.rebuild_index()
                store.save(index_path)
                _broadcast({"event": "reindex"})
            threading.Thread(target=_rebuild, daemon=True).start()
            _added_since_reindex = 0
        siglog.log({"type": "update", "added": len(capsules)})
        CAPSULE_TOTAL.set(len(store.meta))
        return {"added": len(capsules), "progress": payload.get("progress")}

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
        wt: float = 0.0,
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
            expanded = expand_with_links(results, store, depth=depth, limit=limit, weight_threshold=wt)
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
        CAPSULE_TOTAL.set(len(store.meta))
        
        siglog.log({"type": "prune", "removed": removed})
        return {"removed": removed}

    @app.post("/reindex")
    def reindex(model: str | None = None, factory: str | None = None, _auth: bool = Depends(require_api_key)):
        if store is None:
            raise HTTPException(status_code=500, detail="Store not loaded")
        
        store.rebuild_index(model_name=model, index_factory=factory)
        if index_path:
            store.save(index_path)
        
        CAPSULE_TOTAL.set(len(store.meta))
        siglog.log({"type": "reindex", "model": model, "factory": factory})
        return {"message": "Index rebuilt successfully"}

    @app.get("/events")
    async def events():
        from starlette.responses import StreamingResponse
        queue: asyncio.Queue[str] = asyncio.Queue()  # type: ignore
        _event_subscribers.append(queue)

        async def event_gen():
            try:
                while True:
                    data = await queue.get()
                    yield data
            except asyncio.CancelledError:
                pass
            finally:
                _event_subscribers.remove(queue)

        return StreamingResponse(event_gen(), media_type="text/event-stream")

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

# ---------------------------------------------------------------------------
# SSE broadcast helper
# ---------------------------------------------------------------------------

def _broadcast(message: dict[str, Any]):
    """Send JSON-encoded message to all active SSE subscribers."""
    if not _event_subscribers:
        return
    import json
    data = f"data: {json.dumps(message, ensure_ascii=False)}\n\n"
    for q in list(_event_subscribers):
        try:
            q.put_nowait(data)
        except Exception:
            pass
