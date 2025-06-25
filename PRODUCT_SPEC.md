# SIGLA — Semantic Information Graph & Lightweight Agents

**Version:** 0.9 – Internal developer preview  
**Status:** All core features implemented; production‐readiness requires expanded tests & security audit.

---

## 1. Executive Summary
SIGLA is a self-contained local knowledge-graph and retrieval-augmented generation (RAG) engine that helps small/medium LLMs reason with an external memory of *capsules* (atomic facts, reasoning steps, or code snippets).  
It bridges the gap between consumer-grade 7-13B models and 70B-class responses by:

*   Building a **semantic index** (FAISS) over capsule embeddings.
*   Expanding context with a lightweight **CapsuleGraph**.
*   Merging relevant capsules into a concise **prompt or KV-prefix**.
*   Serving answers over a **FastAPI** micro-service and **CLI**.

Everything runs on commodity hardware (CPU or single consumer-GPU) with no proprietary dependencies.

---

## 2. Core Concepts
| Term | Description |
|------|-------------|
| **Capsule** | Dict with `text`, optional `tags`, `links`, `weights`, `source`, `timestamp`. Represents one atomic piece of knowledge. |
| **CapsuleStore** | Wrapper around FAISS index + `meta` list. Handles ingestion, querying, auto-linking, save/load, prune, reindex. |
| **CapsuleGraph** | Lightweight graph over capsules; edges weighted by semantic similarity. Two expansion algos: BFS (`expand_with_links`) & weighted random walk (`random_walk_links`). |
| **DSL** | Helper mini-language (`INTENT`, `RETRIEVE`, `MERGE`, `INJECT`, `EXPAND`, `ANALYZE`) for building prompts in pipelines/notebooks. |
| **KV-prefix** | List of token IDs injected before user prompt; returned by `/ask?kv=true` and accepted by `sigla.runner --prefix`. |

---

## 3. Architecture Diagram

```mermaid
flowchart TD
    A[Client / CLI] --HTTP/CLI--> B[FastAPI Server]
    B -- query --> C[CapsuleStore (FAISS + meta)]
    C -- list/expand --> D[CapsuleGraph]
    C -- embeddings --> E[HF Embedding Model]
    B -- merge --> F[Prompt Builder]
    F -- context --> G[LLM Runner]
```

---

## 4. Components & Responsibilities
1. **sigla.core**  
   * Embedding back-ends (Sentence-Transformers, local HF).  
   * Index factories (`Flat`, `IVF`, `HNSW`).  
   * CRUD on capsules; bidirectional auto-linking with weights.  
   * Persistence (`.index` + `.json`).
2. **sigla.graph**  
   * BFS / Weighted random walk expansion.  
   * Weight threshold filtering.
3. **sigla.server**  
   * FastAPI with endpoints: `/search`, `/ask`, `/capsule/{id}`, `/update`, `/info`, `/list`, `/walk`, `/compress`, `/prune`, `/reindex`, `/metrics`, `/events` (SSE).  
   * Optional API-key auth (`X-API-Key`).  
   * Prometheus metrics (`requests`, `latency`, `capsules_total`).  
   * Auto-reindex after 500 new capsules (configurable).  
   * SSE pushes `update`, `reindex` events.
4. **sigla.scripts (CLI)**  
   * `ingest`, `search`, `info`, `list-models`, `serve`, `convert`, `run-cg`, `dsl`, `module`, `list`, `capsule`, `compress`, `prune`, `reindex`, `shell`, `stats`, `abtest`.
5. **sigla.runner**  
   * Loads `.capsulegraph` archives.  
   * Generates text with optional token prefix (`--prefix`).
6. **sigla.registry**  
   * SQLite + optional Neo4j module registry.

---

## 5. API Specification (FastAPI)
| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/search?query=&top_k=&tags=` | kNN search, returns capsule list with `score`. |
| `GET` | `/ask?query=&kv=true` | RAG answer context. Returns `{context, low_conf, kv, tokens}`. |
| `GET` | `/capsule/{id}` | Retrieve capsule metadata. Requires auth. |
| `POST` | `/update` | Add capsules `[ {text, ...}, ... ]`. Requires auth. |
| `GET` | `/info` | Index summary. |
| `GET` | `/list?limit=&tags=` | List capsules. Requires auth. |
| `GET` | `/walk?query=&algo=bfs|random&wt=` | Graph expansion. |
| `GET` | `/compress?query=` | Summarize retrieved capsules. |
| `POST` | `/prune` | Remove capsules by ids/tags. Requires auth. |
| `POST` | `/reindex` | Rebuild embeddings; optional model/factory. Requires auth. |
| `GET` | `/metrics` | Prometheus exposition. |
| `GET` | `/events` | Server-Sent Events stream (`update`, `reindex`). |

---

## 6. CLI Cheat-Sheet
```bash
# Ingest directory                           
sigla ingest docs/ --output wiki
# Run API server                             
sigla serve --store wiki --port 8080
# Query                                      
curl "localhost:8080/ask?query=Who+is+Ada+Lovelace"
# Walk graph                                 
curl ".../walk?query=AI&algo=random&wt=0.4"
# Interactive shell                          
sigla shell --store wiki --top-k 8
# A/B evaluation                             
sigla abtest qa_dataset.json -s wiki -k 6 -t 0.7
```

---

## 7. Storage Formats
* **Index** — `name.index` (FAISS)
* **Meta** — `name.json` (fields above)
* **CapsuleGraph archive** — `.capsulegraph` tar.gz containing:
  * `quant_model.pth`, `meta.json`, `token_caps.index/json`.

---

## 8. Metrics
| Name | Type | Labels | Meaning |
|------|------|--------|---------|
| `sigla_requests_total` | Counter | `endpoint` | API call count |
| `sigla_request_latency_seconds` | Histogram | `endpoint` | Latency |
| `sigla_capsules_total` | Gauge | — | Current capsule count |

---

## 9. Security & Compliance
* Optional **API-key** via env `SIGLA_API_KEY`.  
* Capsules may contain user data → comply with GDPR: pruning, export planned.  
* No external calls by default; offline-friendly.

---

## 10. Roadmap Snapshot
1. Expand test coverage (integration, property-based).  
2. Improve weight decay & temporal relevance.  
3. Web UI (Vue/React) using `/events` for live status.  
4. Pluggable embedding & LLM adapters (llama.cpp, GGUF).  
5. Production hardening: rate-limits, CORS, RBAC.

---

> Document generated automatically — keep in repo root (`PRODUCT_SPEC.md`). 