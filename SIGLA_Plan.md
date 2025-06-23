# SIGLA Development Roadmap (Revisited)
This roadmap prioritizes low-cost local solutions. Large proprietary models and expensive servers are avoided or used only for occasional offline preprocessing.


This version updates the initial roadmap after reviewing several unrealistic
assumptions.

## Key issues found
- **Raw logs from 70B** – storing complete answers is expensive and risks data leaks. Only short, sanitized capsules will be kept.
- **Heavy LLM usage** – queries to proprietary 70B models are costly. Limit their use and rely mainly on local open-source models.
- **Graph-based retrieval by default** – building a CapsuleGraph from day one complicates the system. Start with simple kNN retrieval and expand later.
- **Autoencoder compression** – training an autoencoder requires additional data and tuning. Using an LLM summarizer is simpler early on.
- **Direct KV-cache injection** – depends on the serving stack. Prompt injection is the stable method while KV experiments run in parallel.
1. **Collect Intent List**
   - Brainstorm common themes, questions and tasks the system must handle.
   - Classify each intent by expected value and complexity.
2. **Query 70B or Similar Models**
   - Use small open-source LLMs (7B–13B) running locally whenever possible. Reserve 70B queries for one-time or offline generation of high-value capsules.
3. **Capsule Extraction**
   - Break answers into atomic statements containing one fact or reasoning step.
   - Each statement becomes a capsule with minimal text.
4. **Embedding and Storage**
   - Use efficient open-source embedding models (e.g., E5 or Llama2-based).
   - Verify critical capsules by comparing with 70B embeddings when possible.
   - Store vectors and metadata (source, tags, quality rating) in a FAISS index. Use `Flat` by default but allow `HNSW` or `IVF` factories for larger datasets.
   - Retrieval multiplies similarity by this rating so important capsules rank higher.

## 2. SIGLA Core
1. **Embedding Requests**
   - Convert user questions into intent vectors using the chosen model.
2. **Retrieval Pipeline**
   - Query FAISS for nearest capsules (kNN).
   - Once basic retrieval quality is measured, optionally expand results using
     a CapsuleGraph.
3. **Capsule Fusion**
   - Weight capsules by relevance and connection strength.
   - Combine a small set of capsules into a single capsule-thought with soft
     attention. Adjust the number dynamically based on token budget.
4. **Interface Functions**
   - Expose operations like `embed_query`, `retrieve_capsules`, and `merge_capsules` as part of a Python module.
   - Initial version implemented in `sigla/core.py` with a FAISS-backed `CapsuleStore`.

## 3. Injecting Thoughts into 1Q
1. **Prompt Method**
   - Insert capsule-thought texts in the prompt using concise templates.
   - Keep total prompt length below model limits.
2. **KV-Cache Method (Experimental)**
   - Only if the serving framework exposes a stable API for KV injection.
   - Convert capsule vectors to the required tensor format and prepend them
     before the user's question.
3. **Choosing a Strategy**
   - Start with prompt injection for rapid iteration.
   - Introduce cache-based injection for efficiency once the pipeline is stable.

## 4. Improving Retrieval and Reasoning
1. **Graph-Based Expansion (Optional)**
   - When simple retrieval misses context, build a CapsuleGraph linking related
     capsules.
   - Explore random walk or BFS to gather supporting capsules.
2. **Capsule Compression**
   - Summarize dense capsules with an LLM instead of training a custom autoencoder.
3. **Reasoning Capsules**
   - Store not just facts but causal or conditional statements.
   - Encourage consistent reasoning patterns when combining capsules.

## 5. Formalizing SIGLA
1. **Mini-Language Syntax**
   - `INTENT(text) -> vector`
   - `RETRIEVE(vector) -> [capsules]`
   - `MERGE(list) -> capsule-thought`
   - `INJECT(capsule-thought) -> model`
2. **Memory Tracking**
   - Log queries and results to grow a long-term memory store.
   - Visualize capsule graphs to debug coverage and quality.

## 6. API and Server Implementation
1. **FastAPI Service**
   - `/ask`: main entry for questions returning 1Q's final answer.
   - `/capsule/{id}`: inspect stored capsules.
2. **Model Connectors**
   - Run local models using CPU-friendly tools like `llama.cpp` or `ggml`. Avoid renting expensive servers.
   - Provide optional adapters for external APIs like Claude or GPT-4 only if the budget allows.
   
3. **Monitoring and Fallbacks**
   - Track latency, number of retrieved capsules, and token count.
   - If retrieval confidence is low, optionally query the model directly or use
     a simpler RAG step.

## 7. Evaluation and Iteration
1. **A/B Testing**
   - Compare 1Q answers with and without SIGLA on sample tasks.
   - Collect user feedback to refine capsule selection.
2. **Index Maintenance**
   - Periodically recompute embeddings and rebuild FAISS indices.
3. **Security Checks**
   - Filter sensitive or unwanted content in capsules.
   - Ensure no personal data from user queries is stored without consent.
## Reality Check
- Ensure each step can run on commodity hardware (CPU or single consumer GPU).
- Keep prompts and capsule storage small to control disk and memory use.
- Regularly reevaluate whether any feature adds clear value for its cost.

## 8. Final Objective
- 1Q approaches the depth of a 70B model using SIGLA capsules without requiring costly servers.
- SIGLA evolves into a modular system that grows memory and reasoning abilities over time.

### Implementation Progress
- `sigla/core.py` provides an initial FAISS-based capsule store with embedding and search.
- `sigla/scripts.py` offers CLI commands to ingest capsules and run searches, and can append to an existing index.
- `sigla/server.py` exposes a FastAPI service for querying and updating the capsule index.
- `sigla/dsl.py` implements INTENT/RETRIEVE/MERGE/INJECT helpers for prompt construction.
- Capsules now receive persistent `id`s and an optional `links` field for building a graph.
- Graph expansion is provided via `sigla.graph.expand_with_links` and the CLI `walk` command.
- Random walk retrieval is implemented via `sigla.graph.random_walk_links` and selectable in the CLI `walk` command.
- The DSL exposes `EXPAND` for link-based retrieval.
- `sigla/log.py` enables optional JSONL query logging for both the CLI and server.
- `sigla/scripts.py` now includes an interactive `shell` command for quick manual tests.
- `sigla/scripts.py` can display a capsule by id via the `capsule` command.
 - `sigla/scripts.py` can summarize log files via the `stats` command, displaying counts and average durations.
- Search, inject, walk and shell commands accept `--tags` to filter results by metadata; the server exposes a matching query parameter.
- Queries can also be restricted by `--sources` to target specific capsule origins.
- `sigla/scripts.py` can show index details via the `info` command.
- `sigla/scripts.py` can list stored capsules via the `list` command.
- `sigla/scripts.py` can remove capsules via the `prune` command.
- `sigla/scripts.py` can summarize retrieved capsules via the `compress` command.
- The `compress` command and `/compress` endpoint allow tuning summary length via
  `--max-length` and `--min-length` options.
- `sigla/scripts.py` can rebuild embeddings via the `reindex` command.
- `sigla/scripts.py` can append capsules via the `update` command; the server
  exposes `/update` for the same purpose.
- `sigla/scripts.py` can convert raw text to capsules via the `capsulate` command.
- Capsules ingested or updated with `--link` automatically connect to nearest neighbors for graph retrieval.
- Ingestion and reindexing support custom FAISS index factories via `--factory`.
- `inject` and `shell` commands, as well as the `/ask` endpoint, accept a
  `temperature` parameter controlling how capsules are merged.
- The API exposes `/info` and `/list` endpoints mirroring the CLI commands.
- `/walk` and `/compress` endpoints support graph expansion and summarization.
- `/prune` and `/reindex` endpoints mirror CLI commands for capsule removal and index rebuilding.
- `sigla/scripts.py` can export capsules via the `export` command; the server provides a `/dump` endpoint.
- `CapsuleStore` loads the embedding model lazily so read-only commands start quickly.
- `sigla/scripts.py` can export capsule links to Graphviz DOT via the `graph` command.
- The API exposes a `/graph` endpoint returning DOT data for visualization.
- Capsule ingestion sanitizes obvious personal data like emails and long numeric
  sequences before storage.
- Capsules can be assigned a numeric rating via `--rating`, influencing search ranking.
- Search and injection commands accept `--min-rating` to skip low-quality capsules.
- Ratings can be adjusted after ingestion via the new `rate` command.
- The API exposes a `/rate` endpoint for the same purpose.
- Capsule metadata (rating and tags) can be modified via the `meta` command and the `/meta` endpoint.
- `CapsuleStore` exposes `embed_query` and `embed_texts` for direct vectorization; `INTENT` uses these helpers.
- If `sentence-transformers` isn't installed, `CapsuleStore` falls back to a
  lightweight hash-based embedder so the CLI works without heavy dependencies.
- `sigla/scripts.py` can output embeddings via the `embed` command and the server
  exposes a matching `/embed` endpoint.
- The package can be invoked as `python -m sigla` thanks to a new `__main__` module.
- Log files now record the `duration` of each command or API call for performance tracking.
- `sigla cache` clears summarizer and embedding caches to free memory when needed.
- `sigla/scripts.py` can compare two texts via the `similarity` command for quick checks.
- The server provides a `/similarity` endpoint returning cosine similarity.
- `list` and `export` commands support filtering by source and rating; the API
  offers the same via `/list` and `/dump` parameters.
- Most commands use `$SIGLA_INDEX` as the default index path if no argument is
  given.
- `sigla/llm.py` wraps local language models via `llama-cpp` or `transformers`.
- The `answer` command and `/answer` endpoint retrieve context and generate a response using a local model.
- `sigla version` prints the package version.
