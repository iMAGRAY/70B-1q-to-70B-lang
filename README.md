# 70B-1q-to-70B-lang

This project experiments with **SIGLA**, a small-scale reasoning layer between a
lightweight model (**1Q**) and a larger language model (**70B**). 70B answers ar
e distilled into compact *capsules* and indexed using FAISS. At runtime, SIGLA r
etrieves relevant capsules and injects them into the 1Q model.

## Installation

Run `pip install faiss-cpu sentence-transformers fastapi uvicorn transformers` to install optional dependencies.

If you see an error about `faiss` being missing, install it explicitly:

```bash
pip install faiss-cpu
# or, for GPUs
pip install faiss-gpu
```
On systems where pip wheels are unavailable (for example on macOS), install via conda:

```bash
conda install -c conda-forge faiss-cpu
```

The first run also downloads the sentence-transformer model from the internet,
so ensure the machine has network access and that your cache directory is writable.

Commands that simply read the index (e.g. `list` or `info`) work without the
embedding model because `CapsuleStore` loads it lazily when needed.
If `sentence-transformers` isn't installed, SIGLA falls back to a simple
hash-based embedder so you can experiment without heavy dependencies.
You can also force this lightweight mode by passing `--model hash` when
ingesting or reindexing so no model download is required.

### Reality Check

SIGLA is designed to run on modest hardware. While the concept originates from
70B-scale models, day-to-day retrieval uses compact open-source embeddings and a
lightweight FAISS index. Heavy LLMs are only needed offline when producing the
initial knowledge base. You can run the CLI and server entirely on a CPU, or a
single consumer GPU, without renting expensive infrastructure.

For more background and long-term goals see `SIGLA_Plan.md`.

## Usage

1. Convert raw text from a larger model into capsules:

```bash
python -m sigla.scripts capsulate answers.txt caps.json --tags философия --source Claude
```

This splits the text into sentences, sanitizes them and saves JSON capsules.

2. Example capsule file:

```json
[
  {"text": "Стоики считали, что разум управляет эмоциями."},
  {"text": "Эпикур видел счастье в отсутствии страданий."}
]
```

3. Build an index (you can tag capsules and choose a FAISS index type):

```bash
python -m sigla.scripts ingest capsules.json myindex --factory HNSW32 --link 3 --tags philosophy --source Claude
# add --no-dedup to keep duplicate texts
# default factory is Flat
# pass --model hash if you don't have sentence-transformers installed
```
Sensitive tokens like email addresses or long numeric IDs are automatically
redacted during ingestion.
Duplicate sentences are skipped unless you pass `--no-dedup`.

Each capsule is assigned a numeric `id` so you can retrieve it later via the API.

4. Append new capsules to the same index later:

```bash
python -m sigla.scripts update more_caps.json myindex --link 3 --tags philosophy --source Claude
# add --no-dedup to allow identical capsules
```

Use this to grow the knowledge base without rebuilding the index.

5. Search for relevant capsules (you can filter by tags):

```bash
python -m sigla.scripts search myindex "философия и счастье" --tags философия
```

You can record queries by adding `--log-file logfile.jsonl`.

The resulting text can be injected into your model prompt or cached at a lower level. Use `--tags` with comma-separated values to restrict results.

6. Perform graph-based retrieval (if capsules include `links`):

```bash
python -m sigla.scripts walk myindex "философия" --depth 2 --limit 8 --algo random --tags философия
```

Use `--algo bfs` (default) to simply follow links breadth-first or `--algo random` with `--restart` to explore the graph via a random walk.
7. Generate a prompt snippet directly:

```bash
python -m sigla.scripts inject myindex "философия и счастье" --top_k 3 --tags философия --temperature 0.7
```

This prints the `[Контекст]` block ready to prepend to 1Q. Adjust `--temperature`
to tune how strongly the best capsules dominate the merge.

8. Generate a compressed summary of top capsules:

```bash
python -m sigla.scripts compress myindex "философия и счастье" --top_k 3 --tags философия
```

This attempts to summarize the retrieved capsules using a local summarization model.

9. Run the API server:

```bash
python -m sigla.server myindex
```

Now you can query it (including optional tags):

```bash
curl "http://localhost:8000/search?query=философия&tags=философия"
```

Or request a ready-to-inject context snippet:

```bash
curl "http://localhost:8000/ask?query=философия&temperature=0.7"
```

You can add more capsules on the fly by posting to `/update` (optionally auto-linking them with `link_neighbors`):

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '[{"text": "Новая мысль"}]' \
  "http://localhost:8000/update?link_neighbors=3"
```

Check the current index summary:

```bash
curl http://localhost:8000/info
```

List stored capsules (limit the number and filter by tags):

```bash
curl "http://localhost:8000/list?limit=5&tags=философия"
```

Export capsules via the API:

```bash
curl "http://localhost:8000/dump?limit=10&tags=философия" > dump.json
```

Walk linked capsules via the API:

```bash
curl "http://localhost:8000/walk?query=философия&depth=2&limit=8"
```

Get a Graphviz representation of capsule links:

```bash
curl "http://localhost:8000/graph?limit=100" > graph.dot
```

Summarize top capsules:

```bash
curl "http://localhost:8000/compress?query=философия&top_k=3"
```

Remove capsules via the API:

```bash
curl -X POST "http://localhost:8000/prune?ids=0,1&tags=философия"
```

Rebuild embeddings through the server:

```bash
curl -X POST "http://localhost:8000/reindex?model=sentence-transformers/all-MiniLM-L6-v2&factory=HNSW32"
```

Both the CLI and server accept a `--log-file` option to record queries and
updates in JSONL format. This is useful for building a memory of interactions:

```bash
python -m sigla.server myindex --log-file sigla.log
```

10. Start an interactive shell:

```bash
python -m sigla.scripts shell myindex --top_k 3 --tags философия --temperature 0.7
```

Type queries one per line; an empty line exits.

11. Show a stored capsule by its id:

```bash
python -m sigla.scripts capsule myindex 0
```

This prints the capsule's text and metadata in JSON form.

12. List stored capsules (optionally filter by tags):

```bash
python -m sigla.scripts list myindex --limit 5 --tags философия
```

13. Export capsules to a JSON file:

```bash
python -m sigla.scripts export myindex dump.json --tags философия
```

This saves matching capsules in `dump.json`.

14. Export the capsule graph to Graphviz DOT:

```bash
python -m sigla.scripts graph myindex graph.dot --limit 100
```

This writes `graph.dot` describing capsules and their links.

15. Summarize a log file to see how commands are used:

```bash
python -m sigla.scripts stats sigla.log
```

This prints a JSON object with counts for each logged event type.

16. Inspect index information:
```bash
python -m sigla.scripts info myindex
```

This lists the embedding model, dimension, capsule count and tag distribution.

17. Prune capsules by id or tags:

```bash
python -m sigla.scripts prune myindex --ids 0,1 --tags philosophy
```

This removes matching capsules and rebuilds the index.

18. Rebuild embeddings with a new model or index type:

```bash
python -m sigla.scripts reindex myindex --model sentence-transformers/all-MiniLM-L6-v2 --factory HNSW32  # optional
```

This recomputes all capsule vectors and updates the FAISS index.

### Using the SIGLA mini-language

`sigla.dsl` exposes helpers following the plan's INTENT → RETRIEVE → MERGE → INJECT pipeline. When capsules contain links, you can also `EXPAND` them. Example:

```python
from sigla import CapsuleStore, INTENT, RETRIEVE, MERGE, INJECT, EXPAND

# "lazy=True" delays loading the embedding model until needed
store = CapsuleStore(lazy=True)
store.load("myindex")
vec = INTENT(store, "философия и счастье")
caps = RETRIEVE(store, vec, top_k=3)
caps = EXPAND(caps, store, depth=1)
snippet = INJECT(MERGE(caps))
print(snippet)
```
This produces a prompt fragment ready to prepend to your 1Q model. You can also
call `store.embed_query("text")` directly if you only need the normalized
vector.

