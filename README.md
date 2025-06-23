# 70B-1q-to-70B-lang

This project experiments with **SIGLA**, a small-scale reasoning layer between a
lightweight model (**1Q**) and a larger language model (**70B**). 70B answers ar
e distilled into compact *capsules* and indexed using FAISS. At runtime, SIGLA r
etrieves relevant capsules and injects them into the 1Q model.

## Installation

Run `pip install faiss-cpu sentence-transformers fastapi uvicorn transformers` to install optional dependencies.

## Usage

1. Prepare a JSON file with capsules of the following form:

```json
[
  {"text": "Стоики считали, что разум управляет эмоциями."},
  {"text": "Эпикур видел счастье в отсутствии страданий."}
]
```

xvy4pj-codex/разработать-sigla-для-моделирования-мышления
2. Build an index:

```bash
python -m sigla.scripts ingest capsules.json myindex
=======
2. Build an index (you can choose a FAISS index type with `--factory`):

```bash
python -m sigla.scripts ingest capsules.json myindex --factory HNSW32  # default is Flat
main
```

Each capsule is assigned a numeric `id` so you can retrieve it later via the API.

3. Search for relevant capsules (you can filter by tags):

```bash
python -m sigla.scripts search myindex "философия и счастье" --tags философия
```

You can record queries by adding `--log-file logfile.jsonl`.

The resulting text can be injected into your model prompt or cached at a lower level. Use `--tags` with comma-separated values to restrict results.

4. Perform graph-based retrieval (if capsules include `links`):

```bash
xvy4pj-codex/разработать-sigla-для-моделирования-мышления
python -m sigla.scripts walk myindex "философия" --depth 2 --limit 8 --tags философия
```

This expands results by following capsule links.
=======
python -m sigla.scripts walk myindex "философия" --depth 2 --limit 8 --algo random --tags философия
```

Use `--algo bfs` (default) to simply follow links breadth-first or `--algo random` with `--restart` to explore the graph via a random walk.
main
5. Generate a prompt snippet directly:

```bash
python -m sigla.scripts inject myindex "философия и счастье" --top_k 3 --tags философия
```

This prints the `[Контекст]` block ready to prepend to 1Q.

6. Generate a compressed summary of top capsules:

```bash
python -m sigla.scripts compress myindex "философия и счастье" --top_k 3 --tags философия
```

This attempts to summarize the retrieved capsules using a local summarization model.

7. Run the API server:

```bash
python -m sigla.server myindex
```

Now you can query it (including optional tags):

```bash
curl "http://localhost:8000/search?query=философия&tags=философия"
```

You can add more capsules on the fly by posting to `/update`:

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '[{"text": "Новая мысль"}]' \
  http://localhost:8000/update
```

Both the CLI and server accept a `--log-file` option to record queries and
updates in JSONL format. This is useful for building a memory of interactions:

```bash
python -m sigla.server myindex --log-file sigla.log
```

8. Start an interactive shell:

```bash
python -m sigla.scripts shell myindex --top_k 3 --tags философия
```

Type queries one per line; an empty line exits.

9. Show a stored capsule by its id:

```bash
python -m sigla.scripts capsule myindex 0
```

This prints the capsule's text and metadata in JSON form.

10. List stored capsules (optionally filter by tags):

```bash
python -m sigla.scripts list myindex --limit 5 --tags философия
```

11. Summarize a log file to see how commands are used:

```bash
python -m sigla.scripts stats sigla.log
```

This prints a JSON object with counts for each logged event type.

12. Inspect index information:
```bash
python -m sigla.scripts info myindex
```

This lists the embedding model, dimension, capsule count and tag distribution.

13. Prune capsules by id or tags:

```bash
python -m sigla.scripts prune myindex --ids 0,1 --tags philosophy
```

This removes matching capsules and rebuilds the index.

xvy4pj-codex/разработать-sigla-для-моделирования-мышления
=======
14. Rebuild embeddings with a new model or index type:

```bash
python -m sigla.scripts reindex myindex --model sentence-transformers/all-MiniLM-L6-v2 --factory HNSW32  # optional
```

This recomputes all capsule vectors and updates the FAISS index.

main
### Using the SIGLA mini-language

`sigla.dsl` exposes helpers following the plan's INTENT → RETRIEVE → MERGE → INJECT pipeline. When capsules contain links, you can also `EXPAND` them. Example:

```python
from sigla import CapsuleStore, INTENT, RETRIEVE, MERGE, INJECT, EXPAND

store = CapsuleStore()
store.load("myindex")
vec = INTENT(store, "философия и счастье")
caps = RETRIEVE(store, vec, top_k=3)
caps = EXPAND(caps, store, depth=1)
snippet = INJECT(MERGE(caps))
print(snippet)
```
This produces a prompt fragment ready to prepend to your 1Q model.

