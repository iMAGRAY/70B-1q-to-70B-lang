# 70B-1q-to-70B-lang

This project experiments with **SIGLA**, a small-scale reasoning layer between a
lightweight model (**1Q**) and a larger language model (**70B**). 70B answers ar
e distilled into compact *capsules* and indexed using FAISS. At runtime, SIGLA r
etrieves relevant capsules and injects them into the 1Q model.

## Installation

Run `pip install faiss-cpu sentence-transformers fastapi uvicorn transformers llama-cpp-python` to install optional dependencies.

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

Verify installation:

```bash
python -m sigla version
```

The first run also downloads the sentence-transformer model from the internet,
so ensure the machine has network access and that your cache directory is writable.

Commands that simply read the index (e.g. `list` or `info`) work without the
embedding model because `CapsuleStore` loads it lazily when needed.
If `sentence-transformers` isn't installed, SIGLA falls back to a simple
hash-based embedder so you can experiment without heavy dependencies.
You can also force this lightweight mode by passing `--model hash` when
ingesting or reindexing so no model download is required.

To use the new `answer` command or the `/answer` endpoint you must have a local language model installed via `llama-cpp-python` or `transformers`.

### Reality Check

SIGLA is designed to run on modest hardware. While the concept originates from
70B-scale models, day-to-day retrieval uses compact open-source embeddings and a
lightweight FAISS index. Heavy LLMs are only needed offline when producing the
initial knowledge base. You can run the CLI and server entirely on a CPU, or a
single consumer GPU, without renting expensive infrastructure.

For more background and long-term goals see `SIGLA_Plan.md`.

## Usage

Run the CLI as `python -m sigla` (a shorthand for `python -m sigla.scripts`).

1. Convert raw text from a larger model into capsules:

```bash
python -m sigla capsulate answers.txt caps.json --tags философия --source Claude
# or use '-' to read from stdin or write to stdout
cat answers.txt | python -m sigla capsulate - - > caps.json
```

This splits the text into sentences, sanitizes them and saves JSON capsules.

2. Example capsule file:

```json
[
  {"text": "Стоики считали, что разум управляет эмоциями."},
  {"text": "Эпикур видел счастье в отсутствии страданий."}
]
```
You can also provide capsules one per line without brackets:

```json
{"text": "Стоики считали, что разум управляет эмоциями."}
{"text": "Эпикур видел счастье в отсутствии страданий."}
```

3. Build an index (you can tag capsules, assign a rating and choose a FAISS index type):

```bash
# rating influences ranking during search (higher is more important)
python -m sigla ingest capsules.json myindex --factory HNSW32 --link 3 --tags philosophy --source Claude --rating 1.2
# add --no-dedup to keep duplicate texts
# default factory is Flat
# pass --model hash if you don't have sentence-transformers installed
```
To avoid repeating the index path for later commands, set an environment variable:

```bash
export SIGLA_INDEX=myindex
```
Sensitive tokens like email addresses or long numeric IDs are automatically
redacted during ingestion.
Duplicate sentences are skipped unless you pass `--no-dedup`.
You can assign a numeric `--rating` to favor important capsules during search.

Each capsule is assigned a numeric `id` so you can retrieve it later via the API.

4. Append new capsules to the same index later:

```bash
python -m sigla update more_caps.json myindex --link 3 --tags philosophy --source Claude --rating 1.2
# add --no-dedup to allow identical capsules
```

Use this to grow the knowledge base without rebuilding the index.

5. Search for relevant capsules (you can filter by tags, source and rating):

```bash
python -m sigla search myindex "философия и счастье" --tags философия --sources Claude --min-rating 0.8
```

You can record queries by adding `--log-file logfile.jsonl`.
Each entry includes a `duration` field showing how long the operation took.

Capsules with higher `rating` values rank above others with the same similarity. Use `--min-rating` to ignore lower-quality capsules.

The resulting text can be injected into your model prompt or cached at a lower level. Use `--tags` with comma-separated values to restrict results.

6. Perform graph-based retrieval (if capsules include `links`):

```bash
python -m sigla walk myindex "философия" --depth 2 --limit 8 --algo random --tags философия
```

Use `--algo bfs` (default) to simply follow links breadth-first or `--algo random` with `--restart` to explore the graph via a random walk.
7. Generate a prompt snippet directly:

```bash
python -m sigla inject myindex "философия и счастье" --top_k 3 --tags философия --sources Claude --temperature 0.7
# ignore capsules rated below 0.5
python -m sigla inject myindex "философия и счастье" --top_k 3 --tags философия --sources Claude --temperature 0.7 --min-rating 0.5
```

This prints the `[Контекст]` block ready to prepend to 1Q. Adjust `--temperature`
to tune how strongly the best capsules dominate the merge.

8. Generate a compressed summary of top capsules:

```bash
python -m sigla compress myindex "философия и счастье" --top_k 3 \
  --tags философия --sources Claude --max-length 80
python -m sigla compress myindex "философия и счастье" --top_k 3 \
  --tags философия --sources Claude --min-rating 0.5 --max-length 80 --min-length 10
```

This attempts to summarize the retrieved capsules using a local summarization model.
The summarizer is cached after the first use so subsequent calls avoid reloading
the transformers pipeline.
Call `sigla.clear_summarizer_cache()` to release the cached model if needed.
Use `--max-length` and `--min-length` to control the size of the summary.
If memory usage grows, clear caches via the CLI:

```bash
python -m sigla cache myindex --embeddings --summarizer
```

9. Run the API server:

```bash
python -m sigla.server myindex
```

Now you can query it (including optional tags):

```bash
curl "http://localhost:8000/search?query=философия&tags=философия&sources=Claude"
curl "http://localhost:8000/search?query=философия&tags=философия&sources=Claude&min_rating=0.8"
```

Or request a ready-to-inject context snippet:

```bash
curl "http://localhost:8000/ask?query=философия&sources=Claude&temperature=0.7"
curl "http://localhost:8000/ask?query=философия&sources=Claude&temperature=0.7&min_rating=0.8"
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

List stored capsules (limit the number and filter by tags, sources and rating):

```bash
curl "http://localhost:8000/list?limit=5&tags=философия&sources=Claude&min_rating=1.0"
```

Export capsules via the API (same filters):

```bash
curl "http://localhost:8000/dump?limit=10&tags=философия&sources=Claude&min_rating=1.0" > dump.json
```

Walk linked capsules via the API:

```bash
curl "http://localhost:8000/walk?query=философия&depth=2&limit=8&sources=Claude"
```

Get a Graphviz representation of capsule links:

```bash
curl "http://localhost:8000/graph?limit=100" > graph.dot
```

Summarize top capsules:

```bash
curl "http://localhost:8000/compress?query=философия&top_k=3&sources=Claude&max_length=80"
curl "http://localhost:8000/compress?query=философия&top_k=3&sources=Claude&min_rating=0.8&max_length=80&min_length=10"
```
The `max_length` and `min_length` parameters let you tune the summary length.

Ask a local model for a final answer:

```bash
curl "http://localhost:8000/answer?query=философия&model_path=tinyllama.gguf&top_k=3"
```

Этот эндпоинт извлекает контекст и сразу генерирует полный ответ через локальный LLM.

Remove capsules via the API:

```bash
curl -X POST "http://localhost:8000/prune?ids=0,1&tags=философия"
```

Rebuild embeddings through the server:

```bash
curl -X POST "http://localhost:8000/reindex?model=sentence-transformers/all-MiniLM-L6-v2&factory=HNSW32"
```

Update capsule ratings via the API:

```bash
curl -X POST "http://localhost:8000/rate?ids=0,1&rating=1.5"
```
You can also update capsules matching certain tags:

```bash
curl -X POST "http://localhost:8000/rate?tags=философия&rating=0.8"
```

Modify metadata and tags through the API:

```bash
curl -X POST "http://localhost:8000/meta?ids=2,3&add=important,history"
curl -X POST "http://localhost:8000/meta?tags=философия&remove=draft&rating=0.8"
```

Request an embedding vector for arbitrary text:

```bash
curl "http://localhost:8000/embed?text=пример"
```

Both the CLI and server accept a `--log-file` option to record queries and
updates in JSONL format. Each record stores the elapsed time under the
`duration` key. This is useful for building a memory of interactions:

```bash
python -m sigla.server myindex --log-file sigla.log
```

Compute similarity between two texts via the API:

```bash
curl "http://localhost:8000/similarity?text_a=текст1&text_b=текст2"
```

10. Start an interactive shell:

```bash
python -m sigla shell myindex --top_k 3 --tags философия --temperature 0.7
```

Type queries one per line; an empty line exits.

11. Show a stored capsule by its id:

```bash
python -m sigla capsule myindex 0
```

This prints the capsule's text and metadata in JSON form.

12. List stored capsules (optionally filter by tags, sources or rating):

```bash
python -m sigla list myindex --limit 5 --tags философия --sources Claude --min-rating 1.0
```

13. Export capsules to a JSON file (with the same filters):

```bash
python -m sigla export myindex dump.json --tags философия --sources Claude --min-rating 1.0
```

This saves matching capsules in `dump.json`.

14. Export the capsule graph to Graphviz DOT:

```bash
python -m sigla graph myindex graph.dot --limit 100
```

This writes `graph.dot` describing capsules and their links.

15. Summarize a log file to see how commands are used and how long they take:

```bash
python -m sigla stats sigla.log
```

This prints a JSON object with counts and the average `duration` for each event type.

16. Inspect index information:
```bash
python -m sigla info myindex
```

This lists the embedding model, dimension, capsule count and tag distribution.

17. Prune capsules by id or tags:

```bash
python -m sigla prune myindex --ids 0,1 --tags philosophy
```

This removes matching capsules and rebuilds the index.

18. Rebuild embeddings with a new model or index type:

```bash
python -m sigla reindex myindex --model sentence-transformers/all-MiniLM-L6-v2 --factory HNSW32  # optional
```

This recomputes all capsule vectors and updates the FAISS index.

19. Get an embedding vector for any text:

```bash
python -m sigla embed "пример текста" --model sentence-transformers/all-MiniLM-L6-v2
# or use an existing index to reuse its embedding model
python -m sigla embed "другой текст" --index myindex
```

This prints the vector as JSON so you can reuse it elsewhere.

20. Adjust capsule ratings:

```bash
python -m sigla rate myindex --ids 0,1 --rating 1.5
```

Use `--tags` instead of `--ids` to update groups of capsules.

21. Modify capsule metadata (rating or tags):

```bash
python -m sigla meta myindex --ids 2,3 --add-tags important,history
python -m sigla meta myindex --tags философия --remove-tags draft --rating 0.8
```

This command lets you adjust ratings and add or remove tags after ingestion.

22. Clear cached models and embeddings when memory is tight:

```bash
python -m sigla cache myindex --embeddings --summarizer
```

23. Compare similarity between two texts:

```bash
python -m sigla similarity "текст один" "текст два" --model hash
```

This prints a cosine similarity score between the inputs.

24. Generate an answer using a local language model:

```bash
python -m sigla answer myindex "кто такие стоики" --model-path tinyllama.gguf --top_k 3 --temperature 0.7
```

`--model-path` указывает файл весов для `llama-cpp` или имя модели Hugging Face. Команда извлекает контекст и печатает полный ответ.

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

