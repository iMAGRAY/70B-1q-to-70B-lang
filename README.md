# 70B-1q-to-70B-lang

This project experiments with **SIGLA**, a small-scale reasoning layer between a
lightweight model (**1Q**) and a larger language model (**70B**). 70B answers ar
e distilled into compact *capsules* and indexed using FAISS. At runtime, SIGLA r
etrieves relevant capsules and injects them into the 1Q model.

## Usage

1. Prepare a JSON file with capsules of the following form:

```json
[
  {"text": "Стоики считали, что разум управляет эмоциями."},
  {"text": "Эпикур видел счастье в отсутствии страданий."}
]
```

2. Build an index:

```bash
python -m sigla.scripts ingest capsules.json myindex
```

3. Search for relevant capsules:

```bash
python -m sigla.scripts search myindex "философия и счастье"
```

The resulting text can be injected into your model prompt or cached at a lower level.

4. Run the API server:

```bash
python -m sigla.server myindex
```

Now you can query it:

```bash
curl "http://localhost:8000/search?query=философия"
```

You can add more capsules on the fly by posting to `/update`:

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '[{"text": "Новая мысль"}]' \
  http://localhost:8000/update
```

