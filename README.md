# SIGLA - Semantic Information Graph with Language Agents

A lightweight FAISS-backed capsule database with local model support for semantic search and knowledge management.

## Features

- **Local Model Support**: Use your own Transformers models for embeddings
- **FAISS Integration**: Efficient vector similarity search
- **Flexible Storage**: JSON metadata with binary index files
- **CLI Interface**: Easy command-line tools for ingestion and search
- **Web API**: FastAPI-based REST interface
- **DSL Functions**: High-level semantic operations (INTENT, RETRIEVE, MERGE, etc.)
- **Graph Operations**: Link-based capsule expansion and random walks

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Using Local Models

SIGLA automatically detects and uses local models in your directory:

```bash
# Auto-detect and use the best local model
python -m sigla ingest documents/ --auto-model

# Use a specific local model
python -m sigla ingest documents/ --local-model ./Qodo-Embed-1-7B

# Use sentence-transformers model (default)
python -m sigla ingest documents/ --model sentence-transformers/all-MiniLM-L6-v2
```

### Available Local Models

List available local models in your directory:

```bash
python -m sigla list-models
```

### Basic Usage

```bash
# Ingest documents
python -m sigla ingest documents/ -o my_store

# Search
python -m sigla search "your query" -s my_store -k 5

# Get store information
python -m sigla info -s my_store

# Start web server
python -m sigla serve -s my_store -p 8000
```

## Python API

```python
from sigla import CapsuleStore, create_store_with_best_local_model

# Create store with best available local model
store = create_store_with_best_local_model()

# Or use a specific local model
store = CapsuleStore.with_local_model("./Qodo-Embed-1-7B")

# Or use sentence-transformers
store = CapsuleStore(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Add documents
capsules = [
    {"text": "Machine learning is a subset of AI", "tags": ["ml", "ai"]},
    {"text": "Deep learning uses neural networks", "tags": ["dl", "neural"]},
]
store.add_capsules(capsules)

# Search
results = store.query("neural networks", top_k=3)
for result in results:
    print(f"Score: {result['score']:.3f}")
    print(f"Text: {result['text']}")
    print(f"Tags: {result.get('tags', [])}")
```

## DSL Functions

```python
from sigla import INTENT, RETRIEVE, MERGE, INJECT, EXPAND

# Extract intent
intent = INTENT("I want to learn about machine learning")

# Retrieve relevant capsules
results = RETRIEVE("machine learning", store, top_k=5)

# Merge multiple capsules
merged_text = MERGE(results, temperature=1.0)

# Inject new capsule
capsule_id = INJECT("New information about AI", store)

# Expand with related capsules
expanded = EXPAND(results[0], store, depth=2)
```

## Local Model Setup

### Supported Model Formats

SIGLA supports any Transformers-compatible model with the following structure:

```
model_directory/
├── config.json
├── pytorch_model.bin (or model.safetensors)
├── tokenizer.json
└── tokenizer_config.json
```

### Example Models

1. **ms-marco-MiniLM-L6-v2**: Compact, fast model for general use
2. **Qodo-Embed-1-7B**: Large, high-quality model for better results

### Device Support

```bash
# Auto-detect best device (CUDA > MPS > CPU)
python -m sigla ingest docs/ --device auto

# Force specific device
python -m sigla ingest docs/ --device cuda
python -m sigla ingest docs/ --device cpu
```

## Web API

Start the web server:

```bash
python -m sigla serve -s my_store -p 8000
```

API endpoints:

- `GET /`: Server info and store statistics
- `POST /search`: Search capsules with JSON body `{"query": "text", "top_k": 5}`

Example:

```bash
curl -X POST "http://localhost:8000/search" \
     -H "Content-Type: application/json" \
     -d '{"query": "machine learning", "top_k": 3}'
```

## Advanced Features

### Graph Operations

```python
from sigla.graph import expand_with_links, random_walk_links

# Expand capsules via their links
expanded = expand_with_links(results, store, depth=2, limit=20)

# Random walk through the graph
walked = random_walk_links(results, store, steps=5, restart=0.3)
```

### Custom Index Types

```python
# Use different FAISS index types
store = CapsuleStore(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    index_factory="IVF100,Flat"  # Faster search for large datasets
)
```

### Batch Operations

```python
# Add many capsules efficiently
large_capsule_list = [{"text": f"Document {i}"} for i in range(10000)]
store.add_capsules(large_capsule_list)

# Remove capsules by ID
store.remove_capsules([1, 3, 5])

# Rebuild index with new model
store.rebuild_index(model_name="local:./better_model")
```

## Performance Tips

1. **Use Local Models**: Local models avoid network overhead and provide consistent performance
2. **Choose Right Model Size**: Balance between quality (larger models) and speed (smaller models)  
3. **Batch Operations**: Add multiple capsules at once for better performance
4. **Index Types**: Use IVF indexes for large datasets (>10k capsules)
5. **Device Selection**: Use CUDA/MPS when available for faster encoding

## File Structure

```
my_store.index    # FAISS binary index
my_store.json     # Metadata and configuration
```

## Dependencies

### Core (required)
- `numpy>=1.21.0`
- `faiss-cpu>=1.7.0` 
- `sentence-transformers>=2.2.0`

### Local Models (recommended)
- `torch>=1.9.0`
- `transformers>=4.20.0`

### Web Server (optional)
- `fastapi>=0.68.0`
- `uvicorn>=0.15.0`

## License

MIT License - see LICENSE file for details.
