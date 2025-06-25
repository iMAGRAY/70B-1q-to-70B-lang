import json
import os
from typing import List, Dict, Any, Optional, Union, cast
from pathlib import Path

try:
    import faiss  # type: ignore
except ImportError:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    AutoTokenizer = None
    AutoModel = None
    torch = None
    F = None
    TORCH_AVAILABLE = False

import numpy as np


class MissingDependencyError(RuntimeError):
    """Raised when optional dependencies are not available."""
    pass


class TransformersEmbeddings:
    """Embedding wrapper for local Transformers models."""
    
    def __init__(self, model_path: str, device: str = "auto"):
        if not TORCH_AVAILABLE or AutoTokenizer is None or AutoModel is None:
            raise MissingDependencyError("transformers and torch packages are required")
        
        self.model_path = model_path
        self.device = self._get_device(device)
        
        print(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()
        
        # Get embedding dimension
        assert torch is not None  # For type checker
        with torch.no_grad():
            test_input = self.tokenizer("test", return_tensors="pt", padding=True, truncation=True)
            test_input = {k: v.to(self.device) for k, v in test_input.items()}
            test_output = self.model(**test_input)
            # Try different ways to get embeddings
            if hasattr(test_output, 'last_hidden_state'):
                self.dimension = test_output.last_hidden_state.shape[-1]
            elif hasattr(test_output, 'pooler_output'):
                self.dimension = test_output.pooler_output.shape[-1]
            else:
                # Fallback - assume standard BERT-like output
                self.dimension = test_output[0].shape[-1]
        
        print(f"Model loaded. Dimension: {self.dimension}")
    
    def _get_device(self, device: str) -> str:
        """Determine the best device to use."""
        if not TORCH_AVAILABLE or torch is None:
            return "cpu"
            
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def _mean_pooling(self, model_output, attention_mask):
        """Apply mean pooling to get sentence embeddings."""
        assert torch is not None  # For type checker
        token_embeddings = model_output[0]  # First element contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        convert_to_numpy: bool = True,
    ) -> Union[np.ndarray, Any]:
        """Encode texts to embeddings."""
        if not TORCH_AVAILABLE or torch is None or F is None:
            raise MissingDependencyError("torch is required for encoding")
            
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize
                encoded_input = self.tokenizer(
                    batch_texts, 
                    padding=True, 
                    truncation=True, 
                    return_tensors='pt',
                    max_length=512
                )
                encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
                
                # Get model output
                model_output = self.model(**encoded_input)
                
                # Get embeddings - try different methods
                if hasattr(model_output, 'pooler_output') and model_output.pooler_output is not None:
                    # Use pooler output if available
                    embeddings = model_output.pooler_output
                elif hasattr(model_output, 'last_hidden_state'):
                    # Use mean pooling of last hidden state
                    embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
                else:
                    # Fallback - mean pool the first output
                    embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
                
                # Normalize embeddings
                embeddings = F.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings.cpu())
        
        # Concatenate all embeddings
        all_embeddings = torch.cat(all_embeddings, dim=0)
        
        if convert_to_numpy:
            return all_embeddings.cpu().numpy()
        return all_embeddings
    
    def get_sentence_embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        return self.dimension


class CapsuleStore:
    """A lightweight FAISS-backed capsule database with local model support."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_factory: str = "Flat",
        local_model_path: Optional[str] = None,
        device: str = "auto",
        auto_link_k: int = 5,
        meta_backend: str = "memory",  # "memory" | "sqlite"
        meta_path: Optional[str] = None,
    ) -> None:
        if faiss is None:
            raise MissingDependencyError("faiss package is required")

        self.model_name = model_name
        self.index_factory = index_factory
        self.local_model_path = local_model_path
        self.device = device
        self.auto_link_k = max(0, int(auto_link_k))
        
        # Initialize model
        if local_model_path and os.path.exists(local_model_path):
            print(f"Using local model: {local_model_path}")
            self.model = TransformersEmbeddings(local_model_path, device=device)
            self.model_name = f"local:{local_model_path}"
        else:
            if SentenceTransformer is None:
                raise MissingDependencyError("sentence-transformers package is required")
            print(f"Using Sentence Transformer: {model_name}")
            self.model = SentenceTransformer(model_name)
        
        self.dimension = self.model.get_sentence_embedding_dimension()
        # If 'auto' – postpone creation until we know dataset size
        if index_factory != "auto":
            self.index = faiss.index_factory(
                self.dimension, index_factory, faiss.METRIC_INNER_PRODUCT
            )  # type: ignore[attr-defined]
        else:
            # start with Flat index placeholder; will switch automatically
            self.index = faiss.index_factory(
                self.dimension, "Flat", faiss.METRIC_INNER_PRODUCT
            )  # type: ignore[attr-defined]

        # ------------------------------------------------------------------
        # Meta storage backend – keeps capsule dictionaries.  The default
        # In-memory backend preserves historical behaviour where
        # ``CapsuleStore.meta`` is a plain Python list so existing code/tests
        # continue to work unchanged.
        # ------------------------------------------------------------------

        from .meta import InMemoryMetaStore, SQLiteMetaStore  # local import

        if meta_backend == "sqlite":
            try:
                self._meta_store = SQLiteMetaStore(meta_path or "capsules.sqlite3")
            except Exception as e:  # pragma: no cover – fallback safety
                print(f"[warn] SQLiteMetaStore init failed ({e}), falling back to memory")
                self._meta_store = InMemoryMetaStore()
        else:
            # default
            self._meta_store = InMemoryMetaStore()

        self.meta: List[Dict[str, Any]] = cast(List[Dict[str, Any]], self._meta_store.all())  # legacy alias
        
        print(f"CapsuleStore initialized:")
        print(f"  Model: {self.model_name}")
        print(f"  Dimension: {self.dimension}")
        print(f"  Index: {index_factory}")

    @classmethod
    def with_local_model(cls, model_path: str, **kwargs):
        """Create CapsuleStore with local model."""
        return cls(local_model_path=model_path, **kwargs)

    def add_capsule(self, content: str, tags: Optional[List[str]] = None, links: Optional[List[int]] = None) -> int:
        """Add a single capsule and return its ID."""
        capsule = {
            "text": content,
            "tags": tags or [],
            "links": links or []
        }
        self.add_capsules([capsule])
        return len(self.meta) - 1

    def add_capsules(self, capsules: List[Dict[str, Any]]) -> None:
        """Embed and add capsules to the index, assigning IDs."""
        if not capsules:
            return
            
        if faiss is None:
            raise MissingDependencyError("faiss package is required")
            
        start_id = len(self.meta)
        texts = [c.get("text", c.get("content", "")) for c in capsules]
        
        print(f"Encoding {len(texts)} texts...")
        vectors = self.model.encode(texts, convert_to_numpy=True)
        
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(vectors)
        
        # Train index if needed
        if not self.index.is_trained:
            print("Training index...")
            self.index.train(vectors)
        
        # Add vectors to index
        self.index.add(vectors)
        print(f"Added {len(vectors)} vectors to index")

        # auto index upgrade if needed
        self._maybe_switch_index()

        new_metas: List[Dict[str, Any]] = []
        for i, cap in enumerate(capsules):
            meta = cap.copy()
            if "content" in meta and "text" not in meta:
                meta["text"] = meta["content"]
            meta.setdefault("tags", [])
            meta.setdefault("links", [])
            meta["id"] = start_id + i
            new_metas.append(meta)

        # Persist – append to meta store
        try:
            self._meta_store.add_many(new_metas)  # type: ignore[attr-defined]
        except AttributeError:
            # fallback (for backends without add_many)
            for m in new_metas:
                self._meta_store.add_many([m])  # type: ignore[attr-defined]

        self.meta = cast(List[Dict[str, Any]], self._meta_store.all())

        # ------------------------------------------------------------------
        # Automatic linking: для каждой новой капсулы ищем ближайшие top_k
        # существующих и добавляем взаимные ссылки. Работает только если
        # было минимум одно «старое» в базе.
        # ------------------------------------------------------------------
        if self.auto_link_k > 0 and start_id > 0:
            for new_local_idx in range(len(capsules)):
                global_idx = start_id + new_local_idx

                # Берём вектор новой капсулы (векторы идут в том же порядке)
                new_vec = vectors[new_local_idx : new_local_idx + 1]
                _, neigh = self.index.search(new_vec, self.auto_link_k + 1)

                for n_idx in neigh[0]:
                    if n_idx == -1 or n_idx == global_idx:
                        continue
                    # Добавляем двустороннюю связь
                    if n_idx not in self.meta[global_idx]["links"]:
                        self.meta[global_idx]["links"].append(int(n_idx))
                    if global_idx not in self.meta[n_idx]["links"]:
                        self.meta[n_idx]["links"].append(int(global_idx))

    def query(
        self, text: str, top_k: int = 5, tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Return top matching capsules, optionally filtering by tags."""
        if self.index.ntotal == 0:
            return []
            
        if faiss is None:
            raise MissingDependencyError("faiss package is required")
            
        vector = self.model.encode([text], convert_to_numpy=True)
        faiss.normalize_L2(vector)
        
        # Search with oversampling for tag filtering
        search_k = min(top_k * 5, self.index.ntotal)
        scores, indices = self.index.search(vector, search_k)
        
        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx >= len(self.meta):
                continue
                
            meta = self.meta[idx]
            
            # Filter by tags if specified
            if tags and not set(tags).intersection(meta.get("tags", [])):
                continue
            
            cap = meta.copy()
            # Ensure consistent field naming
            if "text" not in cap and "content" in cap:
                cap["text"] = cap["content"]
            elif "content" not in cap and "text" in cap:
                cap["content"] = cap["text"]
            
            cap.update({"score": float(score), "id": int(idx)})
            results.append(cap)
            
            if len(results) >= top_k:
                break
                
        return results

    def save(self, path: str) -> None:
        """Save index (*.index) and metadata (*.json)."""
        if faiss is None:
            raise MissingDependencyError("faiss package is required")
            
        index_path = f"{path}.index"
        meta_path = f"{path}.json"
        
        faiss.write_index(self.index, index_path)
        
        save_data = {
            "model": self.model_name,
            "factory": self.index_factory,
            "meta": self.meta,
            "device": getattr(self, 'device', 'auto')
        }
        
        with open(meta_path, "w", encoding="utf-8") as fp:
            json.dump(save_data, fp, ensure_ascii=False, indent=2)
        
        print(f"Saved index to {index_path} and metadata to {meta_path}")

    def load(self, path: str) -> None:
        """Load index and metadata."""
        if faiss is None:
            raise MissingDependencyError("faiss package is required")
            
        index_path = f"{path}.index"
        meta_path = f"{path}.json"
        
        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            raise FileNotFoundError(f"Index files not found: {index_path} or {meta_path}")
        
        self.index = faiss.read_index(index_path)
        
        with open(meta_path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        
        self.meta = data["meta"]
        # Sync meta storage backend
        if hasattr(self, "_meta_store"):
            try:
                self._meta_store._replace_all(self.meta)  # type: ignore[attr-defined]
            except Exception:
                pass
        saved_model = data.get("model", self.model_name)
        self.index_factory = data.get("factory", "Flat")
        self.device = data.get("device", "auto")
        
        # Reinitialize model if different
        if saved_model != self.model_name:
            if saved_model.startswith("local:"):
                local_path = saved_model[6:]  # Remove "local:" prefix
                if os.path.exists(local_path):
                    self.model = TransformersEmbeddings(local_path, device=self.device)
                else:
                    print(f"Warning: Local model path not found: {local_path}")
                    print("Falling back to default model")
                    if SentenceTransformer is None:
                        raise MissingDependencyError("sentence-transformers package is required")
                    self.model = SentenceTransformer(self.model_name)
            else:
                if SentenceTransformer is None:
                    raise MissingDependencyError("sentence-transformers package is required")
                self.model = SentenceTransformer(saved_model)
                self.model_name = saved_model
        
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"Loaded index with {self.index.ntotal} vectors and {len(self.meta)} metadata entries")

    def remove_capsules(self, ids: List[int]) -> int:
        """Remove capsules by id, rebuilding the index."""
        if not ids:
            return 0
            
        if faiss is None:
            raise MissingDependencyError("faiss package is required")
            
        to_remove = set(ids)
        new_meta: List[Dict[str, Any]] = []
        mapping: Dict[int, int] = {}
        texts: List[str] = []

        for old_id, meta in enumerate(self.meta):
            if old_id in to_remove:
                continue
            mapping[old_id] = len(new_meta)
            copy = meta.copy()
            new_meta.append(copy)
            texts.append(copy["text"])

        # Update links
        for meta in new_meta:
            meta["links"] = [mapping[l] for l in meta.get("links", []) if l in mapping]

        # replace meta-store contents; will update alias
        self._meta_store._replace_all(new_meta)  # type: ignore[attr-defined]  # safe for in-memory/sqlite
        self.meta = cast(List[Dict[str, Any]], self._meta_store.all())
        
        # Rebuild index
        self.index = faiss.index_factory(self.dimension, self.index_factory, faiss.METRIC_INNER_PRODUCT)  # type: ignore[attr-defined]
        if texts:
            print(f"Rebuilding index with {len(texts)} texts...")
            vectors = self.model.encode(texts, convert_to_numpy=True)
            faiss.normalize_L2(vectors)
            if not self.index.is_trained:
                self.index.train(vectors)
            self.index.add(vectors)
        
        # Update IDs
        for new_id, meta in enumerate(self.meta):
            meta["id"] = new_id
            
        print(f"Removed {len(ids)} capsules")
        return len(ids)

    def rebuild_index(
        self, model_name: Optional[str] = None, index_factory: Optional[str] = None
    ) -> None:
        """Recompute all embeddings and rebuild the FAISS index."""
        if faiss is None:
            raise MissingDependencyError("faiss package is required")
            
        if model_name and model_name != self.model_name:
            if model_name.startswith("local:"):
                local_path = model_name[6:]
                self.model = TransformersEmbeddings(local_path, device=self.device)
            else:
                if SentenceTransformer is None:
                    raise MissingDependencyError("sentence-transformers package is required")
                self.model = SentenceTransformer(model_name)
            self.model_name = model_name
            self.dimension = self.model.get_sentence_embedding_dimension()
            
        if index_factory:
            self.index_factory = index_factory
            
        texts = [m["text"] for m in self.meta]
        self.index = faiss.index_factory(self.dimension, self.index_factory, faiss.METRIC_INNER_PRODUCT)  # type: ignore[attr-defined]
        
        if texts:
            print(f"Rebuilding index with {len(texts)} texts...")
            vectors = self.model.encode(texts, convert_to_numpy=True)
            faiss.normalize_L2(vectors)
            if not self.index.is_trained:
                self.index.train(vectors)
            self.index.add(vectors)
            
        for idx, meta in enumerate(self.meta):
            meta["id"] = idx
        
        print("Index rebuilt successfully")

    def get_info(self) -> Dict[str, Any]:
        """Get information about the store."""
        tag_counts: Dict[str, int] = {}
        for m in self.meta:
            for t in m.get("tags", []):
                tag_counts[t] = tag_counts.get(t, 0) + 1
        
        return {
            "model": self.model_name,
            "dimension": self.dimension,
            "index_factory": self.index_factory,
            "capsules": len(self.meta),
            "vectors": self.index.ntotal,
            "tags": tag_counts,
            "device": getattr(self, 'device', 'auto')
        }

    # ---------------------------------------------------------------
    # Internal helper: choose index based on corpus size when using
    # index_factory='auto'. For small corpora Flat is fine; для больших
    # (> 30k) используем IVF.
    # ---------------------------------------------------------------

    def _maybe_switch_index(self):
        if self.index_factory != "auto":
            return

        total = self.index.ntotal
        # thresholds can be tuned
        if total < 30000:
            desired = "Flat"
        elif total < 200000:
            desired = "IVF100,Flat"
        else:
            desired = "IVF400,Flat"

        # already correct type?
        if self.index_factory == desired or self.index.__class__.__name__.startswith("IndexIVF") and desired.startswith("IVF"):
            return

        print(f"[auto] Switching index to {desired} for {total} vectors…")
        # extract all vectors
        vectors = self.index.reconstruct_n(0, total) if total else None  # type: ignore[attr-defined]
        self.index = faiss.index_factory(self.dimension, desired, faiss.METRIC_INNER_PRODUCT)  # type: ignore[arg-type]
        if vectors is not None and len(vectors):
            if not self.index.is_trained:
                self.index.train(vectors)  # type: ignore[attr-defined]
            self.index.add(vectors)  # type: ignore[attr-defined]
        self.index_factory = desired


def merge_capsules(capsules: List[Dict[str, Any]], temperature: float = 1.0) -> str:
    """Merge capsule texts using softmax weighting by score."""
    if not capsules:
        return ""
    
    def get_text(cap: Dict[str, Any]) -> str:
        return cap.get("text", cap.get("content", ""))
    
    if len(capsules) == 1:
        return get_text(capsules[0])
    
    # Get scores and texts
    scores = np.array([cap.get("score", 1.0) for cap in capsules])
    texts = [get_text(cap) for cap in capsules]
    
    # Apply softmax with temperature
    exp_scores = np.exp(scores / temperature)
    weights = exp_scores / np.sum(exp_scores)
    
    # Create weighted combination
    result_parts = []
    for i, (text, weight) in enumerate(zip(texts, weights)):
        if weight > 0.1:  # Only include significant contributions
            result_parts.append(text)
    
    return "\n\n".join(result_parts)


def compress_capsules(
    capsules: List[Dict[str, Any]],
    model_name: str = "facebook/bart-large-cnn",
) -> str:
    """Compress capsules by merging capsule texts into a concise summary.

    The function tries to use a Transformers summarisation pipeline when the
    `transformers` package is available (and a compatible model can be
    downloaded or found locally).  If that is not possible, it falls back to
    a simple but robust heuristic: keep the first and last 150 characters with
    an ellipsis in-between.  This guarantees that the function always returns
    *something useful* without adding a hard dependency.
    """
    if not capsules:
        return ""
    
    def get_text(cap: Dict[str, Any]) -> str:
        return cap.get("text", cap.get("content", ""))
    
    # Merge all texts
    combined_text = "\n\n".join(get_text(cap) for cap in capsules)

    # --- Advanced summarisation --------------------------------------------------
    # Если текст слишком большой (> 4000 символов), используем "скользящее окно":
    #   1. Бьём текст на чанки ~1024 символа с overlap 200.
    #   2. Саммаризируем каждый чанк.
    #   3. Саммаризируем объединение всех промежуточных саммари.
    # Это снижает накладные расходы по памяти/токенам и даёт более полное
    # содержание.
    # ---------------------------------------------------------------------------

    try:
        from transformers import pipeline  # type: ignore

        summariser = pipeline(
            "summarization",
            model=model_name,
            truncation=True,
        )

        def _summ(text: str) -> str:
            out = summariser(
                text,
                max_length=180,
                min_length=60,
                do_sample=False,
            )
            if isinstance(out, list) and out:
                return (
                    out[0].get("summary_text")
                    or out[0].get("generated_text")
                    or ""
                ).strip()
            return ""

        if len(combined_text) > 4000:
            chunk_size = 1024
            overlap = 200
            chunks = []
            start = 0
            while start < len(combined_text):
                end = start + chunk_size
                chunk = combined_text[start:end]
                chunks.append(chunk)
                start = end - overlap  # slide with overlap

            partial_summaries = [_summ(c) for c in chunks if c.strip()]
            combined_summary = "\n".join(partial_summaries)
            final_summary = _summ(combined_summary)
            if final_summary:
                return final_summary
        else:
            summary = _summ(combined_text)
            if summary:
                return summary
    except (ImportError, Exception):
        # Either transformers is not installed or the model could not be
        # loaded – fall back to a deterministic truncation routine.
        pass

    # Fallback: simple intelligent truncation
    if len(combined_text) <= 300:
        return combined_text

    first_part = combined_text[:150]
    last_part = combined_text[-150:]

    return f"{first_part}...\n\n...{last_part}"


# Auto-detect local models
def get_available_local_models() -> List[str]:
    """Get list of available local models in the current directory."""
    current_dir = Path(".")
    models = []
    
    search_dirs = [current_dir, current_dir / "models"]

    for base in search_dirs:
        if not base.exists():
            continue
        for item in base.iterdir():
            if not item.is_dir() or item.name.startswith('.'):
                continue
            # Check looks like a HF model dir
            if (item / "config.json").exists() and (
                (item / "pytorch_model.bin").exists()
                or (item / "model.safetensors").exists()
                or any(item.glob("model-*.safetensors"))
            ):
                models.append(str(item))
    
    return sorted(models)


def create_store_with_best_local_model(**kwargs) -> CapsuleStore:
    """Create CapsuleStore with the best available local model."""
    local_models = get_available_local_models()
    
    if not local_models:
        print("No local models found, using default sentence-transformers model")
        return CapsuleStore(**kwargs)
    
    # Prefer larger models (assuming better quality)
    # Sort by directory size (rough approximation)
    def get_dir_size(path_str: str) -> int:
        path = Path(path_str)
        total_size = 0
        try:
            for file in path.rglob("*"):
                if file.is_file():
                    total_size += file.stat().st_size
        except:
            pass
        return total_size
    
    local_models.sort(key=get_dir_size, reverse=True)
    best_model = local_models[0]
    
    print(f"Found {len(local_models)} local models, using: {best_model}")
    return CapsuleStore.with_local_model(best_model, **kwargs)
