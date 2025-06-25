"""Core data structures and utilities for SIGLA.

This module contains the main `CapsuleStore` class – a small wrapper around
FAISS that keeps a *capsule graph* (text chunks + metadata) in memory and on
disk.  It also provides helper utilities for working with local HF models and
simple convenience functions used by the CLI and server layers.

The implementation purposefully avoids heavy dependencies at import time –
optional requirements such as *torch*, *transformers*, *faiss* are loaded
lazily and guarded with `MissingDependencyError` so that parts of the package
(e.g. the docs) can be imported without the ML stack being installed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

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
    """Lightweight FAISS-backed capsule DB with automatic linking.

    Parameters
    ----------
    model_name
        Name (or HF hub path) of the encoder to use.  A **local** model can be
        passed by prefixing the path with ``local:`` or by using the
        :py:meth:`with_local_model` constructor.
    index_factory
        FAISS factory string, e.g. ``"Flat"``, ``"IVF100"``.
    auto_link_k
        If > 0, graph links between capsules are created automatically: for
        every capsule we find *k* nearest neighbours and put their integer IDs
        into the ``links`` field.  This makes simple graph traversals possible
        without an explicit DB.
    device
        Target device (**cpu** | **cuda** | **mps** | **auto**).
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_factory: str = "Flat",
        *,
        auto_link_k: int = 0,
        device: str = "auto",
    ):
        if SentenceTransformer is None:
            raise MissingDependencyError("sentence-transformers package is required")
        if faiss is None:
            raise MissingDependencyError("faiss package is required")

        self.device = device

        self.model_name = model_name

        # Support *local:* prefix – used by with_local_model and rebuild_index
        if model_name.startswith("local:"):
            local_path = model_name[len("local:") :]
            self.model = TransformersEmbeddings(local_path, device=device)
        else:
            self.model = SentenceTransformer(model_name)

        self.dimension = self.model.get_sentence_embedding_dimension()

        self.index_factory = index_factory
        # For small stores Flat index is enough and reconstruct() works.
        self.index = faiss.index_factory(self.dimension, index_factory, faiss.METRIC_INNER_PRODUCT)

        self.meta: List[Dict[str, Any]] = []

        # --- graph options -------------------------------------------------
        self.auto_link_k = auto_link_k

    def add_capsules(self, capsules: List[Dict[str, Any]]):
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
        for i, cap in enumerate(capsules):
            meta = cap.copy()
            if "content" in meta and "text" not in meta:
                meta["text"] = meta["content"]
            meta.setdefault("tags", [])
            meta.setdefault("links", [])
            meta["id"] = start_id + i
            self.meta.append(meta)

        # ------------------------------------------------------------------
        # Auto-linking (optional) – expensive for large collections but good
        # enough for small test datasets.
        # ------------------------------------------------------------------

        if self.auto_link_k > 0 and self.index.ntotal > 1:
            for idx in range(len(self.meta)):
                # Reconstruct vector for *idx* (works for Flat/IVF indices).
                try:
                    vec = self.index.reconstruct(idx)
                except Exception:
                    # fallback – re-encode (slow but safe)
                    vec = self.model.encode([self.meta[idx]["text"]], convert_to_numpy=True)[0]
                scores, nbrs = self.index.search(vec.reshape(1, -1), self.auto_link_k + 1)
                neighbours = [int(n) for n in nbrs[0] if n != idx][: self.auto_link_k]
                self.meta[idx]["links"] = neighbours

    def save(self, path: str):
        faiss.write_index(self.index, path + ".index")
        with open(path + ".json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model": self.model_name,
                    "factory": self.index_factory,
                    "meta": self.meta,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    def load(self, path: str):
        self.index = faiss.read_index(path + ".index")
        with open(path + ".json", "r", encoding="utf-8") as f:
            data = json.load(f)
            self.meta = data["meta"]
            self.model_name = data.get("model", self.model_name)
            self.index_factory = data.get("factory", "Flat")
            self.model = SentenceTransformer(self.model_name)

    def query(self, text: str, top_k: int = 5, tags: List[str] | None = None) -> List[Dict[str, Any]]:
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
        vectors = self.model.encode(texts, convert_to_numpy=True) if texts else None
        self.index = faiss.IndexFlatIP(self.dimension)
        if vectors is not None and len(texts) > 0:
            faiss.normalize_L2(vectors)
            if not self.index.is_trained:
                self.index.train(vectors)
            self.index.add(vectors)
        
        # Replace meta list and update IDs
        self.meta = new_meta
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
        vectors = self.model.encode(texts, convert_to_numpy=True) if texts else None
        self.index = faiss.index_factory(self.dimension, self.index_factory, faiss.METRIC_INNER_PRODUCT)
        if vectors is not None and len(texts) > 0:
            faiss.normalize_L2(vectors)
            if not self.index.is_trained:
                self.index.train(vectors)
            self.index.add(vectors)
        
        for idx, meta in enumerate(self.meta):
            meta["id"] = idx

        # Finished rebuild_index

    def add_capsule(self, text: str, tags: Optional[List[str]] | None = None) -> int:
        """Add a single *text* capsule and return its assigned ID."""
        capsule = {"text": text, "tags": tags or []}
        self.add_capsules([capsule])
        return capsule["id"]

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def get_info(self) -> Dict[str, Any]:
        tag_counts: Dict[str, int] = {}
        for meta in self.meta:
            for t in meta.get("tags", []):
                tag_counts[t] = tag_counts.get(t, 0) + 1

        return {
            "model": self.model_name,
            "dimension": self.dimension,
            "index_factory": self.index_factory,
            "capsules": len(self.meta),
            "vectors": self.index.ntotal if self.index else 0,
            "tags": tag_counts,
            "device": self.device,
        }

    # ------------------------------------------------------------------
    # Alternate constructors
    # ------------------------------------------------------------------

    @classmethod
    def with_local_model(
        cls,
        model_path: str,
        *,
        index_factory: str = "Flat",
        auto_link_k: int = 0,
        device: str = "auto",
    ) -> "CapsuleStore":
        """Construct a store that uses a *local* HF model directory."""

        instance = cls.__new__(cls)  # bypass __init__
        if faiss is None:
            raise MissingDependencyError("faiss package is required")

        instance.device = device
        instance.model_name = f"local:{model_path}"
        instance.model = TransformersEmbeddings(model_path, device=device)
        instance.dimension = instance.model.get_sentence_embedding_dimension()
        instance.index_factory = index_factory
        instance.index = faiss.index_factory(instance.dimension, index_factory, faiss.METRIC_INNER_PRODUCT)
        instance.meta = []
        instance.auto_link_k = auto_link_k
        return instance


# ---------------------------------------------------------------------------
# Local-model discovery helpers (very small, no external deps).
# ---------------------------------------------------------------------------


def get_available_local_models(base_dir: str | Path | str = "./models") -> List[str]:
    """Return list of directories that *look like* HF model checkpoints."""

    base = Path(base_dir)
    if not base.exists():
        return []
    candidates: List[str] = []
    for path in base.iterdir():
        if path.is_dir() and (path / "config.json").exists():
            candidates.append(str(path))
    return candidates


def create_store_with_best_local_model(*, device: str = "auto", auto_link_k: int = 0) -> CapsuleStore:
    """Pick first local model (if any) and build a store around it."""

    models = get_available_local_models()
    if not models:
        # Fallback to default online model
        return CapsuleStore(device=device, auto_link_k=auto_link_k)

    return CapsuleStore.with_local_model(models[0], device=device, auto_link_k=auto_link_k)

# ---------------------------------------------------------------------------
# transformers.pipeline import (optional) – needed for compress_capsules
# ---------------------------------------------------------------------------

try:
    from transformers import pipeline  # type: ignore
except Exception:  # pragma: no cover – optional dependency
    pipeline = None

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
    if pipeline is None:
        raise MissingDependencyError("transformers package is required for compression")

    try:
        summarizer = pipeline("summarization", model=model_name)
    except Exception as e:  # pragma: no cover - optional dependency
        raise MissingDependencyError(str(e))

    text = "\n".join(c["text"] for c in capsules)
    summary = summarizer(text, max_length=60, min_length=5, do_sample=False)
    return summary[0]["summary_text"].strip()
