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
    from transformers import AutoTokenizer, AutoModel, pipeline
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    AutoTokenizer = None
    AutoModel = None
    pipeline = None
    torch = None
    F = None
    TORCH_AVAILABLE = False

import numpy as np


class MissingDependencyError(Exception):
    """Raised when optional dependency is missing but required for operation."""
    
    def __init__(self, dependency: str, operation: str = "this operation"):
        self.dependency = dependency
        self.operation = operation
        super().__init__(
            f"Missing dependency '{dependency}' required for {operation}. "
            f"Install with: pip install {dependency}"
        )


class TransformersEmbeddings:
    """Simple wrapper for HuggingFace transformer models for embeddings."""

    def __init__(self, model_path: str, device: str = "auto"):
        """Initialize with a local model path or HF model name."""
        if not TORCH_AVAILABLE:
            raise MissingDependencyError("torch and transformers packages are required")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.device = self._get_device(device)
        self.model.to(self.device)

    def _get_device(self, device: str) -> str:
        """Determine the torch device to use."""
        if device != "auto":
            return device
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling operation to get sentence embeddings."""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        convert_to_numpy: bool = True,
    ) -> Union[np.ndarray, Any]:
        """Encode a list of texts into embeddings."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded_input = self.tokenizer(
                batch, padding=True, truncation=True, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                model_output = self.model(**encoded_input)

            embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"])
            embeddings = F.normalize(embeddings, p=2, dim=1)

            if convert_to_numpy:
                embeddings = embeddings.cpu().numpy()

            all_embeddings.append(embeddings)

        if convert_to_numpy:
            return np.vstack(all_embeddings)
        else:
            return torch.cat(all_embeddings, dim=0)

    def get_sentence_embedding_dimension(self) -> int:
        """Return the dimension of the sentence embeddings."""
        return self.model.config.hidden_size


class CapsuleStore:
    """A lightweight FAISS-backed capsule database.

    This class provides a simple wrapper around FAISS for storing and querying
    text "capsules" (text chunks with metadata) using semantic similarity.

    Each capsule is a dict with at least a "text" field and optional metadata
    such as "tags", "links", etc.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_factory: str = "Flat",
        *,
        auto_link_k: int = 0,
        device: str = "auto",
    ):
        """Initialize a new capsule store.

        Args:
            model_name: Name of the sentence-transformers model to use
            index_factory: FAISS index factory string
            auto_link_k: Number of similar capsules to auto-link (0 to disable)
            device: Device to use for embeddings ("auto", "cpu", "cuda", etc.)
        """
        if faiss is None:
            raise MissingDependencyError("faiss package is required")

        self.device = device
        self.model_name = model_name
        self.index_factory = index_factory
        self.auto_link_k = auto_link_k
        self.meta: List[Dict[str, Any]] = []

        # Initialize model based on type
        if model_name.startswith("local:"):
            local_path = model_name[6:]
            self.model = TransformersEmbeddings(local_path, device=device)
            self.dimension = self.model.get_sentence_embedding_dimension()
        else:
            if SentenceTransformer is None:
                raise MissingDependencyError("sentence-transformers package is required")
            self.model = SentenceTransformer(model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()

        # Initialize empty FAISS index
        self.index = faiss.index_factory(self.dimension, index_factory, faiss.METRIC_INNER_PRODUCT)

    def add_capsules(
        self,
        capsules: List[Dict[str, Any]],
        *,
        vectors: "np.ndarray | None" = None,
    ) -> None:
        """Add capsules to the index.

        Parameters
        ----------
        capsules : list[dict]
            Capsule dictionaries with at least the ``text`` field.
        vectors : np.ndarray | None, optional
            Pre-computed embedding vectors shaped ``(N, dim)``.  When provided
            we skip the costly ``model.encode`` call – useful for bulk loading
            when embeddings are already available (e.g. token embeddings in
            :func:`sigla.converter.convert_to_capsulegraph`).
        """
        if not capsules:
            return

        start_id = len(self.meta)

        # Compute embeddings if not provided
        if vectors is None:
            texts = [c["text"] for c in capsules]
            vectors = self.model.encode(texts, convert_to_numpy=True)

        import numpy as np  # local import to avoid hard dep in type stubs
        if not isinstance(vectors, np.ndarray):  # type: ignore[comparison-overlap]
            raise TypeError("vectors must be a numpy.ndarray")

        if vectors.shape[0] != len(capsules):
            raise ValueError("vectors.shape[0] must match len(capsules)")

        faiss.normalize_L2(vectors)

        # Train index if needed
        if not self.index.is_trained and vectors.shape[0] > 0:
            self.index.train(vectors)

        # Add vectors to index
        self.index.add(vectors)

        # Process metadata
        for i, cap in enumerate(capsules):
            meta = cap.copy()
            meta.setdefault("links", [])
            meta["id"] = start_id + i
            self.meta.append(meta)

        # Auto-link if enabled
        if self.auto_link_k > 0 and len(self.meta) > 1:
            self._update_auto_links(start_id, len(capsules))

    def _update_auto_links(self, start_idx: int, count: int) -> None:
        """Update auto-links for new capsules."""
        # Get vectors for new capsules
        new_texts = [self.meta[i]["text"] for i in range(start_idx, start_idx + count)]
        new_vectors = self.model.encode(new_texts, convert_to_numpy=True)
        faiss.normalize_L2(new_vectors)

        # For each new capsule, find similar existing capsules
        for i, vec in enumerate(new_vectors):
            idx = start_idx + i
            scores, indices = self.index.search(vec.reshape(1, -1), self.auto_link_k + 1)
            # Add links (skip self-link)
            for j, sim_idx in enumerate(indices[0]):
                if sim_idx != idx and sim_idx >= 0:
                    self.meta[idx]["links"].append(int(sim_idx))
                    # Store similarity score as weight
                    w_list = self.meta[idx].setdefault("weights", [])
                    w_list.append(float(scores[0][j]))
                    # add reciprocal link
                    back_links = self.meta[sim_idx].setdefault("links", [])
                    back_weights = self.meta[sim_idx].setdefault("weights", [])
                    if idx not in back_links:
                        back_links.append(idx)
                        back_weights.append(float(scores[0][j]))

    def save(self, path: str):
        """Save the index and metadata to disk."""
        faiss.write_index(self.index, path + ".index")
        with open(path + ".json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model": self.model_name,
                    "factory": self.index_factory,
                    "auto_link_k": self.auto_link_k,
                    "meta": self.meta,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    def load(self, path: str):
        """Load the index and metadata from disk."""
        self.index = faiss.read_index(path + ".index")
        with open(path + ".json", "r", encoding="utf-8") as f:
            data = json.load(f)
            self.meta = data["meta"]
            self.model_name = data.get("model", self.model_name)
            self.index_factory = data.get("factory", "Flat")
            self.auto_link_k = data.get("auto_link_k", 0)

        # Re-initialize the model
        if self.model_name.startswith("local:"):
            local_path = self.model_name[6:]
            self.model = TransformersEmbeddings(local_path, device=self.device)
        else:
            if SentenceTransformer is None:
                raise MissingDependencyError("sentence-transformers package is required")
            self.model = SentenceTransformer(self.model_name)

    def query(self, text: str, top_k: int = 5, tags: List[str] | None = None) -> List[Dict[str, Any]]:
        """Return top matching capsules, optionally filtering by tags."""
        vector = self.model.encode([text], convert_to_numpy=True)
        faiss.normalize_L2(vector)
        # oversample to account for tag filtering
        scores, indices = self.index.search(vector, top_k * 5)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta = self.meta[idx]
            if tags and not set(tags).intersection(meta.get("tags", [])):
                continue
            cap = meta.copy()
            cap["score"] = float(score)
            cap["id"] = idx
            results.append(cap)
            if len(results) >= top_k:
                break
        return results

    def remove_capsules(self, ids: List[int]) -> int:
        """Remove capsules by id, rebuilding the index."""
        if not ids:
            return 0
        to_remove = set(ids)
        mapping: Dict[int, int] = {}
        new_meta: List[Dict[str, Any]] = []
        texts: List[str] = []
        for old_id, meta in enumerate(self.meta):
            if old_id in to_remove:
                continue
            mapping[old_id] = len(new_meta)
            copy = meta.copy()
            new_meta.append(copy)
            texts.append(copy["text"])
        for meta in new_meta:
            meta["links"] = [mapping[l] for l in meta.get("links", []) if l in mapping]
        vectors = self.model.encode(texts, convert_to_numpy=True) if texts else None
        self.index = faiss.index_factory(self.dimension, self.index_factory, faiss.METRIC_INNER_PRODUCT)
        if vectors is not None and len(texts) > 0:
            faiss.normalize_L2(vectors)
            if not self.index.is_trained:
                self.index.train(vectors)
            self.index.add(vectors)
        for new_id, meta in enumerate(new_meta):
            meta["id"] = new_id
        removed = len(self.meta) - len(new_meta)
        self.meta = new_meta
        return removed

    def rebuild_index(self, model_name: str | None = None, index_factory: str | None = None) -> None:
        """Recompute all embeddings and rebuild the FAISS index."""
            
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
    
    # Combine all texts
    texts = [c.get("text", "") for c in capsules]
    combined = " ".join(texts)
    
    try:
        if pipeline is None:
            raise MissingDependencyError("transformers package is required for compression")
        
        # Use summarization pipeline
        summarizer = pipeline("summarization", model=model_name)
        summary = summarizer(combined, max_length=150, min_length=30, do_sample=False)
        
        if summary and summary[0]["summary_text"]:
            return summary[0]["summary_text"]
    except Exception:
        pass  # Fall back to simple truncation
    
    # Simple fallback: truncate with ellipsis
    if len(combined) > 300:
        return combined[:150] + "..." + combined[-150:]
    return combined 