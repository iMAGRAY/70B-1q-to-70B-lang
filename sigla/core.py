import json
from typing import List, Dict, Any

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None

try:
    from transformers import pipeline
except Exception:  # pragma: no cover - optional dependency
    pipeline = None


class MissingDependencyError(RuntimeError):
    """Raised when optional dependencies are not available."""
    pass


class CapsuleStore:
    """A lightweight FAISS-backed capsule database."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        if SentenceTransformer is None:
            raise MissingDependencyError("sentence-transformers package is required")
        if faiss is None:
            raise MissingDependencyError("faiss package is required")

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dimension)
        self.meta: List[Dict[str, Any]] = []

    def add_capsules(self, capsules: List[Dict[str, Any]]):
        """Embed and add capsules to the index, assigning IDs."""
        start = len(self.meta)
        texts = [c["text"] for c in capsules]
        vectors = self.model.encode(texts, convert_to_numpy=True)
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        for i, cap in enumerate(capsules):
            meta = cap.copy()
            meta.setdefault("links", [])
            meta["id"] = start + i
            self.meta.append(meta)

    def save(self, path: str):
        faiss.write_index(self.index, path + ".index")
        with open(path + ".json", "w", encoding="utf-8") as f:
            json.dump({"model": self.model_name, "meta": self.meta}, f, ensure_ascii=False, indent=2)

    def load(self, path: str):
        self.index = faiss.read_index(path + ".index")
        with open(path + ".json", "r", encoding="utf-8") as f:
            data = json.load(f)
            self.meta = data["meta"]
            self.model_name = data.get("model", self.model_name)
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
        self.index = faiss.IndexFlatIP(self.dimension)
        if vectors is not None and len(texts) > 0:
            faiss.normalize_L2(vectors)
            self.index.add(vectors)
        for new_id, meta in enumerate(new_meta):
            meta["id"] = new_id
        removed = len(self.meta) - len(new_meta)
        self.meta = new_meta
        return removed


def merge_capsules(capsules: List[Dict[str, Any]], temperature: float = 1.0) -> str:
    """Merge capsules using a softmax-weighted combination."""
    if not capsules:
        return ""

    try:
        import numpy as np
    except Exception:  # pragma: no cover - optional dependency
        np = None

    if np is None:
        # Fallback to simple ranking if numpy isn't available
        sorted_caps = sorted(capsules, key=lambda c: c.get("score", 0), reverse=True)
        texts = [c["text"] for c in sorted_caps]
        return "\n".join(texts)

    scores = np.array([c.get("score", 0.0) for c in capsules], dtype=float)
    # Softmax weighting for smoother importance distribution
    scores = scores / (temperature if temperature else 1.0)
    scores = scores - scores.max()
    weights = np.exp(scores)
    weights /= weights.sum()

    ordering = np.argsort(-weights)
    texts = [capsules[i]["text"] for i in ordering]
    return "\n".join(texts)


def compress_capsules(capsules: List[Dict[str, Any]], model_name: str = "sshleifer/distilbart-cnn-12-6") -> str:
    """Summarize a list of capsules into a short snippet."""
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
