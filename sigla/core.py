import json
from typing import List, Dict, Any
import re

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


# Basic regex to redact obvious personal data such as emails or long digit
# sequences (phone numbers, IDs). This is a minimal security safeguard to avoid
# storing sensitive information.
_PERSONAL_RE = re.compile(r"([\w.+-]+@[\w-]+\.[\w.-]+)|([0-9]{4,})")


def sanitize_text(text: str) -> str:
    """Remove simple personal identifiers from text."""
    return _PERSONAL_RE.sub("[REDACTED]", text)


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


class _HashModel:
    """Fallback embedder using SHA-256 hashing."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def encode(self, texts: List[str], convert_to_numpy: bool = True):
        import hashlib
        import numpy as np

        vectors = []
        for text in texts:
            h = hashlib.sha256(text.encode("utf-8")).digest()
            arr = np.frombuffer(h, dtype=np.uint8).astype("float32")
            reps = (self.dimension + len(arr) - 1) // len(arr)
            vec = np.tile(arr, reps)[: self.dimension]
            vectors.append(vec)
        return np.stack(vectors)

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension


def capsulate_text(text: str, tags: List[str] | None = None, source: str | None = None) -> List[Dict[str, Any]]:
    """Split raw text into sanitized capsules."""
    sentences = [s.strip() for s in re.split(_SENTENCE_RE, text) if s.strip()]
    capsules = []
    for sent in sentences:
        capsule: Dict[str, Any] = {"text": sanitize_text(sent)}
        if tags:
            capsule["tags"] = list(tags)
        if source:
            capsule.setdefault("metadata", {})["source"] = source
        capsules.append(capsule)
    return capsules


class CapsuleStore:
    """A lightweight FAISS-backed capsule database."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_factory: str = "Flat",
        lazy: bool = False,
    ):
        if faiss is None:
            raise MissingDependencyError("faiss package is required")

        self.model_name = model_name
        self.index_factory = index_factory
        self.model = None
        self.dimension = 0
        self.index = None
        self.meta: List[Dict[str, Any]] = []
        # Simple in-memory cache for embeddings to avoid repeated model calls
        self._cache: Dict[str, Any] = {}

        if not lazy:
            self._ensure_model()
            self.index = faiss.index_factory(self.dimension, index_factory, faiss.METRIC_INNER_PRODUCT)

    def _ensure_model(self) -> None:
        """Load the embedding model if it hasn't been loaded yet."""
        if self.model is None:
            if SentenceTransformer is None or self.model_name in {"hash", "none"}:
                # Fallback to a simple hash-based embedder if transformers are missing
                # or the user explicitly requests it via ``model_name='hash'``.
                self.model = _HashModel()
                self.dimension = self.model.get_sentence_embedding_dimension()
            else:
                self.model = SentenceTransformer(self.model_name)
                self.dimension = self.model.get_sentence_embedding_dimension()
            if self.index is not None and self.index.d != self.dimension:
                raise RuntimeError("Index dimension does not match embedding model")

    def embed_texts(self, texts: List[str]):
        """Return normalized embeddings for a list of texts."""
        self._ensure_model()
        try:
            import numpy as np
        except Exception:  # pragma: no cover - optional dependency
            np = None

        if np is None:
            vectors = self.model.encode(texts, convert_to_numpy=True)
            faiss.normalize_L2(vectors)
            return vectors

        missing = []
        for t in texts:
            if t not in self._cache:
                missing.append(t)
        if missing:
            new_vectors = self.model.encode(missing, convert_to_numpy=True)
            for t, v in zip(missing, new_vectors):
                self._cache[t] = v

        vectors = np.stack([self._cache[t] for t in texts])
        faiss.normalize_L2(vectors)
        return vectors

    def embed_query(self, text: str):
        """Return a single normalized embedding vector."""
        return self.embed_texts([text])

    def add_capsules(
        self,
        capsules: List[Dict[str, Any]],
        link_neighbors: int = 0,
        dedup: bool = True,
    ) -> int:
        """Embed and add capsules to the index, assigning IDs.

        Parameters
        ----------
        capsules:
            A list of capsule dictionaries with at least a ``text`` field.
        link_neighbors:
            If greater than zero, automatically link each new capsule to the
            nearest existing capsules in the index.
        """
        self._ensure_model()
        start = len(self.meta)
        existing = {m["text"] for m in self.meta} if dedup else set()
        cleaned_caps: List[Dict[str, Any]] = []
        texts: List[str] = []
        for c in capsules:
            clean_text = sanitize_text(c["text"])
            if dedup and clean_text in existing:
                continue
            cleaned = c.copy()
            cleaned["text"] = clean_text
            cleaned_caps.append(cleaned)
            texts.append(clean_text)
            existing.add(clean_text)
        if not cleaned_caps:
            return 0
        vectors = self.embed_texts(texts)

        if not self.index.is_trained:
            self.index.train(vectors)
        self.index.add(vectors)

        for i, cap in enumerate(cleaned_caps):
            meta = cap.copy()
            meta.setdefault("links", [])
            meta["id"] = start + i
            self.meta.append(meta)

        if link_neighbors > 0 and self.index.ntotal > 0:
            # Search for neighbors including the newly added vectors.
            _, idxs = self.index.search(vectors, link_neighbors + 1)
            new_links: Dict[int, List[int]] = {}
            for i, neighbors in enumerate(idxs):
                links = []
                for n in neighbors:
                    if n == -1 or n == start + i:
                        continue
                    links.append(int(n))
                if links:
                    meta_links = self.meta[start + i].setdefault("links", [])
                    for n in links:
                        if n not in meta_links:
                            meta_links.append(n)
                new_links[start + i] = links

            # Symmetrically link neighbors back to the new capsules
            for new_id, links in new_links.items():
                for n in links:
                    if n < 0 or n >= len(self.meta):
                        continue
                    rev = self.meta[n].setdefault("links", [])
                    if new_id not in rev:
                        rev.append(new_id)

        return len(cleaned_caps)

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
        self.dimension = self.index.d
        self.model = None
        self._cache = {}

    def query(
        self,
        text: str,
        top_k: int = 5,
        tags: List[str] | None = None,
        sources: List[str] | None = None,
        min_rating: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Return top matching capsules filtered by tags, source and rating."""
        self._ensure_model()
        vector = self.embed_query(text)
        # oversample to account for tag filtering
        scores, indices = self.index.search(vector, top_k * 5)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta = self.meta[idx]
            if tags and not set(tags).intersection(meta.get("tags", [])):
                continue
            if sources:
                src = meta.get("metadata", {}).get("source")
                if src not in sources:
                    continue
            cap = meta.copy()
            rating = (
                cap.get("rating")
                or cap.get("metadata", {}).get("rating")
                or 1.0
            )
            if rating < min_rating:
                continue
            cap["score"] = float(score) * float(rating)
            cap["id"] = int(idx)
            results.append(cap)
            if len(results) >= top_k:
                break
        return results

    def remove_capsules(self, ids: List[int]) -> int:
        """Remove capsules by id, rebuilding the index."""
        if not ids:
            return 0
        self._ensure_model()
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
        vectors = self.embed_texts(texts) if texts else None
        self.index = faiss.IndexFlatIP(self.dimension)
        if vectors is not None and len(texts) > 0:
            self.index.add(vectors)
        for new_id, meta in enumerate(new_meta):
            meta["id"] = new_id
        removed = len(self.meta) - len(new_meta)
        self.meta = new_meta
        self._cache = {}
        return removed

    def rebuild_index(self, model_name: str | None = None, index_factory: str | None = None) -> None:
        """Recompute all embeddings and rebuild the FAISS index."""
        if model_name:
            if SentenceTransformer is None:
                raise MissingDependencyError("sentence-transformers package is required")
            self.model = SentenceTransformer(model_name)
            self.model_name = model_name
            self.dimension = self.model.get_sentence_embedding_dimension()
        if index_factory:
            self.index_factory = index_factory
        self._ensure_model()
        texts = [m["text"] for m in self.meta]
        vectors = self.embed_texts(texts) if texts else None
        self.index = faiss.index_factory(self.dimension, self.index_factory, faiss.METRIC_INNER_PRODUCT)
        if vectors is not None and len(texts) > 0:
            self.index.add(vectors)
        for idx, meta in enumerate(self.meta):
            meta["id"] = idx
        self._cache = {}

    def update_metadata(
        self,
        ids: List[int],
        rating: float | None = None,
        add_tags: List[str] | None = None,
        remove_tags: List[str] | None = None,
    ) -> int:
        """Update capsule metadata in place.

        Parameters
        ----------
        ids:
            Capsule IDs to modify.
        rating:
            New rating value to assign. ``None`` leaves it unchanged.
        add_tags:
            Tags to add to each capsule's ``tags`` list.
        remove_tags:
            Tags to remove from each capsule's ``tags`` list.
        Returns
        -------
        int
            Number of capsules modified.
        """
        updated = 0
        for cid in ids:
            if cid < 0 or cid >= len(self.meta):
                continue
            meta = self.meta[cid]
            changed = False
            if rating is not None:
                meta["rating"] = rating
                meta.setdefault("metadata", {})["rating"] = rating
                changed = True
            if add_tags:
                current = set(meta.get("tags", []))
                new = current.union(add_tags)
                if new != current:
                    meta["tags"] = sorted(new)
                    changed = True
            if remove_tags:
                current = set(meta.get("tags", []))
                new = current.difference(remove_tags)
                if new != current:
                    meta["tags"] = sorted(new)
                    changed = True
            if changed:
                updated += 1
        return updated


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
