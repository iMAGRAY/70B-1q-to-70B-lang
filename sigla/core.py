import json
from typing import List, Dict, Any

import faiss
from sentence_transformers import SentenceTransformer


class CapsuleStore:
    """A lightweight FAISS-backed capsule database."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dimension)
        self.meta: List[Dict[str, Any]] = []

    def add_capsules(self, capsules: List[Dict[str, Any]]):
        """Embed and add capsules to the index."""
        texts = [c["text"] for c in capsules]
        vectors = self.model.encode(texts, convert_to_numpy=True)
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self.meta.extend(capsules)

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

    def query(self, text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        vector = self.model.encode([text], convert_to_numpy=True)
        faiss.normalize_L2(vector)
        scores, indices = self.index.search(vector, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            cap = self.meta[idx].copy()
            cap["score"] = float(score)
            results.append(cap)
        return results


def merge_capsules(capsules: List[Dict[str, Any]]) -> str:
    """Simple merge by concatenation with weights."""
    sorted_caps = sorted(capsules, key=lambda c: c.get("score", 0), reverse=True)
    texts = [c["text"] for c in sorted_caps]
    return " \n".join(texts)
