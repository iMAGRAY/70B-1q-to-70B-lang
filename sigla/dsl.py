"""SIGLA DSL - Domain Specific Language for semantic operations."""

from typing import List, Dict, Any, Optional
import re

try:
    import faiss  # type: ignore
except ImportError:
    faiss = None

from .core import CapsuleStore, merge_capsules, MissingDependencyError
from .graph import expand_with_links


def INTENT(store: CapsuleStore, text: str) -> str:
    """Extract semantic intent from text using keyword analysis and query expansion."""
    # Clean and normalize text
    text = text.lower().strip()
    
    # Remove common stop words and extract meaningful terms
    stop_words = {'и', 'в', 'на', 'с', 'по', 'для', 'к', 'от', 'из', 'о', 'об', 'что', 'как', 'где', 'когда'}
    words = [w for w in re.findall(r'\w+', text) if w not in stop_words and len(w) > 2]
    
    # If we have meaningful words, use them; otherwise fall back to first few words
    if words:
        # Take up to 5 most meaningful words
        intent_words = words[:5]
    else:
        # Fallback to first few words
        intent_words = text.split()[:3]
    
    return " ".join(intent_words)


def RETRIEVE(store: CapsuleStore, intent_vector: str, top_k: int = 5, tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Retrieve semantically relevant capsules based on intent."""
    if not intent_vector.strip():
        return []
    
    # Use the store's query method which handles semantic similarity
    results = store.query(intent_vector, top_k=top_k, tags=tags)
    
    # Enhance results with relevance scoring
    for result in results:
        # Add semantic relevance indicators
        result['relevance'] = 'high' if result.get('score', 0) > 0.8 else 'medium' if result.get('score', 0) > 0.5 else 'low'
        result['intent_match'] = intent_vector
    
    return results


def MERGE(capsules: List[Dict[str, Any]], temperature: float = 1.0) -> str:
    """Intelligently merge capsule texts with context preservation."""
    if not capsules:
        return ""
    
    # Use the core merge function but enhance with structure
    merged_text = merge_capsules(capsules, temperature=temperature)
    
    # Add semantic structure markers
    if len(capsules) > 1:
        merged_text = f"[MERGED FROM {len(capsules)} SOURCES]\n{merged_text}"
    
    return merged_text


def INJECT(text: str, store: Optional[CapsuleStore] = None, tags: Optional[List[str]] = None) -> str:
    """Inject new knowledge into the store and return enhanced text."""
    if store is not None:
        # Create a properly structured capsule
        capsule = {
            "text": text,
            "tags": tags or ["injected", "dsl"],
            "source": "DSL_INJECT",
            "timestamp": __import__('time').time()
        }
        
        # Add to store
        capsule_id = store.add_capsule(text, tags=tags or ["injected"])
        
        # Return enhanced text with metadata
        return f"[INJECTED AS CAPSULE {capsule_id}] {text}"
    
    # If no store is provided, simply return the text unchanged – the caller is
    # responsible for managing persistence in this scenario.
    return text


def EXPAND(
    capsule: Dict[str, Any],
    store: CapsuleStore,
    depth: int = 1,
    limit: int = 10,
    algo: str = "bfs",
    weight_threshold: float = 0.0,
) -> List[Dict[str, Any]]:
    """Expand capsule knowledge through graph traversal with multiple algorithms."""
    if not capsule or not store:
        return [capsule] if capsule else []
    
    # Use graph expansion
    if algo == "random":
        from .graph import random_walk_links
        return random_walk_links([capsule], store, steps=depth, limit=limit)
    else:
        return expand_with_links([capsule], store, depth=depth, limit=limit, weight_threshold=weight_threshold)


def ANALYZE(capsules: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze a collection of capsules for patterns and insights."""
    if not capsules:
        return {"error": "No capsules to analyze"}
    
    # Basic statistics
    total_length = sum(len(c.get("text", "")) for c in capsules)
    avg_score = sum(c.get("score", 0) for c in capsules) / len(capsules)
    
    # Tag analysis
    all_tags = []
    for c in capsules:
        all_tags.extend(c.get("tags", []))
    
    tag_counts = {}
    for tag in all_tags:
        tag_counts[tag] = tag_counts.get(tag, 0) + 1
    
    # Content analysis
    all_text = " ".join(c.get("text", "") for c in capsules)
    word_count = len(all_text.split())
    
    return {
        "capsule_count": len(capsules),
        "total_text_length": total_length,
        "average_score": avg_score,
        "word_count": word_count,
        "top_tags": sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:5],
        "analysis_summary": f"Analyzed {len(capsules)} capsules with {word_count} words and average relevance {avg_score:.3f}"
    }
