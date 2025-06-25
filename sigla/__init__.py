"""SIGLA - Semantic Information Graph with Language Agents

A lightweight FAISS-backed capsule database with local model support.
"""

__version__ = "0.2.0"

from .core import (
    CapsuleStore,
    TransformersEmbeddings,
    merge_capsules,
    compress_capsules,
    get_available_local_models,
    create_store_with_best_local_model,
    MissingDependencyError,
)

from .dsl import (
    INTENT,
    RETRIEVE,
    MERGE,
    INJECT,
    EXPAND,
)

from .graph import (
    expand_with_links,
    random_walk_links,
)

__all__ = [
    # Core classes
    "CapsuleStore",
    "TransformersEmbeddings",
    "MissingDependencyError",
    
    # Core functions
    "merge_capsules",
    "compress_capsules",
    "get_available_local_models", 
    "create_store_with_best_local_model",
    
    # DSL functions
    "INTENT",
    "RETRIEVE",
    "MERGE",
    "INJECT",
    "EXPAND",
    
    # Graph functions
    "expand_with_links",
    "random_walk_links",
]
