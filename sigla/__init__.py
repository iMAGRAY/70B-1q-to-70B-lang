__version__ = "0.1.1"

# Public API re-exports
from .core import (
    CapsuleStore,
    TransformersEmbeddings,
    MissingDependencyError,
    merge_capsules,
    compress_capsules,
    get_available_local_models,
    create_store_with_best_local_model,
)

from .dsl import INTENT, RETRIEVE, MERGE, INJECT, EXPAND
from .graph import random_walk_links
from .log import start as start_log, log as log_event

try:
    from .server import app as SiglaApp
except Exception:  # pragma: no cover - optional dependency
    SiglaApp = None

__all__ = [
    # Core classes
    "CapsuleStore",
    "TransformersEmbeddings",
    "MissingDependencyError",
    
    # Core helpers
    "merge_capsules",
    "compress_capsules",
    "get_available_local_models", 
    "create_store_with_best_local_model",
    
    # DSL helpers
    "INTENT",
    "RETRIEVE",
    "MERGE",
    "INJECT",
    "EXPAND",

    # Graph utilities
    "random_walk_links",

    # Logging
    "start_log",
    "log_event",
    
    # Server
    "SiglaApp",
]
