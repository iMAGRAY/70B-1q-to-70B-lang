__version__ = "0.1.1"
TrainingStart
from .core import CapsuleStore, merge_capsules, compress_capsules
=======

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
main
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
TrainingStart
    "compress_capsules", 
    "SiglaApp",
=======
    "compress_capsules",
    "get_available_local_models", 
    "create_store_with_best_local_model",
    
    # DSL helpers
main
    "INTENT",
    "RETRIEVE",
    "MERGE",
    "INJECT",
    "EXPAND",
TrainingStart
    "random_walk_links",
=======

    # Graph utilities
    "random_walk_links",

    # Logging
main
    "start_log",
    "log_event",
]
