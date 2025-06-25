__version__ = "0.1.1"

"""SIGLA public package API.

Этот модуль переэкспортирует основные классы/функции, чтобы пользователь мог
```
from sigla import CapsuleStore, GitHubLLM
```
без глубоких импортов.
"""

# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

from .core import (
    CapsuleStore,
    TransformersEmbeddings,
    MissingDependencyError,
    merge_capsules,
    compress_capsules,
    get_available_local_models,
    create_store_with_best_local_model,
)

# ---------------------------------------------------------------------------
# DSL & Graph helpers
# ---------------------------------------------------------------------------

from .dsl import INTENT, RETRIEVE, MERGE, INJECT, EXPAND
from .graph import random_walk_links

# ---------------------------------------------------------------------------
# LLM wrappers
# ---------------------------------------------------------------------------

from .llm import GitHubLLM  # noqa: F401  (imported for re-export)

# ---------------------------------------------------------------------------
# Logging & optional server
# ---------------------------------------------------------------------------

from .log import start as start_log, log as log_event

try:
    from .server import app as SiglaApp  # pragma: no cover – optional
except Exception:  # pragma: no cover – optional dependency
    SiglaApp = None

# ---------------------------------------------------------------------------
# Public symbols
# ---------------------------------------------------------------------------

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

    # DSL
    "INTENT",
    "RETRIEVE",
    "MERGE",
    "INJECT",
    "EXPAND",

    # Graph
    "random_walk_links",

    # LLM
    "GitHubLLM",

    # Logging
    "start_log",
    "log_event",

    # Optional FastAPI app
    "SiglaApp",
]
