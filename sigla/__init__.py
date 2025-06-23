from .core import (
    CapsuleStore,
    merge_capsules,
    compress_capsules,
    text_similarity,
    sanitize_text,
    capsulate_text,
    clear_summarizer_cache,
)
from .dsl import INTENT, RETRIEVE, MERGE, INJECT, EXPAND
from .graph import random_walk_links, to_dot
from .log import start as start_log, log as log_event

try:
    from .server import app as SiglaApp
except Exception:  # pragma: no cover - optional dependency
    SiglaApp = None

__all__ = [
    "CapsuleStore",
    "merge_capsules",
    "compress_capsules",
    "text_similarity",
    "sanitize_text",
    "capsulate_text",
    "SiglaApp",
    "INTENT",
    "RETRIEVE",
    "MERGE",
    "INJECT",
    "EXPAND",
    "random_walk_links",
    "to_dot",
    "start_log",
    "log_event",
    "clear_summarizer_cache",
]
