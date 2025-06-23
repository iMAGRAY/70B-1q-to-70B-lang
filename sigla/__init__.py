from .core import CapsuleStore, merge_capsules
from .dsl import INTENT, RETRIEVE, MERGE, INJECT, EXPAND
from .log import start as start_log, log as log_event

try:
    from .server import app as SiglaApp
except Exception:  # pragma: no cover - optional dependency
    SiglaApp = None

__all__ = [
    "CapsuleStore",
    "merge_capsules",
    "SiglaApp",
    "INTENT",
    "RETRIEVE",
    "MERGE",
    "INJECT",
    "EXPAND",
    "start_log",
    "log_event",
]
