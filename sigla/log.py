import json
import time
from typing import Optional, Dict, Any

_log_path: Optional[str] = None

def start(path: str) -> None:
    """Enable logging to the given file."""
    global _log_path
    _log_path = path


def log(event: Dict[str, Any]) -> None:
    """Append an event to the log if logging is enabled."""
    if _log_path is None:
        return
    event = event.copy()
    event["ts"] = time.time()
    with open(_log_path, "a", encoding="utf-8") as f:
        json.dump(event, f, ensure_ascii=False)
        f.write("\n")


def record(event_type: str, start: float, **data: Any) -> None:
    """Log an event with its duration."""
    data = data.copy()
    data["type"] = event_type
    data["duration"] = time.time() - start
    log(data)
