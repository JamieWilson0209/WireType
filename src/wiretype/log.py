"""One timestamped log line, flushed, so SGE logs read in order."""
from __future__ import annotations

import time


def log(message: str) -> None:
    """Print one line prefixed with the time, and flush it."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)
