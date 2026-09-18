"""The one watch session a kernel owns.

Same reasoning as `worker.py`: marimo re-runs a cell whenever its inputs
change, so a cell that created a watch session would create another on every
tick, each with its own gym. The session lives here instead, and the notebook
reaches it through these functions.
"""

from __future__ import annotations

import atexit
import threading
from typing import Any

from .snapshots import snapshot_dir
from .training import SessionConfig
from .viewer import WatchSession

__all__ = ["active_watch", "start_watch", "stop_watch"]

_ACTIVE: WatchSession | None = None
_LOCK = threading.Lock()


def active_watch() -> WatchSession | None:
    """The session this kernel owns, if there is one."""
    with _LOCK:
        return _ACTIVE


def start_watch(config: SessionConfig, **overrides: Any) -> WatchSession:
    """Begin watching, replacing any session already running."""
    global _ACTIVE
    stop_watch()
    session = WatchSession(
        snapshot_dir(config.checkpoint_dir, config.run_name),
        max_frames=config.max_frames,
        binary=config.binary,
        **overrides,
    )
    with _LOCK:
        _ACTIVE = session
    return session


def stop_watch() -> None:
    """Close the session and its gym. Idempotent, and safe at exit."""
    global _ACTIVE
    with _LOCK:
        session, _ACTIVE = _ACTIVE, None
    if session is not None:
        session.close()


atexit.register(stop_watch)
