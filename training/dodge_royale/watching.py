"""One browser-game policy service per dashboard kernel.

The iframe runs the Bevy game. This module owns only its local policy service;
no marimo cell owns a game loop, and no Python code renders game frames.
"""

from __future__ import annotations

import atexit
import threading

from .browser_watch import GameWatch
from .snapshots import snapshot_dir
from .training import SessionConfig

__all__ = ["active_watch", "start_watch", "stop_watch"]

_ACTIVE: GameWatch | None = None
_LOCK = threading.Lock()


def active_watch() -> GameWatch | None:
    with _LOCK:
        return _ACTIVE


def start_watch(config: SessionConfig, *, web_url: str = "http://127.0.0.1:4000/") -> GameWatch:
    """Start serving the newest policy to a Bevy web game iframe."""
    global _ACTIVE
    stop_watch()
    watch = GameWatch(snapshot_dir(config.checkpoint_dir, config.run_name), web_url=web_url)
    with _LOCK:
        _ACTIVE = watch
    return watch


def stop_watch() -> None:
    """Stop inference and release the loopback server."""
    global _ACTIVE
    with _LOCK:
        watch, _ACTIVE = _ACTIVE, None
    if watch is not None:
        watch.close()


atexit.register(stop_watch)
