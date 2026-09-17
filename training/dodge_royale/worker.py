"""One background thread that owns a PPO run, and the gym process under it.

The dashboard is a notebook, and a notebook cell must return promptly or the
page stops responding. Training does not return promptly. So training happens
here, on a thread, and cells read :meth:`TrainingWorker.status` -- a frozen
snapshot taken under a lock, which the reader can hold for as long as it likes
without racing the trainer.

**Only one worker runs at a time, and it is not owned by a cell.** marimo
re-runs a cell whenever anything it depends on changes, so a worker created
inside one would be created again on every rerun, each time launching another
gym and another set of arenas. The single instance lives in this module
instead, reached through :func:`active_worker`, and :func:`start_worker`
refuses to start a second.

Everything that ends a run closes the gym: finishing, stopping, failing, an
interpreter exit. A `dodge-royale gym` left behind holds a CPU core and
competes with the next run for it.
"""

from __future__ import annotations

import atexit
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback

from .metrics import MetricsCollector, Snapshot
from .history import HistoryWriter
from .training import (
    CheckpointWriter,
    SessionConfig,
    attach_history,
    build_model,
    inherit_history,
    session,
)

__all__ = [
    "TrainingWorker",
    "WorkerState",
    "WorkerStatus",
    "active_worker",
    "start_worker",
    "stop_active_worker",
]


class WorkerState(str, Enum):
    """Where a run is. A string enum so a notebook can print it directly."""

    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"

    @property
    def is_active(self) -> bool:
        """Whether a gym is running, or about to be."""
        return self in (WorkerState.STARTING, WorkerState.RUNNING, WorkerState.PAUSED, WorkerState.STOPPING)

    @property
    def is_finished(self) -> bool:
        return self in (WorkerState.STOPPED, WorkerState.FAILED)


@dataclass(frozen=True)
class WorkerStatus:
    """Everything a cell needs, as plain values it can keep.

    Frozen, and built from copies: a cell that renders one of these is not
    reading the trainer's live state while the trainer mutates it.
    """

    state: WorkerState
    metrics: Snapshot
    history: tuple[Snapshot, ...]
    error: str | None = None
    checkpoints: tuple[Path, ...] = ()
    config: SessionConfig | None = None
    resumed_from: Path | None = None

    @property
    def running(self) -> bool:
        return self.state is WorkerState.RUNNING


class _Control(BaseCallback):
    """The worker's hands inside PPO's loop.

    PPO offers no way in from outside once `learn` is running, so pausing,
    stopping and saving all have to happen at a point PPO hands control back.
    That point is this callback.
    """

    def __init__(self, worker: "TrainingWorker") -> None:
        super().__init__()
        self._worker = worker

    def _on_step(self) -> bool:
        worker = self._worker
        if worker._stop.is_set():
            # False ends `learn` cleanly at a step boundary, which leaves the
            # model and its buffers consistent. Killing the thread would not.
            return False

        # Pause parks here rather than spinning, and wakes on stop as well as
        # on resume, so a paused run can still be stopped.
        while worker._paused.is_set() and not worker._stop.is_set():
            worker._set_state(WorkerState.PAUSED)
            worker._publish(force=True)
            worker._resumed.wait(0.1)
        if worker._stop.is_set():
            return False
        if worker._state is WorkerState.PAUSED:
            worker._set_state(WorkerState.RUNNING)

        # Saving runs here, on the training thread, because a save from
        # another thread would read parameters mid-update.
        if worker._save_wanted.is_set():
            worker._save_wanted.clear()
            worker._save_now()

        worker._publish()
        return True


class TrainingWorker:
    """A PPO run on its own thread, with controls and bounded snapshots."""

    def __init__(
        self,
        config: SessionConfig,
        *,
        resume: str | Path | None = None,
        history: int = 512,
        publish_interval: float = 0.5,
    ) -> None:
        self.config = config
        self.resume = Path(resume) if resume else None
        self.publish_interval = publish_interval

        self._lock = threading.RLock()
        self._state = WorkerState.IDLE
        self._error: str | None = None
        # Bounded on purpose: a run of any length must not grow this without
        # limit, and a chart cannot show more points than it has pixels.
        self._history: deque[Snapshot] = deque(maxlen=history)
        self._latest = Snapshot()
        self._checkpoints: list[Path] = []

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._resumed = threading.Event()
        self._save_wanted = threading.Event()
        self._last_publish = 0.0

        self._collector = MetricsCollector()
        self._inherited_episodes = 0
        self._model: Any = None
        self._writer = CheckpointWriter.for_config(config)

    # -- state, all of it behind the lock --

    def _set_state(self, state: WorkerState) -> None:
        with self._lock:
            self._state = state

    @property
    def state(self) -> WorkerState:
        with self._lock:
            return self._state

    def status(self) -> WorkerStatus:
        """A consistent reading of everything, taken under the lock."""
        with self._lock:
            return WorkerStatus(
                state=self._state,
                metrics=self._latest,
                history=tuple(self._history),
                error=self._error,
                checkpoints=tuple(self._checkpoints),
                config=self.config,
                resumed_from=self.resume,
            )

    def _publish(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_publish < self.publish_interval:
            return
        self._last_publish = now
        snapshot = self._collector.snapshot()
        with self._lock:
            self._latest = snapshot
            self._history.append(snapshot)

    # -- controls --

    def start(self) -> None:
        """Begin training. Refuses to start a run that is already going."""
        with self._lock:
            if self._state.is_active:
                raise RuntimeError(f"this worker is already {self._state.value}")
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("this worker's thread is still running")
            self._state = WorkerState.STARTING
            self._error = None
        self._stop.clear()
        self._paused.clear()
        self._resumed.clear()
        self._save_wanted.clear()
        self._thread = threading.Thread(
            target=self._run, name="dodge-royale-training", daemon=True
        )
        self._thread.start()

    def pause(self) -> None:
        self._resumed.clear()
        self._paused.set()

    def resume_training(self) -> None:
        self._paused.clear()
        self._resumed.set()

    def request_save(self) -> None:
        """Ask the training thread to checkpoint at its next step."""
        self._save_wanted.set()

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the run finishes. Returns whether it did.

        For a caller with no clock of its own -- a script, or a test -- that
        would otherwise have to poll `state`.
        """
        thread = self._thread
        if thread is None:
            return True
        thread.join(timeout)
        return not thread.is_alive()

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the run finishes. Returns whether it did.

        For a caller with no clock of its own -- a script, or a test -- that
        would otherwise have to poll `state`.
        """
        thread = self._thread
        if thread is None:
            return True
        thread.join(timeout)
        return not thread.is_alive()

    def stop(self, timeout: float = 30.0) -> None:
        """End the run and wait for the gym to be gone.

        Safe to call from any state, including one that never started and one
        that already failed, because the dashboard calls it on teardown
        without knowing which.
        """
        thread = self._thread
        if thread is None or not thread.is_alive():
            with self._lock:
                if self._state.is_active:
                    self._state = WorkerState.STOPPED
            return
        with self._lock:
            if self._state is not WorkerState.FAILED:
                self._state = WorkerState.STOPPING
        self._stop.set()
        # Wake a paused run so it can notice the stop.
        self._paused.clear()
        self._resumed.set()
        thread.join(timeout)
        if thread.is_alive():  # pragma: no cover - needs a wedged child
            with self._lock:
                self._error = f"the training thread did not stop within {timeout}s"
                self._state = WorkerState.FAILED

    # -- the run itself --

    def _save_now(self) -> Path | None:
        if self._model is None:
            return None
        path = self._writer.save(self._model, step=int(self._model.num_timesteps))
        with self._lock:
            self._checkpoints.append(path)
        return path

    def _run(self) -> None:
        try:
            # `session` closes the gym on every exit from this block, which is
            # what makes a failure here cost a message rather than a stray
            # child process.
            inherited = inherit_history(self._writer, self.resume)
            with session(self.config) as env, HistoryWriter(
                self._writer.history()
            ) as episodes:
                attach_history(self._collector, episodes, self.config)
                self._inherited_episodes = inherited
                self._model = build_model(self.config, env, resume=self.resume)
                self._set_state(WorkerState.RUNNING)
                self._publish(force=True)
                self._model.learn(
                    total_timesteps=self.config.total_timesteps,
                    callback=[self._collector, _Control(self)],
                )
                # A run that reaches its budget, or is stopped, keeps its
                # weights: losing them would make stopping expensive.
                self._save_now()
                self._publish(force=True)
            with self._lock:
                self._state = WorkerState.STOPPED
        except BaseException as error:  # noqa: BLE001 - reported, not swallowed
            with self._lock:
                self._error = f"{type(error).__name__}: {error}"
                self._state = WorkerState.FAILED
            self._publish(force=True)
        finally:
            self._model = None


# --- the single active worker -------------------------------------------

_ACTIVE: TrainingWorker | None = None
_ACTIVE_LOCK = threading.Lock()


def active_worker() -> TrainingWorker | None:
    """The worker this kernel owns, if there is one.

    A notebook cell calls this rather than constructing one, so a rerun finds
    the existing run instead of starting a second alongside it.
    """
    with _ACTIVE_LOCK:
        return _ACTIVE


def start_worker(
    config: SessionConfig, *, resume: str | Path | None = None, **kwargs: Any
) -> TrainingWorker:
    """Start the one worker, replacing a finished one.

    Refuses while a run is active. Two workers would mean two gyms, twice the
    arenas, and two sets of numbers claiming to describe the same run.
    """
    global _ACTIVE
    with _ACTIVE_LOCK:
        if _ACTIVE is not None and _ACTIVE.state.is_active:
            raise RuntimeError(
                f"a run is already {_ACTIVE.state.value}; stop it before starting another"
            )
        worker = TrainingWorker(config, resume=resume, **kwargs)
        _ACTIVE = worker
    worker.start()
    return worker


def stop_active_worker(timeout: float = 30.0) -> None:
    """Stop whatever is running. Idempotent, and safe at interpreter exit."""
    global _ACTIVE
    with _ACTIVE_LOCK:
        worker = _ACTIVE
    if worker is not None:
        worker.stop(timeout)


# The backstop. A notebook killed from the terminal, or a kernel that exits
# without the dashboard's teardown running, still takes its gym with it.
atexit.register(stop_active_worker, 10.0)
