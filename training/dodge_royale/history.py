"""Every episode a model has ever finished, kept beside the model.

A run reports rolling averages while it goes, but those are a window: the last
hundred episodes, and nothing before them. To ask whether a policy is better
than it was an hour ago, or how it behaved when the arena held forty enemies
rather than a hundred, the individual episodes have to survive. So each one is
appended here as it finishes.

**The enemy count travels with every record.** Survival time means nothing on
its own -- three seconds against 100 enemies and three seconds against 12 are
not the same result -- and the count can change between runs, or between a run
and the one it resumed from. Storing it per episode rather than per file is
what lets a later reader group by difficulty instead of guessing.

**The history belongs to the model, not to the process.** It is written beside
the checkpoint, under the same name, so a checkpoint and its past move
together. Resuming appends rather than starting again, which is what makes
"full training history" true across restarts.

The format is JSON Lines: one object per episode, appended and flushed as it
happens. A run killed halfway leaves every episode it finished, which a single
JSON document would not.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable, Iterator

from .telemetry import FRAMES_PER_SECOND

__all__ = ["EpisodeRecord", "HistoryWriter", "history_path", "read_history"]

#: Schema version for a record. Bumped if a field changes meaning, so a reader
#: can tell an old file from a new one rather than misreading it.
RECORD_VERSION = 1


@dataclass(frozen=True)
class EpisodeRecord:
    """One finished episode, and the conditions it was played under."""

    #: Environment steps the run had taken when this episode ended, so the
    #: history can be plotted against training progress rather than only
    #: against episode number.
    timesteps: int
    #: Which env of the batch finished.
    env: int
    #: Frames the episode lasted. The authority: seconds is derived from it.
    frames: int
    #: Enemies the arena was configured to hold. The reason two equal survival
    #: times can mean opposite things.
    enemies: int
    #: Total reward over the episode.
    reward: float
    #: True if the player was hit, False if the frame budget ran out. A
    #: timeout is a censored survival time, not a longer one.
    died: bool
    #: Enemy-on-enemy kills during the episode.
    enemies_destroyed: int
    #: The seed the episode was played on, so it can be replayed exactly.
    episode_seed: int
    #: Frames each predicted path held its action, which changes what the
    #: policy was choosing between.
    hold_frames: int
    #: Wall-clock seconds since the epoch, for ordering across restarts.
    recorded_at: float
    version: int = RECORD_VERSION

    @property
    def seconds(self) -> float:
        """Survival time. One action per frame, so this is frames over 60."""
        return self.frames / FRAMES_PER_SECOND

    @classmethod
    def from_summary(
        cls,
        summary: dict,
        *,
        timesteps: int,
        env: int,
        enemies: int,
        hold_frames: int,
        episode_seed: int,
        recorded_at: float,
    ) -> "EpisodeRecord":
        return cls(
            timesteps=int(timesteps),
            env=int(env),
            frames=int(summary.get("frames", 0)),
            enemies=int(enemies),
            reward=float(summary.get("return", 0.0)),
            died=bool(summary.get("died", False)),
            enemies_destroyed=int(summary.get("enemies_destroyed", 0)),
            episode_seed=int(episode_seed),
            hold_frames=int(hold_frames),
            recorded_at=float(recorded_at),
        )


def run_history_path(directory: str | Path, run_name: str) -> Path:
    """The live history a run appends to, one per run name.

    Separate from a checkpoint's copy: the run appends here continuously, and
    each checkpoint takes a snapshot of it, so a checkpoint carries everything
    that had happened by the time it was written.
    """
    return Path(directory) / f"{run_name}.history.jsonl"


def history_path(checkpoint: str | Path) -> Path:
    """Where a checkpoint's history lives: beside it, under the same name."""
    path = Path(checkpoint)
    return path.with_suffix(".history.jsonl")


class HistoryWriter:
    """Appends episodes to a run's history file.

    Opened once and kept open, flushed after every episode. An episode that
    has been reported is an episode that survives the process being killed,
    which is the point of keeping a history at all.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")
        self._written = 0

    @property
    def written(self) -> int:
        """Episodes this writer has appended."""
        return self._written

    def append(self, record: EpisodeRecord) -> None:
        json.dump(asdict(record), self._handle)
        self._handle.write("\n")
        # Flushed, not buffered: a run that is stopped or crashes should keep
        # every episode it actually finished.
        self._handle.flush()
        self._written += 1

    def extend(self, records: Iterable[EpisodeRecord]) -> None:
        for record in records:
            self.append(record)

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()

    def __enter__(self) -> "HistoryWriter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def read_history(path: str | Path) -> list[EpisodeRecord]:
    """Read a history file back.

    A truncated final line is skipped rather than raising: a run killed
    mid-write should not make its whole history unreadable.
    """
    source = Path(path)
    if not source.exists():
        return []
    known = {field.name for field in fields(EpisodeRecord)}
    records: list[EpisodeRecord] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(raw, dict):
            continue
        records.append(EpisodeRecord(**{k: v for k, v in raw.items() if k in known}))
    return records


def iter_history(path: str | Path) -> Iterator[EpisodeRecord]:
    """The same, without holding the whole file in memory."""
    source = Path(path)
    if not source.exists():
        return
    known = {field.name for field in fields(EpisodeRecord)}
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(raw, dict):
                yield EpisodeRecord(**{k: v for k, v in raw.items() if k in known})
