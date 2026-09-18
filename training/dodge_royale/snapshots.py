"""A snapshot of the policy after every update, for something else to watch.

Adapted from DodgeAI ``dodge/training.py`` at revision e32f222
(``snapshot_name``, ``newest_snapshot``, ``_publish_snapshot``). DodgeAI is a
separate project and is neither imported nor modified.

Checkpoints are saved on request and at the end of a run. Snapshots are
different: one per completed update, written automatically, so a viewer can
always pick up the most recent policy without the run being asked for it.

Three things make that safe to read while a run is writing:

**Atomic publication.** A snapshot is written to a hidden name and renamed
into place. A reader scanning the directory sees either the previous file or
the new one, never a half-written zip. Readers skip dotfiles for the same
reason.

**Numeric order from lexicographic order.** The update number is zero-padded,
so sorting names sorts updates and "newest" needs no metadata.

**Pruning is best effort.** On Windows a viewer holding an old snapshot open
locks it, and losing that race must not interrupt training.
"""

from __future__ import annotations

import secrets
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback

__all__ = [
    "SNAPSHOT_RETENTION",
    "SnapshotPublisher",
    "newest_snapshot",
    "publish_snapshot",
    "snapshot_dir",
    "snapshot_name",
]

#: Snapshots kept on disk. Enough that a viewer reloading between episodes
#: always finds one, few enough that a long run does not fill a disk with
#: policies nobody will look at.
SNAPSHOT_RETENTION = 5


def snapshot_dir(checkpoint_dir: str | Path, run_name: str) -> Path:
    """Where a run publishes its per-update snapshots."""
    return Path(checkpoint_dir) / f"{run_name}-live"


def snapshot_name(update: int) -> str:
    """Zero-padded, so lexicographic order is numeric order."""
    return f"update-{int(update):08d}.zip"


def newest_snapshot(directory: str | Path, after: int = -1) -> tuple[Path, int] | None:
    """The highest completed update in `directory` above `after`.

    Dotfiles are skipped: a snapshot is written to a hidden temporary name and
    only becomes visible on its rename, so anything starting with a dot is a
    partial write.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return None
    best: tuple[Path, int] | None = None
    for path in directory.glob("update-*.zip"):
        if path.name.startswith("."):
            continue
        try:
            update = int(path.stem.removeprefix("update-"))
        except ValueError:
            continue
        if update > after and (best is None or update > best[1]):
            best = (path, update)
    return best


def prune_snapshots(directory: str | Path, keep: int = SNAPSHOT_RETENTION) -> None:
    """Keep the newest few. Best effort, because a reader may hold one open."""
    directory = Path(directory)
    if not directory.is_dir():
        return
    snapshots = sorted(
        path for path in directory.glob("update-*.zip") if not path.name.startswith(".")
    )
    for path in snapshots[: max(len(snapshots) - keep, 0)]:
        try:
            path.unlink()
        except OSError:
            # A viewer has it open. It will be pruned on a later update.
            pass


def publish_snapshot(model, directory: str | Path, update: int) -> Path | None:
    """Write one immutable snapshot, then prune. Never raises into training.

    Written to a hidden name and renamed, so a viewer scanning the directory
    cannot open a half-written zip. A failure here is reported by returning
    None rather than by ending a run: a snapshot nobody can read is worth less
    than the training that would have been lost.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # SB3 appends ".zip" only when the path it is given has an empty suffix,
    # so the temporary stem must carry no dot beyond the leading one that
    # hides it. A name like ".pending-1.zip" would already look suffixed, SB3
    # would write it unchanged, and the rename below would find nothing.
    stem = directory / f".pending-{int(update):08d}-{secrets.token_hex(4)}"
    try:
        model.save(stem)
        Path(f"{stem}.zip").replace(directory / snapshot_name(update))
    except Exception:  # noqa: BLE001 - a failed snapshot must not stop training
        for leftover in directory.glob(".pending-*"):
            try:
                leftover.unlink()
            except OSError:
                pass
        return None
    prune_snapshots(directory)
    return directory / snapshot_name(update)


class SnapshotPublisher(BaseCallback):
    """Publish the policy after every completed PPO update.

    Published on the *next* rollout's start rather than at the end of the one
    before it: SB3 calls `_on_rollout_end` before `train`, so a snapshot taken
    there would be the policy that has not yet learned from the rollout it
    just collected. One final snapshot is written when training stops, so the
    last update is never the one nobody can watch.
    """

    def __init__(self, directory: str | Path, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.directory = Path(directory)
        self.updates = 0
        self.published: list[Path] = []
        self._pending = False

    def _publish(self) -> None:
        if not self._pending:
            return
        self._pending = False
        written = publish_snapshot(self.model, self.directory, self.updates)
        if written is not None:
            self.published.append(written)

    def _on_rollout_start(self) -> None:
        # Anything pending here is an update that has now finished training.
        self._publish()

    def _on_rollout_end(self) -> None:
        self.updates += 1
        self._pending = True

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self) -> None:
        self._publish()

    @property
    def latest(self) -> Path | None:
        found = newest_snapshot(self.directory)
        return found[0] if found else None
