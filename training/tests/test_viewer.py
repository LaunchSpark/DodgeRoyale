"""Per-update snapshots, and the viewer that watches the newest one."""

from __future__ import annotations

import numpy as np
import pytest

from dodge_royale.protocol import Layout
from dodge_royale.snapshots import (
    SNAPSHOT_RETENTION,
    SnapshotPublisher,
    newest_snapshot,
    prune_snapshots,
    publish_snapshot,
    snapshot_dir,
    snapshot_name,
)
from dodge_royale.viewer import render_observation


@pytest.fixture(scope="session")
def layout(manifest) -> Layout:
    return Layout.from_json(manifest["layout"])


class FakeModel:
    """Something with a `save`, which is all publishing needs."""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    def save(self, path):
        if self.fail:
            raise RuntimeError("no")
        from pathlib import Path

        target = Path(f"{path}.zip")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"policy")


# --- snapshot naming and discovery ----------------------------------------


def test_names_sort_numerically_because_they_are_padded():
    names = [snapshot_name(n) for n in (2, 10, 1, 100)]
    assert sorted(names) == [snapshot_name(n) for n in (1, 2, 10, 100)]


def test_the_newest_is_the_highest_update(tmp_path):
    for update in (1, 7, 3):
        publish_snapshot(FakeModel(), tmp_path, update)
    found = newest_snapshot(tmp_path)
    assert found is not None and found[1] == 7


def test_newest_can_be_asked_for_something_newer_than_what_is_loaded(tmp_path):
    for update in (1, 2, 3):
        publish_snapshot(FakeModel(), tmp_path, update)
    assert newest_snapshot(tmp_path, after=3) is None, "nothing newer than the newest"
    assert newest_snapshot(tmp_path, after=1)[1] == 3


def test_a_partial_write_is_never_picked_up(tmp_path):
    """A snapshot is hidden until its rename, so a reader cannot open half."""
    publish_snapshot(FakeModel(), tmp_path, 1)
    (tmp_path / ".pending-00000009-abcd.zip").write_bytes(b"half")
    found = newest_snapshot(tmp_path)
    assert found is not None and found[1] == 1, "the dotfile must be ignored"


def test_a_missing_directory_has_no_snapshots(tmp_path):
    assert newest_snapshot(tmp_path / "absent") is None


def test_a_failed_save_leaves_no_rubbish_and_does_not_raise(tmp_path):
    assert publish_snapshot(FakeModel(fail=True), tmp_path, 1) is None
    assert list(tmp_path.glob("*")) == [], "no half-written file survives"


def test_only_the_newest_few_are_kept(tmp_path):
    for update in range(1, SNAPSHOT_RETENTION + 4):
        publish_snapshot(FakeModel(), tmp_path, update)
    kept = sorted(path.name for path in tmp_path.glob("update-*.zip"))
    assert len(kept) == SNAPSHOT_RETENTION
    assert kept[-1] == snapshot_name(SNAPSHOT_RETENTION + 3), "the newest survives"


def test_pruning_survives_a_file_it_cannot_delete(tmp_path, monkeypatch):
    """A viewer holding a snapshot open must not interrupt training."""
    for update in range(1, SNAPSHOT_RETENTION + 3):
        publish_snapshot(FakeModel(), tmp_path, update)

    from pathlib import Path

    original = Path.unlink

    def stubborn(self, *args, **kwargs):
        raise OSError("locked by a reader")

    monkeypatch.setattr(Path, "unlink", stubborn)
    prune_snapshots(tmp_path)  # must not raise
    monkeypatch.setattr(Path, "unlink", original)


def test_the_snapshot_directory_is_named_for_the_run(tmp_path):
    assert snapshot_dir(tmp_path, "royale").name == "royale-live"


# --- the publisher ---------------------------------------------------------


def test_a_snapshot_is_published_after_the_update_not_before(tmp_path):
    """SB3 calls `_on_rollout_end` before `train`, so publishing there would
    save a policy that has not learned from the rollout it just collected."""
    publisher = SnapshotPublisher(tmp_path)
    publisher.model = FakeModel()

    publisher._on_rollout_end()  # rollout collected, update not yet run
    assert newest_snapshot(tmp_path) is None, "nothing published before training"

    publisher._on_rollout_start()  # the update has now finished
    found = newest_snapshot(tmp_path)
    assert found is not None and found[1] == 1


def test_the_last_update_is_published_when_training_ends(tmp_path):
    publisher = SnapshotPublisher(tmp_path)
    publisher.model = FakeModel()
    publisher._on_rollout_end()
    publisher._on_training_end()
    assert newest_snapshot(tmp_path)[1] == 1, "the final update is watchable too"


def test_every_update_gets_its_own_snapshot(tmp_path):
    publisher = SnapshotPublisher(tmp_path)
    publisher.model = FakeModel()
    for _ in range(3):
        publisher._on_rollout_end()
        publisher._on_rollout_start()
    assert publisher.updates == 3
    assert newest_snapshot(tmp_path)[1] == 3


# --- rendering -------------------------------------------------------------


def blank(layout: Layout) -> np.ndarray:
    return np.zeros(layout.observation_values, dtype=np.float32)


def put(observation, layout, channel, row, column, value=1.0):
    cells = layout.grid * layout.grid
    index = layout.channels.index(channel)
    start = layout.grid_section.offset + index * cells
    observation[start + row * layout.grid + column] = value


def test_the_image_is_the_grid_at_the_requested_zoom(layout):
    image = render_observation(blank(layout), layout, scale=8, show_paths=False)
    assert image.shape == (layout.grid * 8, layout.grid * 8, 3)
    assert image.dtype == np.uint8


def test_each_hazard_draws_in_its_own_colour(layout):
    from dodge_royale.viewer import CHANNEL_COLOURS

    for channel, colour in CHANNEL_COLOURS.items():
        observation = blank(layout)
        put(observation, layout, channel, 20, 30)
        image = render_observation(observation, layout, scale=4, show_paths=False)
        patch = image[20 * 4 + 1, 30 * 4 + 1]
        assert tuple(int(v) for v in patch) == colour, channel


def test_a_more_dangerous_hazard_wins_a_contested_cell(layout):
    """Drawn in the encoder's own priority order, so the picture agrees with
    what actually owns the cell."""
    observation = blank(layout)
    put(observation, layout, "normal-enemy", 10, 10)
    put(observation, layout, "blast", 10, 10)
    image = render_observation(observation, layout, scale=4, show_paths=False)
    from dodge_royale.viewer import CHANNEL_COLOURS

    assert tuple(int(v) for v in image[41, 41]) == CHANNEL_COLOURS["blast"]


def test_an_expiring_blast_is_drawn_dimmer_than_a_fresh_one(layout):
    """Phase is the only thing separating them; size passes through twice."""
    fresh, expiring = blank(layout), blank(layout)
    for observation, phase in ((fresh, 1.0), (expiring, -0.9)):
        put(observation, layout, "blast", 12, 12)
        put(observation, layout, "blast-phase", 12, 12, phase)
    bright = render_observation(fresh, layout, scale=4, show_paths=False)[49, 49]
    dim = render_observation(expiring, layout, scale=4, show_paths=False)[49, 49]
    assert int(bright.sum()) > int(dim.sum())


def test_the_chosen_action_is_drawn_differently(layout):
    from dodge_royale.viewer import CHOSEN_COLOUR

    observation = blank(layout)
    base = layout.path_section.offset
    horizons = len(layout.horizons)
    # Action 3's path goes up and right; every other action stays at centre.
    for horizon in range(horizons):
        observation[base + (3 * horizons + horizon) * 2] = 0.5
        observation[base + (3 * horizons + horizon) * 2 + 1] = -0.5

    image = render_observation(observation, layout, scale=8, chosen=3)
    assert (image == np.array(CHOSEN_COLOUR, dtype=np.uint8)).all(-1).any(), (
        "the chosen path is highlighted"
    )


def test_paths_can_be_turned_off(layout):
    observation = blank(layout)
    base = layout.path_section.offset
    observation[base] = 0.5
    with_paths = render_observation(observation, layout, scale=8, show_paths=True)
    without = render_observation(observation, layout, scale=8, show_paths=False)
    assert not np.array_equal(with_paths, without)


def test_a_path_outside_the_window_does_not_crash(layout):
    """Paths are deliberately unclipped and may leave the window."""
    observation = blank(layout)
    base = layout.path_section.offset
    for index in range(layout.path_section.length):
        observation[base + index] = 9.0 if index % 2 == 0 else -9.0
    render_observation(observation, layout, scale=8)  # must not raise
