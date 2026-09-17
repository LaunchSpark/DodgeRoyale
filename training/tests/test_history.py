"""The per-episode training history a model carries with it."""

from __future__ import annotations

import json
import time

import pytest

from dodge_royale.history import (
    EpisodeRecord,
    HistoryWriter,
    history_path,
    iter_history,
    read_history,
    run_history_path,
)
from dodge_royale.metrics import MetricsCollector
from dodge_royale.telemetry import FRAMES_PER_SECOND


def summary(frames: int = 60, died: bool = True, kills: int = 0, reward: float = 1.0):
    return {
        "frames": frames,
        "run_frames": frames,
        "seconds": frames / FRAMES_PER_SECOND,
        "enemies_destroyed": kills,
        "return": reward,
        "died": died,
    }


def record(**overrides) -> EpisodeRecord:
    base = dict(
        timesteps=1024, env=0, frames=300, enemies=100, reward=1.5, died=True,
        enemies_destroyed=2, episode_seed=7, hold_frames=24, recorded_at=time.time(),
    )
    base.update(overrides)
    return EpisodeRecord(**base)


# --- the record ------------------------------------------------------------


def test_survival_is_frames_at_the_simulations_rate():
    assert record(frames=60).seconds == pytest.approx(1.0)
    assert record(frames=347).seconds == pytest.approx(347 / 60)


def test_a_record_carries_the_enemy_count_it_was_played_against():
    """The whole point: three seconds against 12 enemies is not three seconds
    against 100, and the count changes between runs."""
    easy = record(frames=300, enemies=12)
    hard = record(frames=300, enemies=100)
    assert easy.seconds == hard.seconds
    assert easy.enemies != hard.enemies


def test_a_record_is_built_from_an_episode_summary():
    built = EpisodeRecord.from_summary(
        summary(frames=347, died=True, kills=3, reward=-1.2),
        timesteps=5000, env=1, enemies=100, hold_frames=24,
        episode_seed=99, recorded_at=1.0,
    )
    assert built.frames == 347
    assert built.enemies == 100
    assert built.enemies_destroyed == 3
    assert built.reward == pytest.approx(-1.2)
    assert built.died is True
    assert built.episode_seed == 99
    assert built.env == 1
    assert built.timesteps == 5000


# --- the file --------------------------------------------------------------


def test_episodes_round_trip_through_the_file(tmp_path):
    path = tmp_path / "run.history.jsonl"
    with HistoryWriter(path) as writer:
        writer.extend([record(frames=100), record(frames=200, enemies=40)])
    back = read_history(path)
    assert [r.frames for r in back] == [100, 200]
    assert [r.enemies for r in back] == [100, 40]


def test_each_episode_is_flushed_as_it_happens(tmp_path):
    """A run that is killed keeps every episode it actually finished."""
    path = tmp_path / "run.history.jsonl"
    writer = HistoryWriter(path)
    writer.append(record(frames=11))
    # Not closed, not exited: the record must already be on disk.
    assert len(read_history(path)) == 1
    writer.append(record(frames=22))
    assert [r.frames for r in read_history(path)] == [11, 22]
    writer.close()


def test_appending_continues_an_existing_file(tmp_path):
    path = tmp_path / "run.history.jsonl"
    with HistoryWriter(path) as writer:
        writer.append(record(frames=1))
    with HistoryWriter(path) as writer:
        writer.append(record(frames=2))
    assert [r.frames for r in read_history(path)] == [1, 2]


def test_a_truncated_last_line_does_not_lose_the_rest(tmp_path):
    """A process killed mid-write must not make its history unreadable."""
    path = tmp_path / "run.history.jsonl"
    with HistoryWriter(path) as writer:
        writer.extend([record(frames=1), record(frames=2)])
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"timesteps": 3, "frames":')  # cut off
    assert [r.frames for r in read_history(path)] == [1, 2]


def test_an_unknown_field_is_ignored_rather_than_fatal(tmp_path):
    path = tmp_path / "run.history.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        row = {**{f: 0 for f in ("timesteps", "env", "frames", "enemies",
                                 "enemies_destroyed", "episode_seed", "hold_frames")},
               "reward": 0.0, "died": False, "recorded_at": 0.0,
               "something_added_later": 1}
        handle.write(json.dumps(row) + "\n")
    assert len(read_history(path)) == 1


def test_iterating_gives_the_same_records(tmp_path):
    path = tmp_path / "run.history.jsonl"
    with HistoryWriter(path) as writer:
        writer.extend([record(frames=n) for n in (1, 2, 3)])
    assert [r.frames for r in iter_history(path)] == [1, 2, 3]


def test_missing_history_reads_as_empty(tmp_path):
    assert read_history(tmp_path / "absent.jsonl") == []
    assert list(iter_history(tmp_path / "absent.jsonl")) == []


def test_a_checkpoints_history_sits_beside_it(tmp_path):
    assert history_path(tmp_path / "royale-final.zip").name == "royale-final.history.jsonl"
    assert run_history_path(tmp_path, "royale").name == "royale.history.jsonl"


# --- the collector writes it ----------------------------------------------


def test_the_collector_records_every_finished_episode(tmp_path):
    path = tmp_path / "run.history.jsonl"
    with HistoryWriter(path) as writer:
        collector = MetricsCollector(history=writer, enemies=64, hold_frames=24)
        collector.locals = {
            "infos": [
                {"episode_summary": summary(frames=120, died=True), "episode_seed": 11},
                {"frames": 5},  # still running: nothing to record
                {"episode_summary": summary(frames=240, died=False), "episode_seed": 12},
            ]
        }
        collector._on_step()

    records = read_history(path)
    assert len(records) == 2, "only finished episodes are recorded"
    assert [r.frames for r in records] == [120, 240]
    assert [r.died for r in records] == [True, False]
    assert [r.episode_seed for r in records] == [11, 12]
    assert {r.enemies for r in records} == {64}
    assert [r.env for r in records] == [0, 2], "the env index is the one that finished"


def test_a_collector_without_a_history_still_reports_metrics():
    collector = MetricsCollector()
    collector.locals = {"infos": [{"episode_summary": summary(frames=60)}]}
    collector._on_step()
    assert collector.snapshot().episodes == 1
