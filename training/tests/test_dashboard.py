"""The worker's lifecycle, the metric definitions, and the notebook itself.

The worker is tested against a fake gym, because the cases worth testing are
the ones a working run cannot produce on demand: a start refused while another
run is going, a pause that a stop has to interrupt, a failure that must still
close the child.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from dodge_royale import worker as worker_module
from dodge_royale.metrics import METRIC_KEYS, METRICS, MetricsCollector, Snapshot
from dodge_royale.policies import ROYALE_ARCHITECTURE
from dodge_royale.protocol import Layout, ProtocolError, ResetBatch, StepBatch, Transition
from dodge_royale.rewards import Rewards
from dodge_royale.telemetry import FRAMES_PER_SECOND
from dodge_royale.training import SessionConfig
from dodge_royale.vec_env import RoyaleVecEnv
from dodge_royale.worker import (
    TrainingWorker,
    WorkerState,
    active_worker,
    start_worker,
    stop_active_worker,
)


@pytest.fixture(scope="session")
def layout(manifest) -> Layout:
    return Layout.from_json(manifest["layout"])


class FakeGym:
    """A gym that never was, which finishes an episode every few steps."""

    def __init__(self, layout: Layout, envs: int = 2, episode_every: int = 4) -> None:
        self.layout = layout
        self.envs = envs
        self.closed = 0
        self.steps = 0
        self.episode_every = episode_every
        width = layout.observation_values
        self.initial = ResetBatch(
            seeds=tuple(range(envs)),
            observations=np.zeros((envs, width), dtype=np.float32),
        )

    def step(self, actions):
        self.steps += 1
        width = self.layout.observation_values
        done = self.steps % self.episode_every == 0
        frame = self.episode_every if done else self.steps % self.episode_every
        return StepBatch(
            transitions=tuple(
                Transition(
                    frame=frame,
                    terminated=False,
                    truncated=done,
                    enemy_deaths=0,
                    episode_seed=index,
                    reset_seed=1000 + index if done else None,
                )
                for index in range(self.envs)
            ),
            observations=np.zeros((self.envs, width), dtype=np.float32),
            terminal=(
                {index: np.zeros(width, dtype=np.float32) for index in range(self.envs)}
                if done
                else {}
            ),
        )

    def reset(self, seed=None):
        return self.initial

    def close(self):
        self.closed += 1


@pytest.fixture
def fake_gyms(layout, monkeypatch):
    """Make every session build a fake gym, and record them."""
    built: list[FakeGym] = []

    def factory(config, rewards=None):
        config.validate()
        # The real gym announces the hold it was configured with, so the fake
        # has to as well: `hold_frames` is part of the layout, and a fake that
        # ignored it would make a hold change invisible to the very check that
        # is supposed to catch it.
        announced = layout.as_dict()
        announced["hold_frames"] = config.hold_frames
        gym = FakeGym(Layout.from_json(announced), envs=config.envs)
        built.append(gym)
        return RoyaleVecEnv(client=gym, rewards=rewards or Rewards())

    monkeypatch.setattr(worker_module, "session", _session_of(factory))
    return built


def _session_of(factory):
    import contextlib

    @contextlib.contextmanager
    def session(config, rewards=None):
        env = factory(config, rewards)
        try:
            yield env
        finally:
            env.close()

    return session


@pytest.fixture(autouse=True)
def no_leftover_worker():
    """Every test leaves the module-level worker stopped."""
    yield
    stop_active_worker(10.0)
    worker_module._ACTIVE = None


def small(**overrides) -> SessionConfig:
    base = dict(
        envs=2,
        n_steps=8,
        n_epochs=1,
        total_timesteps=32,
        device="cpu",
        run_name="dash-test",
    )
    base.update(overrides)
    return SessionConfig(**base)


def wait_for(predicate, timeout: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


# --- metric definitions ---------------------------------------------------


def test_every_definition_has_a_field_on_the_snapshot():
    """The registry and the snapshot must not drift.

    The dashboard renders whatever the registry lists, so a definition with no
    field behind it would render as an empty row forever.
    """
    fields = set(Snapshot().as_dict())
    for metric in METRICS:
        assert metric.key in fields, f"{metric.key} has no field on Snapshot"


def test_the_metrics_the_plan_asks_for_are_all_present():
    required = {
        "survival_seconds",
        "episode_return",
        "deaths",
        "timeouts",
        "steps_per_second",
        "value_loss",
        "policy_loss",
        "entropy_loss",
        "device",
    }
    assert required <= set(METRIC_KEYS)


def test_definitions_are_unique_and_described():
    keys = [metric.key for metric in METRICS]
    assert len(keys) == len(set(keys))
    for metric in METRICS:
        assert metric.label and metric.description
        assert metric.group in {"episode", "throughput", "optimizer", "run"}


def test_a_missing_value_renders_rather_than_raising():
    """An empty run has no episodes; every row must still draw."""
    rows = Snapshot().rows()
    assert len(rows) == len(METRICS)
    assert all(isinstance(shown, str) and shown for _, shown in rows)


def test_optimizer_metrics_are_absent_rather_than_zero_before_the_first_update():
    """Zero would chart as a real reading of a loss that does not exist yet."""
    collector = MetricsCollector()
    snapshot = collector.snapshot()
    assert snapshot.value_loss is None
    assert snapshot.policy_loss is None
    assert snapshot.updates == 0


def test_survival_is_reported_in_seconds_at_the_simulations_rate():
    collector = MetricsCollector()
    collector.locals = {
        "infos": [
            {
                "episode_summary": {
                    "seconds": 60 / FRAMES_PER_SECOND,
                    "return": 1.5,
                    "died": True,
                }
            }
        ]
    }
    collector._on_step()
    snapshot = collector.snapshot()
    assert snapshot.survival_seconds == pytest.approx(1.0)
    assert snapshot.episode_return == pytest.approx(1.5)
    assert snapshot.deaths == 1
    assert snapshot.timeouts == 0


def test_deaths_and_timeouts_are_counted_apart():
    collector = MetricsCollector()
    collector.locals = {
        "infos": [
            {"episode_summary": {"seconds": 1.0, "return": 0.0, "died": True}},
            {"episode_summary": {"seconds": 2.0, "return": 0.0, "died": False}},
        ]
    }
    collector._on_step()
    snapshot = collector.snapshot()
    assert (snapshot.deaths, snapshot.timeouts, snapshot.episodes) == (1, 1, 2)


def test_the_episode_window_is_bounded():
    """A long run must not accumulate every episode it ever finished."""
    collector = MetricsCollector(window=10)
    for index in range(50):
        collector.locals = {
            "infos": [
                {"episode_summary": {"seconds": float(index), "return": 0.0, "died": False}}
            ]
        }
        collector._on_step()
    assert len(collector._lengths) == 10
    assert collector.episodes == 50, "the count is of the whole run"


# --- lifecycle ------------------------------------------------------------


def test_a_run_starts_reports_and_stops(fake_gyms):
    worker = TrainingWorker(small(), publish_interval=0.0)
    worker.start()
    assert worker.wait(60), "the run finished"
    status = worker.status()
    assert status.state is WorkerState.STOPPED
    assert status.error is None
    assert status.metrics.timesteps >= 32
    assert fake_gyms[0].closed == 1


def test_starting_twice_is_refused(fake_gyms):
    worker = TrainingWorker(small(total_timesteps=100_000), publish_interval=0.0)
    worker.start()
    assert wait_for(lambda: worker.state is WorkerState.RUNNING)
    with pytest.raises(RuntimeError, match="already"):
        worker.start()
    worker.stop()
    assert len(fake_gyms) == 1, "the refusal did not build a second gym"


def test_the_module_level_worker_refuses_a_second_run(fake_gyms):
    """A marimo rerun must not be able to launch a parallel gym."""
    start_worker(small(total_timesteps=100_000))
    assert wait_for(lambda: active_worker().state is WorkerState.RUNNING)
    with pytest.raises(RuntimeError, match="stop it before starting another"):
        start_worker(small())
    assert len(fake_gyms) == 1
    stop_active_worker()


def test_active_worker_is_the_same_object_across_calls(fake_gyms):
    """What makes a rerun find the run instead of starting one."""
    worker = start_worker(small(total_timesteps=100_000))
    assert active_worker() is worker
    assert active_worker() is worker
    worker.stop()


def test_a_finished_run_can_be_replaced(fake_gyms):
    first = start_worker(small())
    assert first.wait(60)
    second = start_worker(small())
    assert second is not first
    second.stop()
    assert len(fake_gyms) == 2


def test_pause_and_resume(fake_gyms):
    worker = TrainingWorker(small(total_timesteps=100_000), publish_interval=0.0)
    worker.start()
    assert wait_for(lambda: worker.state is WorkerState.RUNNING)

    worker.pause()
    assert wait_for(lambda: worker.state is WorkerState.PAUSED), "it paused"
    paused_at = fake_gyms[0].steps
    time.sleep(0.3)
    assert fake_gyms[0].steps - paused_at <= 1, "a paused run is not stepping"

    worker.resume_training()
    assert wait_for(lambda: worker.state is WorkerState.RUNNING), "it resumed"
    assert wait_for(lambda: fake_gyms[0].steps > paused_at + 1), "and is stepping again"
    worker.stop()


def test_a_paused_run_can_still_be_stopped(fake_gyms):
    """Otherwise pausing would be a way to wedge the dashboard."""
    worker = TrainingWorker(small(total_timesteps=100_000), publish_interval=0.0)
    worker.start()
    assert wait_for(lambda: worker.state is WorkerState.RUNNING)
    worker.pause()
    assert wait_for(lambda: worker.state is WorkerState.PAUSED)

    worker.stop(timeout=30)
    assert worker.state is WorkerState.STOPPED
    assert fake_gyms[0].closed == 1


def test_stop_closes_the_gym(fake_gyms):
    worker = TrainingWorker(small(total_timesteps=100_000), publish_interval=0.0)
    worker.start()
    assert wait_for(lambda: worker.state is WorkerState.RUNNING)
    worker.stop()
    assert worker.state is WorkerState.STOPPED
    assert fake_gyms[0].closed == 1, "the child process must not outlive the run"


def test_stop_is_idempotent_and_safe_before_a_start(fake_gyms):
    worker = TrainingWorker(small())
    worker.stop()  # never started
    worker.stop()
    assert worker.state in (WorkerState.IDLE, WorkerState.STOPPED)


def test_a_failure_is_reported_and_still_closes_the_gym(fake_gyms, monkeypatch):
    def explode(*_args, **_kwargs):
        raise RuntimeError("the model would not build")

    monkeypatch.setattr(worker_module, "build_model", explode)
    worker = TrainingWorker(small())
    worker.start()
    assert worker.wait(60)

    status = worker.status()
    assert status.state is WorkerState.FAILED
    assert "the model would not build" in status.error
    assert fake_gyms[0].closed == 1, "a failed build must not leak a child"


def test_a_failed_run_does_not_block_the_next_one(fake_gyms, monkeypatch):
    calls = {"n": 0}
    real = worker_module.build_model

    def once(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("first attempt fails")
        return real(*args, **kwargs)

    monkeypatch.setattr(worker_module, "build_model", once)
    first = start_worker(small())
    assert first.wait(60)
    assert first.state is WorkerState.FAILED

    second = start_worker(small())
    assert second.wait(60)
    assert second.state is WorkerState.STOPPED


def test_stop_active_worker_is_safe_with_nothing_running():
    stop_active_worker(1.0)  # must not raise


# --- snapshots ------------------------------------------------------------


def test_status_is_a_snapshot_the_reader_can_keep(fake_gyms):
    """A cell holds one of these while the trainer keeps running."""
    worker = TrainingWorker(small(total_timesteps=100_000), publish_interval=0.0)
    worker.start()
    assert wait_for(lambda: worker.status().metrics.timesteps > 0)

    held = worker.status()
    steps_then = held.metrics.timesteps
    history_then = len(held.history)
    assert wait_for(lambda: worker.status().metrics.timesteps > steps_then)

    assert held.metrics.timesteps == steps_then, "the held snapshot did not move"
    assert len(held.history) == history_then
    worker.stop()


def test_history_is_bounded(fake_gyms):
    worker = TrainingWorker(
        small(total_timesteps=100_000), history=5, publish_interval=0.0
    )
    worker.start()
    assert wait_for(lambda: len(worker.status().history) == 5)
    time.sleep(0.3)
    assert len(worker.status().history) <= 5, "a long run must not grow this"
    worker.stop()


def test_status_is_readable_from_another_thread(fake_gyms):
    """The dashboard reads from marimo's thread while the worker writes."""
    worker = TrainingWorker(small(total_timesteps=100_000), publish_interval=0.0)
    worker.start()
    assert wait_for(lambda: worker.state is WorkerState.RUNNING)

    problems: list[BaseException] = []

    def reader():
        try:
            for _ in range(200):
                status = worker.status()
                assert isinstance(status.metrics.timesteps, int)
                tuple(status.history)
        except BaseException as error:  # noqa: BLE001 - reported below
            problems.append(error)

    threads = [threading.Thread(target=reader) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    worker.stop()
    assert not problems, problems


# --- checkpoints ----------------------------------------------------------


def test_a_save_request_is_serviced_on_the_training_thread(fake_gyms, tmp_path):
    worker = TrainingWorker(
        small(total_timesteps=100_000, checkpoint_dir=str(tmp_path)),
        publish_interval=0.0,
    )
    worker.start()
    assert wait_for(lambda: worker.state is WorkerState.RUNNING)

    worker.request_save()
    assert wait_for(lambda: worker.status().checkpoints), "a checkpoint appeared"
    saved = worker.status().checkpoints[0]
    assert saved.exists()
    worker.stop()


def test_a_finished_run_saves_its_weights(fake_gyms, tmp_path):
    worker = TrainingWorker(small(checkpoint_dir=str(tmp_path)))
    worker.start()
    assert worker.wait(60)
    assert worker.status().checkpoints, "reaching the budget keeps the weights"


def test_a_checkpoint_from_another_layout_is_refused(fake_gyms, layout, tmp_path):
    """Including a changed prediction hold, which is part of the layout.

    Nothing about the observation's *shape* changes with the hold, so this is
    exactly the case that would load and train on happily.
    """
    from stable_baselines3 import PPO

    from dodge_royale.policies import ROYALE
    from dodge_royale.training import build_model, session

    config = small(checkpoint_dir=str(tmp_path))
    with session(config) as env:
        model = build_model(config, env)
        path = tmp_path / "trained.zip"
        model.save(path)

    # Same everything except the hold, which the gym announces in its layout.
    changed = small(hold_frames=config.hold_frames + 1, checkpoint_dir=str(tmp_path))
    worker = TrainingWorker(changed, resume=path)
    worker.start()
    assert worker.wait(60)

    status = worker.status()
    assert status.state is WorkerState.FAILED
    assert "hold_frames" in status.error or "not the one this policy" in status.error
    assert fake_gyms[-1].closed == 1, "the refusal still closed the gym"


# --- the notebook ---------------------------------------------------------


def test_the_notebook_is_a_valid_marimo_app():
    import dodge_royale.dashboard as dashboard

    assert hasattr(dashboard, "app")
    # marimo raises on duplicate definitions across cells, so constructing the
    # app at import time is itself the check that the notebook is coherent.
    assert dashboard.app is not None


def test_the_notebook_declares_its_dependencies():
    from pathlib import Path

    source = Path(dashboard_path()).read_text(encoding="utf-8")
    assert source.startswith("# /// script"), "PEP 723 header"
    assert "dodge-royale-training[dashboard]" in source


def test_the_notebook_never_constructs_a_worker_directly():
    """A cell that built one would build another on every rerun."""
    from pathlib import Path

    source = Path(dashboard_path()).read_text(encoding="utf-8")
    assert "TrainingWorker(" not in source, (
        "the notebook must go through start_worker/active_worker, which enforce "
        "a single run"
    )


def test_watch_agent_is_documented_as_unavailable():
    from pathlib import Path

    source = Path(dashboard_path()).read_text(encoding="utf-8")
    assert "Watch Agent is not available" in source
    assert "autopilot" in source


def dashboard_path() -> str:
    import dodge_royale.dashboard as dashboard

    return dashboard.__file__
