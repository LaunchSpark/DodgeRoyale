"""The session and the CLI, driven without launching a gym.

A fake client stands in for the child process, because the cases worth testing
are the ones where something goes wrong: a rejected checkpoint, a failed build,
an interrupt. A working gym cannot exercise any of them, and every one of them
must still close the process.
"""

from __future__ import annotations

import numpy as np
import pytest

from dodge_royale import training as training_module
from dodge_royale.policies import ARCHITECTURES, ROYALE_ARCHITECTURE
from dodge_royale.protocol import GymError, Layout, ProtocolError, ResetBatch, StepBatch, Transition
from dodge_royale.rewards import Rewards
from dodge_royale.metrics import MetricsCollector
from dodge_royale.train import build_parser, config_from_args, main
from dodge_royale.training import (
    DEFAULTS,
    CheckpointWriter,
    SessionConfig,
    build_model,
    check_env,
    describe,
    make_env,
    minibatch_size,
    session,
)
from dodge_royale.vec_env import RoyaleVecEnv


@pytest.fixture(scope="session")
def layout(manifest) -> Layout:
    return Layout.from_json(manifest["layout"])


class FakeGym:
    """A gym that never was, but closes like one."""

    def __init__(self, layout: Layout, envs: int = 2) -> None:
        self.layout = layout
        self.envs = envs
        self.closed = 0
        width = layout.observation_values
        self.initial = ResetBatch(
            seeds=tuple(range(envs)),
            observations=np.zeros((envs, width), dtype=np.float32),
        )

    def step(self, actions):
        width = self.layout.observation_values
        return StepBatch(
            transitions=tuple(
                Transition(
                    frame=1,
                    terminated=False,
                    truncated=False,
                    enemy_deaths=0,
                    episode_seed=index,
                    reset_seed=None,
                )
                for index in range(self.envs)
            ),
            observations=np.zeros((self.envs, width), dtype=np.float32),
            terminal={},
        )

    def reset(self, seed=None):
        return self.initial

    def close(self):
        self.closed += 1


@pytest.fixture
def fake_env(layout, monkeypatch):
    """Patch `make_env` so a session builds a fake gym rather than a child."""
    built: list[RoyaleVecEnv] = []

    def factory(config, rewards=None):
        config.validate()
        env = RoyaleVecEnv(
            client=FakeGym(layout, envs=config.envs), rewards=rewards or Rewards()
        )
        built.append(env)
        return env

    monkeypatch.setattr(training_module, "make_env", factory)
    return built


def small(**overrides) -> SessionConfig:
    base = dict(
        envs=2,
        n_steps=8,
        n_epochs=1,
        total_timesteps=16,
        device="cpu",
        minibatch_cap=128,
    )
    base.update(overrides)
    return SessionConfig(**base)


# --- defaults ------------------------------------------------------------


def test_the_defaults_are_the_ones_the_design_names():
    assert DEFAULTS.envs == 8
    assert DEFAULTS.n_steps == 1024
    assert DEFAULTS.enemies == 100
    assert DEFAULTS.max_frames == 3600
    assert DEFAULTS.hold_frames == 24
    assert DEFAULTS.threads == 2
    assert DEFAULTS.minibatch_cap == 128


def test_the_cli_defaults_match_the_session_defaults():
    """Two places that could drift, and a drift nobody would notice."""
    args = build_parser().parse_args([])
    assert config_from_args(args) == DEFAULTS


def test_hold_frames_is_not_an_action_repeat():
    """The plan's specific warning: they are different knobs.

    Reinterpreting one as the other would quarter the decision rate while
    every shape stayed correct.
    """
    help_text = build_parser().format_help()
    assert "not an action repeat" in help_text.lower()
    args = build_parser().parse_args(["--hold-frames", "48"])
    assert config_from_args(args).hold_frames == 48


# --- minibatch bounding --------------------------------------------------


def test_a_minibatch_is_bounded_independently_of_the_rollout():
    """Rollout size scales with envs; a minibatch that scaled with it would
    grow until a device ran out."""
    assert minibatch_size(8 * 1024, cap=128) == 128
    assert minibatch_size(64 * 1024, cap=128) == 128
    assert minibatch_size(8 * 1024) == 128


def test_a_small_smoke_rollout_gets_a_valid_size_not_the_cap():
    """A minibatch larger than the rollout is not a minibatch."""
    assert minibatch_size(64, cap=128) == 64
    assert minibatch_size(16, cap=128) == 16


def test_a_minibatch_always_divides_the_rollout():
    """PPO warns and truncates on an uneven split, losing samples quietly."""
    for envs in (1, 2, 3, 5, 7, 8, 64):
        for steps in (7, 8, 16, 100, 1024):
            rollout = envs * steps
            size = minibatch_size(rollout, cap=128)
            assert rollout % size == 0, (envs, steps, size)
            assert 1 <= size <= min(128, rollout)


def test_the_cap_can_be_overridden():
    assert minibatch_size(8 * 1024, cap=256) == 256
    assert minibatch_size(8 * 1024, cap=64) == 64


def test_an_impossible_bound_is_refused():
    with pytest.raises(ValueError):
        minibatch_size(0)
    with pytest.raises(ValueError):
        minibatch_size(1024, cap=0)


# --- the discounts, on both paths ----------------------------------------


def test_a_fresh_model_takes_the_tables_discounts(fake_env, layout):
    config = small()
    with session(config) as env:
        model = build_model(config, env)
    entry = ARCHITECTURES[ROYALE_ARCHITECTURE]
    assert model.gamma == pytest.approx(entry.gamma)
    assert model.gae_lambda == pytest.approx(entry.gae_lambda)
    assert model.gae_lambda != 0.95, "PPO's default is the value the plan warns about"


def test_a_resumed_model_takes_the_tables_discounts_over_the_saved_ones(
    fake_env, layout, tmp_path
):
    """SB3 restores what was saved, so a resume must override it."""
    config = small(checkpoint_dir=str(tmp_path))
    with session(config) as env:
        model = build_model(config, env)
        model.gamma = 0.99
        model.gae_lambda = 0.95  # as if saved before the table was corrected
        path = tmp_path / "stale.zip"
        model.save(path)

    with session(config) as env:
        resumed = build_model(config, env, resume=path)
    entry = ARCHITECTURES[ROYALE_ARCHITECTURE]
    assert resumed.gae_lambda == pytest.approx(entry.gae_lambda)
    assert resumed.gamma == pytest.approx(entry.gamma)


def test_the_live_layout_reaches_a_new_models_extractor(fake_env, layout):
    config = small()
    with session(config) as env:
        model = build_model(config, env)
    assert model.policy.features_extractor.layout == layout


# --- reload compatibility ------------------------------------------------


def test_a_checkpoint_from_another_layout_is_refused_before_the_env_attaches(
    fake_env, layout, tmp_path, monkeypatch
):
    """The order matters: a mismatch must be a refusal, not a run.

    Checked by making the compatibility check raise and asserting PPO.load was
    never reached, so the refusal cannot be happening after a model was built
    around the wrong observations.
    """
    config = small()
    with session(config) as env:
        model = build_model(config, env)
        path = tmp_path / "other.zip"
        model.save(path)

    loaded: list[object] = []
    monkeypatch.setattr(
        training_module, "require_loadable",
        lambda *_: (_ for _ in ()).throw(ProtocolError("layouts differ")),
    )
    import stable_baselines3

    monkeypatch.setattr(
        stable_baselines3.PPO, "load",
        classmethod(lambda cls, *a, **k: loaded.append(a) or None),
    )
    with session(config) as env:
        with pytest.raises(ProtocolError, match="layouts differ"):
            build_model(config, env, resume=path)
    assert loaded == [], "the checkpoint must be refused before it is loaded"


def test_a_missing_checkpoint_is_named_rather_than_traced(fake_env, tmp_path):
    config = small()
    with session(config) as env:
        with pytest.raises(GymError, match="no checkpoint"):
            build_model(config, env, resume=tmp_path / "absent.zip")


def test_changing_the_arena_does_not_change_the_model_shape(fake_env, layout, tmp_path):
    """Enemy count and frame budget restart envs; they are not model shape.

    The observation layout does not depend on them, so a checkpoint has to
    survive a change to either.
    """
    first = small(enemies=100, max_frames=3600)
    with session(first) as env:
        model = build_model(first, env)
        path = tmp_path / "run.zip"
        model.save(path)

    second = small(enemies=25, max_frames=600)
    with session(second) as env:
        resumed = build_model(second, env, resume=path)
    assert resumed.policy.features_extractor.layout == layout


# --- cleanup -------------------------------------------------------------


def test_the_session_closes_the_gym_on_the_way_out(fake_env):
    config = small()
    with session(config) as env:
        pass
    assert env._client.closed == 1


def test_the_session_closes_the_gym_when_the_body_raises(fake_env):
    config = small()
    with pytest.raises(RuntimeError, match="boom"):
        with session(config) as env:
            raise RuntimeError("boom")
    assert env._client.closed == 1, "a failure must not leak a child process"


def test_a_failed_model_build_still_closes_the_gym(fake_env, tmp_path):
    config = small()
    captured = {}
    with pytest.raises(GymError):
        with session(config) as env:
            captured["env"] = env
            build_model(config, env, resume=tmp_path / "missing.zip")
    assert captured["env"]._client.closed == 1


def test_an_interrupted_run_saves_before_it_stops(fake_env, layout, tmp_path, monkeypatch):
    """Losing an interrupted run's weights would make stopping expensive."""
    config = small(checkpoint_dir=str(tmp_path), run_name="interrupted")

    import stable_baselines3

    def interrupt(self, *args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(stable_baselines3.PPO, "learn", interrupt)
    with pytest.raises(GymError, match="interrupted"):
        training_module.train(config)

    saved = list(tmp_path.glob("interrupted-*.zip"))
    assert saved, "the weights survived the interrupt"
    assert fake_env[-1]._client.closed == 1, "and the gym still closed"


# --- check-env -----------------------------------------------------------


def test_check_env_drives_the_batch_rather_than_a_scalar_checker(fake_env, layout):
    config = small()
    with session(config) as env:
        check_env(env)  # must not raise


def test_check_env_reports_a_batch_of_the_wrong_shape(fake_env, layout, monkeypatch):
    config = small()
    with session(config) as env:
        monkeypatch.setattr(
            env, "reset", lambda: np.zeros((env.num_envs, 3), dtype=np.float32)
        )
        with pytest.raises(ValueError, match="expected"):
            check_env(env)


# --- configuration validation --------------------------------------------


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("envs", 0),
        ("n_steps", 0),
        ("threads", 0),
        ("hold_frames", 0),
        ("minibatch_cap", 0),
        ("total_timesteps", 0),
        ("max_frames", 0),
    ],
)
def test_an_impossible_configuration_is_refused_before_a_gym_launches(field, value):
    with pytest.raises(ValueError, match=field):
        SessionConfig(**{field: value}).validate()


def test_an_unknown_architecture_names_the_ones_that_exist():
    with pytest.raises(ValueError, match="velocity-flow-royale"):
        SessionConfig(architecture="velocity-flow-v2").validate()


def test_zero_enemies_is_allowed_for_a_controlled_run():
    SessionConfig(enemies=0).validate()


# --- telemetry -----------------------------------------------------------


def test_the_collector_counts_finished_episodes_not_auto_resets():
    """One collector serves the CLI and the dashboard, so this is the same
    code path both report from."""
    recorder = MetricsCollector()
    recorder.locals = {
        "infos": [
            {"frames": 12},  # still running: no summary, nothing recorded
            {
                "episode_summary": {
                    "frames": 60,
                    "run_frames": 60,
                    "seconds": 1.0,
                    "enemies_destroyed": 3,
                    "return": 0.5,
                    "died": True,
                }
            },
        ]
    }
    assert recorder._on_step() is True
    snapshot = recorder.snapshot()
    assert snapshot.episodes == 1
    assert snapshot.survival_seconds == pytest.approx(1.0)
    assert snapshot.deaths == 1
    assert snapshot.timeouts == 0


# --- checkpoints ---------------------------------------------------------


def test_checkpoints_are_named_by_run_and_step(tmp_path, fake_env, layout):
    config = small(checkpoint_dir=str(tmp_path), run_name="alpha")
    writer = CheckpointWriter.for_config(config)
    with session(config) as env:
        model = build_model(config, env)
        first = writer.save(model, step=1024)
        final = writer.save(model)
    assert first.name == "alpha-000001024.zip"
    assert final.name == "alpha-final.zip"
    assert writer.latest() is not None


def test_latest_is_none_when_nothing_has_been_saved(tmp_path):
    writer = CheckpointWriter(directory=tmp_path / "empty", run_name="alpha")
    assert writer.latest() is None


# --- the CLI -------------------------------------------------------------


def test_the_cli_has_no_game_switch():
    """Royale only: a selector with one setting is a second code path."""
    help_text = build_parser().format_help()
    assert "--game" not in help_text


def test_a_dry_run_prints_the_configuration_without_launching_a_gym(capsys, monkeypatch):
    def refuse(*_args, **_kwargs):
        raise AssertionError("a dry run must not launch a gym")

    monkeypatch.setattr(training_module, "make_env", refuse)
    assert main(["--dry-run"]) == 0
    printed = capsys.readouterr().out
    assert "velocity-flow-royale" in printed
    assert "8192 samples" in printed, "8 envs x 1024 steps"
    assert "minibatch      128" in printed


def test_the_summary_reports_the_discounts_that_will_be_used():
    entry = ARCHITECTURES[ROYALE_ARCHITECTURE]
    printed = describe(DEFAULTS)
    assert f"{entry.gamma:.6f}" in printed
    assert f"{entry.gae_lambda:.6f}" in printed


def test_check_env_through_the_cli(fake_env, capsys):
    assert main(["--check-env", "--envs", "2", "--n-steps", "8"]) == 0
    printed = capsys.readouterr().out
    assert "resets and steps" in printed
    assert fake_env[-1]._client.closed == 1, "check-env closes the gym too"


def test_asking_for_two_different_checkpoints_is_refused():
    with pytest.raises(SystemExit):
        main(["--resume", "a.zip", "--resume-latest"])


def test_resume_latest_without_a_checkpoint_is_refused(tmp_path):
    with pytest.raises(SystemExit):
        main(["--resume-latest", "--checkpoint-dir", str(tmp_path)])


def test_an_invalid_pairing_is_reported_rather_than_traced(capsys):
    with pytest.raises(SystemExit):
        main(["--envs", "0"])


def test_a_gym_failure_is_a_message_and_a_non_zero_exit(monkeypatch, capsys):
    def fail(*_args, **_kwargs):
        raise GymError("no gym binary at nowhere")

    monkeypatch.setattr(training_module, "make_env", fail)
    assert main(["--envs", "2", "--n-steps", "8", "--total-timesteps", "16"]) == 1
    assert "no gym binary" in capsys.readouterr().err


def test_a_short_training_run_through_the_cli(fake_env, tmp_path, capsys):
    """The whole path: parse, build, learn, save, close."""
    code = main(
        [
            "--envs", "2",
            "--n-steps", "8",
            "--n-epochs", "1",
            "--total-timesteps", "16",
            "--device", "cpu",
            "--checkpoint-dir", str(tmp_path),
            "--run-name", "smoke",
        ]
    )
    assert code == 0
    printed = capsys.readouterr().out
    assert "saved" in printed
    assert (tmp_path / "smoke-final.zip").exists()
    assert fake_env[-1]._client.closed == 1


def test_the_package_exposes_the_entry_point_module():
    """`python -m dodge_royale.train` has to resolve."""
    import importlib

    module = importlib.import_module("dodge_royale.train")
    assert callable(module.main)


def test_no_dodgeai_import_is_required(monkeypatch):
    """The trainer must not need DodgeAI on the path to run."""
    import sys

    for name in list(sys.modules):
        assert not name.startswith("dodge."), f"{name} leaked in from DodgeAI"
    assert "dodge" not in sys.modules or sys.modules["dodge"].__name__ != "dodge"
