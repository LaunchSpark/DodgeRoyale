"""Cross-language acceptance: the real binary, end to end.

Everything here drives the release gym as a child process. Where the other
suites use fakes to reach states a real run will not produce on demand, these
deliberately do not: the point is that the two languages agree about the
bytes, the seeds and the numbers when nothing is mocked.

Every test has a deadline, and every one closes its child however it ends.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest

from dodge_royale.metrics import MetricsCollector
from dodge_royale.policies import ARCHITECTURES, ROYALE_ARCHITECTURE, require_loadable
from dodge_royale.protocol import GymClient, GymError, ProtocolError, find_binary
from dodge_royale.rewards import Rewards
from dodge_royale.training import SessionConfig, build_model, minibatch_size, session
from dodge_royale.vec_env import RoyaleVecEnv


def gym_available() -> bool:
    try:
        find_binary()
    except GymError:
        return False
    return True


pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not gym_available(), reason="no gym binary built"),
]

#: Small enough that every test here finishes in seconds, large enough that
#: episodes actually end.
SMALL = dict(envs=2, enemies=12, max_frames=8, threads=1)


def client(**overrides) -> GymClient:
    settings = {**SMALL, "seed": 7}
    settings.update(overrides)
    return GymClient(**settings)


# --- the protocol, against the real server --------------------------------


def test_the_binary_is_the_one_we_think_it_is():
    """The plan asks for an exact absolute path, so record which one ran."""
    binary = find_binary()
    assert binary.is_absolute()
    assert binary.exists()
    # `DODGE_ROYALE_BIN` wins when set; otherwise it is this checkout's release
    # build. Either way the test reports what it actually drove.
    print(f"\ngym binary: {binary}")


def test_handshake_step_reset_and_close():
    with client() as gym:
        assert gym.handshake.protocol_version == 1
        assert gym.handshake.envs == 2
        assert gym.handshake.root_seed == 7
        assert gym.handshake.enemy_count == 12
        assert gym.layout.observation_values == 28_782

        frame_zero = gym.initial
        assert frame_zero.observations.shape == (2, gym.layout.observation_values)

        batch = gym.step([2, 5])
        assert batch.transitions[0].frame == 1
        assert np.isfinite(batch.observations).all()

        again = gym.reset(7)
        assert again.seeds == frame_zero.seeds
        assert np.array_equal(again.observations, frame_zero.observations)
    assert gym.returncode == 0, "CLOSE exits cleanly"


def test_an_invalid_action_is_an_error_record_and_a_failed_exit():
    """The error path, which must end the session rather than half-step it."""
    gym = client()
    try:
        # The client refuses it before it reaches the wire, which is a plain
        # ValueError: nothing was sent, so there is no session to fail.
        with pytest.raises(ValueError, match="not one of"):
            gym.step([0, 99])
        # Send a bad action past the client's own check, the way a broken
        # client would, and the server must answer with an ERROR record.
        gym._send(bytes([0x01]) + (2).to_bytes(4, "little") + bytes([0, 200]))
        with pytest.raises((GymError, ProtocolError)):
            gym._read_step()
    finally:
        gym.close()
    assert gym.returncode not in (0, None), "a protocol error is a failed exit"


def test_an_unseeded_reset_advances_the_stream_and_a_seeded_one_repeats_it():
    with client() as gym:
        first = gym.initial.seeds
        advanced = gym.reset().seeds
        assert advanced != first, "an unseeded reset continues the stream"
        repeated = gym.reset(7).seeds
        assert repeated == first, "a seeded reset returns to episode zero"


# --- replay across processes and worker counts ----------------------------

REPLAY_ACTIONS = [
    [1, 2], [3, 4], [5, 6], [7, 8], [0, 1], [2, 3], [4, 5], [6, 7],
]


def replay(threads: int, seed: int = 11) -> tuple[list, list]:
    """Drive a fresh process through a fixed action script."""
    observations, metadata = [], []
    with GymClient(**{**SMALL, "seed": seed, "threads": threads}) as gym:
        observations.append(gym.initial.observations.copy())
        metadata.append(tuple(gym.initial.seeds))
        for actions in REPLAY_ACTIONS:
            batch = gym.step(actions)
            observations.append(batch.observations.copy())
            metadata.append(
                tuple(
                    (t.frame, t.terminated, t.truncated, t.enemy_deaths, t.episode_seed, t.reset_seed)
                    for t in batch.transitions
                )
            )
    return observations, metadata


def test_the_same_seed_replays_identically_in_a_second_process():
    """Same build, same platform, byte-identical. Not a cross-platform claim."""
    first_obs, first_meta = replay(threads=1)
    second_obs, second_meta = replay(threads=1)
    assert first_meta == second_meta
    for one, two in zip(first_obs, second_obs):
        assert np.array_equal(one, two), "observations differ between processes"


def test_worker_count_does_not_change_the_result():
    one_obs, one_meta = replay(threads=1)
    four_obs, four_meta = replay(threads=4)
    assert one_meta == four_meta, "metadata depends on the seed, not the scheduling"
    for one, four in zip(one_obs, four_obs):
        assert np.array_equal(one, four), "worker count changed an observation"


def test_a_different_seed_produces_a_different_run():
    """Compared over the whole replay, not at frame zero.

    The window is player-centred and the player always starts at rest in the
    middle of it, so at a small enemy count two seeds can genuinely agree on
    frame zero while placing every enemy differently in the world. The seeds
    and the frames that follow are where the difference has to show.
    """
    first_obs, first_meta = replay(threads=1, seed=11)
    other_obs, other_meta = replay(threads=1, seed=12)

    assert first_meta[0] != other_meta[0], "different roots are different seeds"
    assert any(
        not np.array_equal(one, two) for one, two in zip(first_obs, other_obs)
    ), "two seeds produced an identical run"


# --- terminal observations, both endings ----------------------------------


def test_a_timeout_carries_a_terminal_observation_and_a_reset_seed():
    with RoyaleVecEnv(**SMALL, seed=7, rewards=Rewards()) as env:
        for _ in range(SMALL["max_frames"]):
            _, _, dones, infos = env.step(np.zeros(env.num_envs, dtype=np.int64))
        assert all(dones), "the budget ends every episode together"
        for info in infos:
            assert info["TimeLimit.truncated"] is True
            assert "terminal_observation" in info
            assert info["frames"] == SMALL["max_frames"]
            assert info["reset_seed"] is not None


def test_a_death_terminates_without_truncating():
    """A full arena and an idle player, run until something catches him."""
    config = dict(envs=2, enemies=100, max_frames=100_000, threads=2, seed=31)
    deadline = time.monotonic() + 240
    with RoyaleVecEnv(**config, rewards=Rewards()) as env:
        died = None
        while died is None and time.monotonic() < deadline:
            _, _, _, infos = env.step(np.zeros(env.num_envs, dtype=np.int64))
            for index, info in enumerate(infos):
                if info.get("run_over") and not info.get("TimeLimit.truncated", False):
                    died = (index, info)
                    break
    assert died is not None, "an idle player is caught eventually"
    index, info = died
    assert info["TimeLimit.truncated"] is False, "a death is not a timeout"
    assert "terminal_observation" in info
    assert info["training_events"]["deaths"] == 1
    assert info["episode_summary"]["died"] is True


def test_a_retained_observation_is_not_rewritten_by_later_steps():
    """The aliasing bug that would look like a training problem."""
    with RoyaleVecEnv(**SMALL, seed=7, rewards=Rewards()) as env:
        observations = env.reset()
        kept = observations.copy()
        first = observations
        for _ in range(4):
            env.step(np.zeros(env.num_envs, dtype=np.int64))
        assert np.array_equal(first, kept), "the array handed out was rewritten"


# --- a real PPO update ----------------------------------------------------


def test_one_real_ppo_update_changes_a_parameter():
    """The plan's shape exactly: two envs, n_steps=16, one epoch."""
    config = SessionConfig(
        **SMALL, seed=7, n_steps=16, n_epochs=1, total_timesteps=32, device="cpu"
    )
    assert minibatch_size(config.rollout_samples(), config.minibatch_cap) == 32

    metrics = MetricsCollector()
    with session(config) as env:
        model = build_model(config, env)
        before = {
            name: parameter.detach().clone()
            for name, parameter in model.policy.named_parameters()
        }
        model.learn(total_timesteps=config.total_timesteps, callback=metrics)
        after = dict(model.policy.named_parameters())

        changed = [
            name
            for name, original in before.items()
            if not np.allclose(
                original.numpy(), after[name].detach().numpy(), atol=0, rtol=0
            )
        ]
        assert changed, "no trainable parameter moved; the update did nothing"
        print(f"\n{len(changed)} of {len(before)} parameter tensors changed")

    snapshot = metrics.snapshot()
    assert snapshot.timesteps >= 32
    for key in ("value_loss", "policy_loss", "entropy_loss", "approx_kl"):
        value = getattr(snapshot, key)
        assert value is not None and np.isfinite(value), f"{key} is {value}"


def test_save_reload_and_predict_again(tmp_path):
    import torch
    from stable_baselines3 import PPO

    config = SessionConfig(
        **SMALL,
        seed=7,
        n_steps=16,
        n_epochs=1,
        total_timesteps=32,
        device="cpu",
        checkpoint_dir=str(tmp_path),
    )
    with session(config) as env:
        model = build_model(config, env)
        model.learn(total_timesteps=config.total_timesteps)
        observations = env.reset()
        model.policy.set_training_mode(False)
        with torch.no_grad():
            values_before = model.policy.predict_values(
                torch.as_tensor(observations)
            )
        path = tmp_path / "acceptance.zip"
        model.save(path)

    # A fresh process-level session, and a reload that must be refused or
    # accepted on the layout rather than on faith.
    with session(config) as env:
        require_loadable(path, env.layout)
        reloaded = PPO.load(
            path,
            env=env,
            device="cpu",
            **ARCHITECTURES[ROYALE_ARCHITECTURE].resume_kwargs(),
        )
        reloaded.policy.set_training_mode(False)
        with torch.no_grad():
            values_after = reloaded.policy.predict_values(
                torch.as_tensor(observations)
            )
        assert torch.allclose(values_before, values_after, atol=1e-6)

        # And it can still step.
        actions, _ = reloaded.predict(observations, deterministic=True)
        stepped, rewards, dones, _ = env.step(actions)
        assert stepped.shape == observations.shape
        assert np.isfinite(rewards).all()


def test_a_checkpoint_from_a_different_hold_is_refused(tmp_path):
    """Changing the prediction hold changes the layout, not the shape."""
    config = SessionConfig(
        **SMALL, seed=7, n_steps=16, n_epochs=1, total_timesteps=16,
        device="cpu", checkpoint_dir=str(tmp_path),
    )
    with session(config) as env:
        model = build_model(config, env)
        path = tmp_path / "held.zip"
        model.save(path)
        trained_layout = env.layout

    changed = SessionConfig(**{**config.__dict__, "hold_frames": config.hold_frames + 1})
    with session(changed) as env:
        assert env.layout != trained_layout, "the hold is part of the layout"
        with pytest.raises(ProtocolError, match="hold_frames|not the one this policy"):
            build_model(changed, env, resume=path)


def test_changing_the_arena_does_not_invalidate_a_checkpoint(tmp_path):
    """Enemy count and frame budget restart envs; they are not model shape."""
    first = SessionConfig(
        **SMALL, seed=7, n_steps=16, n_epochs=1, total_timesteps=16,
        device="cpu", checkpoint_dir=str(tmp_path),
    )
    with session(first) as env:
        model = build_model(first, env)
        path = tmp_path / "arena.zip"
        model.save(path)

    second = SessionConfig(
        envs=2, enemies=40, max_frames=32, threads=1, seed=9,
        n_steps=16, n_epochs=1, total_timesteps=16, device="cpu",
        checkpoint_dir=str(tmp_path),
    )
    with session(second) as env:
        resumed = build_model(second, env, resume=path)
        assert resumed.policy.features_extractor.layout == env.layout


# --- the dashboard's own duration reporting -------------------------------


def test_the_dashboard_reports_survival_in_seconds():
    """A budget of N frames must chart as N/60 seconds, from a real gym."""
    frames = 30
    with RoyaleVecEnv(
        envs=2, enemies=12, max_frames=frames, threads=1, seed=7, rewards=Rewards()
    ) as env:
        summaries = []
        for _ in range(frames):
            _, _, _, infos = env.step(np.zeros(env.num_envs, dtype=np.int64))
            summaries += [i["episode_summary"] for i in infos if "episode_summary" in i]
    assert summaries, "an episode finished"
    for summary in summaries:
        assert summary["frames"] == frames
        assert summary["seconds"] == pytest.approx(frames / 60.0)


# --- no DodgeAI ------------------------------------------------------------


def test_nothing_imports_dodgeai():
    """The trainer must run with no DodgeAI checkout present."""
    import sys

    leaked = [name for name in sys.modules if name == "dodge" or name.startswith("dodge.")]
    assert not leaked, f"DodgeAI modules are loaded: {leaked}"


def test_the_package_lives_entirely_in_this_checkout():
    import dodge_royale

    root = Path(dodge_royale.__file__).resolve().parent.parent.parent
    assert (root / "Cargo.toml").exists(), "the trainer sits inside the Rust checkout"
    assert (root / "tests" / "fixtures" / "gym-v1" / "manifest.json").exists()
