"""`RoyaleVecEnv` against a fake client, and against the committed fixtures.

A fake gym rather than a real one, because the cases that matter are the ones
a real arena will not produce on demand: a death and a timeout on the same
frame, an env finishing while its neighbour carries on, a 60-frame episode
ending exactly on the second.
"""

from __future__ import annotations

import io
import json

import numpy as np
import pytest
from gymnasium import spaces

from conftest import FIXTURES
from dodge_royale.protocol import (
    GymError,
    Layout,
    ResetBatch,
    StepBatch,
    Transition,
    decode_step,
    find_binary,
)
from dodge_royale.rewards import Rewards
from dodge_royale.telemetry import FRAMES_PER_SECOND, EpisodeLog
from dodge_royale.vec_env import RoyaleVecEnv, observation_space


def _gym_available() -> bool:
    try:
        find_binary()
    except GymError:
        return False
    return True


@pytest.fixture(scope="session")
def layout(manifest) -> Layout:
    return Layout.from_json(manifest["layout"])


class FakeGym:
    """A `GymClient` stand-in that replays whatever a test hands it.

    Narrow on purpose: the protocol is already tested against real bytes, so
    this exists to let the environment be driven through endings a live arena
    would take thousands of frames to reach.
    """

    def __init__(self, layout: Layout, envs: int = 2, width: int | None = None) -> None:
        self.layout = layout
        self.envs = envs
        self._width = width if width is not None else layout.observation_values
        self.initial = ResetBatch(
            seeds=tuple(100 + index for index in range(envs)),
            observations=np.zeros((envs, self._width), dtype=np.float32),
        )
        self.queue: list[StepBatch] = []
        self.sent: list[list[int]] = []
        self.reset_seeds: list[int | None] = []
        self.closed = False
        self._episode = 0

    def step(self, actions):
        self.sent.append(list(actions))
        if not self.queue:
            raise AssertionError("the fake gym was stepped more often than it was loaded")
        return self.queue.pop(0)

    def reset(self, seed=None):
        self.reset_seeds.append(seed)
        self._episode += 1
        base = 1000 * self._episode if seed is None else seed
        return ResetBatch(
            seeds=tuple(base + index for index in range(self.envs)),
            observations=np.full(
                (self.envs, self._width), float(self._episode), dtype=np.float32
            ),
        )

    def close(self):
        self.closed = True


def batch_of(
    transitions: list[Transition],
    *,
    width: int,
    fill: float = 0.0,
    terminal: dict[int, float] | None = None,
) -> StepBatch:
    """A step response whose observations say which episode they came from."""
    envs = len(transitions)
    return StepBatch(
        transitions=tuple(transitions),
        observations=np.full((envs, width), fill, dtype=np.float32),
        terminal={
            env: np.full(width, value, dtype=np.float32)
            for env, value in (terminal or {}).items()
        },
    )


def running(frame: int = 1, deaths: int = 0, seed: int = 100) -> Transition:
    return Transition(
        frame=frame,
        terminated=False,
        truncated=False,
        enemy_deaths=deaths,
        episode_seed=seed,
        reset_seed=None,
    )


def finished(
    *, frame: int, terminated: bool, truncated: bool, deaths: int = 0, seed: int = 100
) -> Transition:
    return Transition(
        frame=frame,
        terminated=terminated,
        truncated=truncated,
        enemy_deaths=deaths,
        episode_seed=seed,
        reset_seed=seed + 5000,
    )


def make_env(layout: Layout, *, envs: int = 2, width: int = 8, rewards=None):
    gym = FakeGym(layout, envs=envs, width=width)
    env = RoyaleVecEnv(client=gym, rewards=rewards or Rewards())
    # The fake's observations are `width` wide for legibility; the space is
    # built from the real layout, which is only consulted for its sections.
    return env, gym


# --- done, truncation, and which observation is which --------------------


def test_done_is_terminated_or_truncated(layout):
    env, gym = make_env(layout)
    cases = [
        (running(), False),
        (finished(frame=9, terminated=True, truncated=False), True),
        (finished(frame=9, terminated=False, truncated=True), True),
        (finished(frame=9, terminated=True, truncated=True), True),
    ]
    for transition, expected in cases:
        gym.queue.append(
            batch_of([transition, running()], width=8, terminal={0: 7.0} if expected else None)
        )
        env.step_async(np.array([0, 0]))
        _, _, dones, _ = env.step_wait()
        assert bool(dones[0]) is expected, transition
    env.close()


def test_time_limit_truncated_is_truncation_without_death(layout):
    """The flag a value function acts on, in all four combinations.

    Both flags can be set on the same frame: a player can die as the budget
    expires. A learner that bootstrapped from that would be valuing a state
    the player is dead in, so the death has to win.
    """
    env, gym = make_env(layout)
    cases = [
        (True, False, False),  # died: no bootstrap
        (False, True, True),  # timed out: bootstrap
        (True, True, False),  # died as the clock ran out: still no bootstrap
    ]
    for terminated, truncated, expected in cases:
        gym.queue.append(
            batch_of(
                [finished(frame=9, terminated=terminated, truncated=truncated), running()],
                width=8,
                terminal={0: 7.0},
            )
        )
        env.step_async(np.array([0, 0]))
        _, _, _, infos = env.step_wait()
        assert infos[0]["TimeLimit.truncated"] is expected, (terminated, truncated)
    env.close()


def test_a_live_transition_carries_no_terminal_observation(layout):
    """SB3 reads that key as "this episode ended"; a stale one ends a live one."""
    env, gym = make_env(layout)
    gym.queue.append(batch_of([running(), running()], width=8, terminal={}))
    env.step_async(np.array([0, 0]))
    _, _, dones, infos = env.step_wait()
    for info, done in zip(infos, dones):
        assert not done
        assert "terminal_observation" not in info
        assert "TimeLimit.truncated" not in info
        assert "episode" not in info
    env.close()


def test_the_batch_observation_is_the_reset_episode_and_the_info_is_the_finished_one(layout):
    """The auto-reset split, which is the whole point of this adapter.

    `obs` is what the agent acts on next; `terminal_observation` is what the
    critic bootstraps from. Swapping them trains the policy on a world it is
    not in.
    """
    env, gym = make_env(layout)
    gym.queue.append(
        batch_of(
            [finished(frame=60, terminated=False, truncated=True), running()],
            width=8,
            fill=2.0,  # the replacement episode
            terminal={0: 9.0},  # the episode that ended
        )
    )
    env.step_async(np.array([0, 0]))
    obs, _, dones, infos = env.step_wait()

    assert dones[0]
    assert np.all(obs[0] == 2.0), "the batch observation is the new episode"
    assert np.all(infos[0]["terminal_observation"] == 9.0), "the info is the old one"
    assert infos[0]["frames"] == 60, "the metadata describes the episode that ended"
    assert infos[0]["run_frames"] == 60
    env.close()


def test_a_finished_episodes_metadata_is_not_the_replacements(layout):
    env, gym = make_env(layout)
    gym.queue.append(
        batch_of(
            [finished(frame=137, terminated=True, truncated=False, deaths=3), running()],
            width=8,
            terminal={0: 1.0},
        )
    )
    env.step_async(np.array([0, 0]))
    _, _, _, infos = env.step_wait()
    assert infos[0]["frames"] == 137, "not the new episode's zero"
    assert infos[0]["run_over"] is True
    assert infos[0]["episode_summary"]["died"] is True
    assert infos[0]["episode"]["l"] == 137
    env.close()


# --- reward --------------------------------------------------------------


def test_survival_is_paid_once_per_surviving_step():
    rewards = Rewards(survival_per_frame=0.02, death_penalty=2.0, uncontrolled_score_weight=0.05)
    assert rewards.for_step(terminated=False, truncated=False, enemy_deaths=0) == pytest.approx(
        0.02
    )


def test_a_death_costs_the_penalty_and_earns_no_survival():
    rewards = Rewards(survival_per_frame=0.02, death_penalty=2.0, uncontrolled_score_weight=0.05)
    assert rewards.for_step(terminated=True, truncated=False, enemy_deaths=0) == pytest.approx(
        -2.0
    )


def test_a_timeout_is_not_a_death():
    """The budget running out is not a failure; the player was fine."""
    rewards = Rewards(survival_per_frame=0.02, death_penalty=2.0, uncontrolled_score_weight=0.05)
    assert rewards.for_step(terminated=False, truncated=True, enemy_deaths=0) == pytest.approx(
        0.02
    )


def test_a_death_on_the_frame_the_budget_expires_is_still_a_death():
    rewards = Rewards(survival_per_frame=0.02, death_penalty=2.0, uncontrolled_score_weight=0.05)
    assert rewards.for_step(terminated=True, truncated=True, enemy_deaths=0) == pytest.approx(
        -2.0
    )


def test_enemy_kills_earn_the_uncontrolled_share():
    rewards = Rewards(survival_per_frame=0.02, death_penalty=2.0, uncontrolled_score_weight=0.05)
    got = rewards.for_step(terminated=False, truncated=False, enemy_deaths=4)
    assert got == pytest.approx(0.02 + 0.05 * 0.5 * 4)


def test_reward_is_counted_once_per_step_and_never_for_a_reset(layout):
    """A reset produces no transition, so it can earn nothing.

    Twelve surviving steps should be worth twelve survival payments, no matter
    how many resets happen around them.
    """
    rewards = Rewards(survival_per_frame=0.5, death_penalty=0.0, uncontrolled_score_weight=0.0)
    env, gym = make_env(layout, rewards=rewards)

    total = 0.0
    for frame in range(1, 13):
        gym.queue.append(batch_of([running(frame=frame), running(frame=frame)], width=8))
        env.step_async(np.array([0, 0]))
        _, step_rewards, _, _ = env.step_wait()
        total += float(step_rewards[0])
        env.reset()  # between every step; a reset must add nothing

    assert total == pytest.approx(12 * 0.5)
    env.close()


def test_unknown_reward_keys_are_ignored_rather_than_refused(tmp_path):
    """Reward files are shared with schemes that have controls Royale lacks."""
    path = tmp_path / "rewards.json"
    path.write_text(
        json.dumps(
            {
                "survival_per_frame": 0.03,
                "death_penalty": 1.5,
                "uncontrolled_score_weight": 0.1,
                "edge_penalty": 9.0,
                "score_weight": 9.0,
            }
        )
    )
    rewards = Rewards.load(path)
    assert rewards.survival_per_frame == pytest.approx(0.03)
    assert not hasattr(rewards, "edge_penalty")


# --- the mixed batch, from the committed fixture -------------------------


def test_a_neighbours_reset_does_not_disturb_a_surviving_env(manifest, layout):
    """The real mixed batch: one env dies, the other is mid-episode.

    Everything about the survivor -- its observation, its death count, its
    episode tracking, its seed -- has to be untouched by what happened next
    door. A tracker or a seed indexed one place off would show up here and
    almost nowhere else.
    """
    width = manifest["observation_values"]
    raw = (FIXTURES / "step-death.bin").read_bytes()
    real = decode_step(io.BytesIO(raw), 2, width)

    dead = [i for i, t in enumerate(real.transitions) if t.terminated]
    alive = [i for i, t in enumerate(real.transitions) if not t.done]
    assert len(dead) == 1 and len(alive) == 1, "the fixture is the mixed batch"
    dead_env, live_env = dead[0], alive[0]

    gym = FakeGym(layout, envs=2, width=width)
    env = RoyaleVecEnv(client=gym, rewards=Rewards())
    seeds_before = env.episode_seeds

    # A few ordinary frames first, so the survivor has history to preserve.
    for frame in (316, 317):
        gym.queue.append(
            batch_of(
                [running(frame=frame, deaths=1), running(frame=frame, deaths=1)],
                width=width,
            )
        )
        env.step_async(np.array([0, 0]))
        env.step_wait()

    gym.queue.append(real)
    env.step_async(np.array([0, 0]))
    obs, rewards, dones, infos = env.step_wait()

    assert dones[dead_env] and not dones[live_env]

    # The survivor.
    live_info = infos[live_env]
    assert "terminal_observation" not in live_info
    assert "TimeLimit.truncated" not in live_info
    assert "run_over" not in live_info
    assert live_info["frames"] == real.transitions[live_env].frame
    assert live_info["enemy_deaths"] == real.transitions[live_env].enemy_deaths
    assert live_info["training_events"]["deaths"] == 0
    assert live_info["training_events"]["survival_frames"] == 1
    assert np.array_equal(obs[live_env], real.observations[live_env])
    assert env.episode_seeds[live_env] == seeds_before[live_env], "its stream is untouched"

    # The neighbour that died.
    dead_info = infos[dead_env]
    assert dead_info["run_over"] is True
    assert dead_info["TimeLimit.truncated"] is False, "a death, not a timeout"
    assert np.array_equal(dead_info["terminal_observation"], real.terminal[dead_env])
    assert dead_info["training_events"]["deaths"] == 1
    assert dead_info["training_events"]["survival_frames"] == 0
    assert env.episode_seeds[dead_env] == real.transitions[dead_env].reset_seed

    # And the two observations are genuinely different worlds.
    assert not np.array_equal(obs[dead_env], dead_info["terminal_observation"])
    env.close()


def test_the_survivors_episode_keeps_accumulating_across_a_neighbours_death(layout):
    """A reset must clear one tracker, not the batch."""
    rewards = Rewards(survival_per_frame=1.0, death_penalty=0.0, uncontrolled_score_weight=0.0)
    env, gym = make_env(layout, rewards=rewards)

    for frame in (1, 2, 3):
        gym.queue.append(batch_of([running(frame=frame), running(frame=frame)], width=8))
        env.step_async(np.array([0, 0]))
        env.step_wait()

    gym.queue.append(
        batch_of(
            [finished(frame=4, terminated=True, truncated=False), running(frame=4)],
            width=8,
            terminal={0: 3.0},
        )
    )
    env.step_async(np.array([0, 0]))
    _, _, _, infos = env.step_wait()
    assert infos[0]["episode"]["r"] == pytest.approx(3.0), (
        "three surviving frames; the fatal fourth earns no survival"
    )

    # Env 1 has run five frames by now and must still be counting from one.
    # Its fifth is a timeout, which is not a death and does earn its frame.
    gym.queue.append(
        batch_of(
            [running(frame=1), finished(frame=5, terminated=False, truncated=True)],
            width=8,
            terminal={1: 3.0},
        )
    )
    env.step_async(np.array([0, 0]))
    _, _, _, infos = env.step_wait()
    assert infos[1]["frames"] == 5, "the survivor's episode was never reset"
    assert infos[1]["episode"]["r"] == pytest.approx(5.0)
    env.close()


# --- a synthetic episode on the dashboard --------------------------------


def test_a_sixty_frame_episode_charts_as_one_second(layout):
    """The unit conversion the whole dashboard rests on.

    One action per 60 Hz frame, so sixty frames is a second. Getting this wrong
    would misreport every survival chart by a constant factor, which is exactly
    the kind of error that looks like a plausible result.
    """
    rewards = Rewards(survival_per_frame=0.02, death_penalty=2.0, uncontrolled_score_weight=0.05)
    env, gym = make_env(layout, rewards=rewards)
    log = EpisodeLog()

    kills_at = {10: 2, 25: 1, 44: 3}
    for frame in range(1, 61):
        deaths = kills_at.get(frame, 0)
        last = frame == 60
        transition = (
            finished(frame=frame, terminated=True, truncated=False, deaths=deaths)
            if last
            else running(frame=frame, deaths=deaths)
        )
        gym.queue.append(
            batch_of(
                [transition, running(frame=frame)],
                width=8,
                terminal={0: 5.0} if last else None,
            )
        )
        env.step_async(np.array([0, 0]))
        _, _, _, infos = env.step_wait()
        if "episode_summary" in infos[0]:
            log.record(infos[0]["episode_summary"])

    assert len(log.episodes) == 1
    episode = log.episodes[0]
    assert episode["frames"] == 60
    assert episode["run_frames"] == 60
    assert episode["seconds"] == pytest.approx(1.0)
    assert log.total_seconds == pytest.approx(1.0)
    assert log.deaths == 1
    assert log.enemies_destroyed == sum(kills_at.values()) == 6

    # 59 surviving frames, one fatal, six kills at the uncontrolled share.
    expected = 59 * 0.02 - 2.0 + 0.05 * 0.5 * 6
    assert episode["return"] == pytest.approx(expected)
    assert FRAMES_PER_SECOND == 60.0
    env.close()


def test_per_step_events_are_not_running_totals(layout):
    """A dashboard sums what it is given; totals would be summed again."""
    env, gym = make_env(layout)
    for frame in range(1, 5):
        gym.queue.append(
            batch_of([running(frame=frame, deaths=2), running(frame=frame)], width=8)
        )
        env.step_async(np.array([0, 0]))
        _, _, _, infos = env.step_wait()
        assert infos[0]["training_events"]["enemies_destroyed"] == 2, "this step, not so far"
        assert infos[0]["training_events"]["survival_frames"] == 1
        assert infos[0]["training_events"]["lives_spent"] == 0
    env.close()


# --- the VecEnv contract -------------------------------------------------


def test_the_spaces_come_from_the_layout(manifest, layout):
    space = observation_space(layout)
    assert space.dtype == np.float32
    assert space.shape == (layout.observation_values,)

    paths = layout.path_section
    assert np.all(np.isinf(space.low[paths.offset : paths.stop])), "paths are not clipped"
    assert np.all(np.isinf(space.high[paths.offset : paths.stop]))
    grid = layout.grid_section
    assert np.all(space.low[grid.offset : grid.stop] == -1.0)
    assert np.all(space.high[grid.offset : grid.stop] == 1.0)


def test_the_action_space_is_the_nine_actions(layout):
    env, _ = make_env(layout)
    assert isinstance(env.action_space, spaces.Discrete)
    assert env.action_space.n == len(layout.actions) == 9
    env.close()


def test_get_attr_answers_render_mode_and_refuses_anything_else(layout):
    """VecEnv.__init__ probes render_mode and catches only AttributeError."""
    env, _ = make_env(layout, envs=3)
    assert env.get_attr("render_mode") == [None, None, None]
    assert env.render_mode is None
    with pytest.raises(AttributeError):
        env.get_attr("some_attribute_a_wrapper_invented")
    env.close()


def test_get_attr_honours_indices(layout):
    env, _ = make_env(layout, envs=4)
    assert len(env.get_attr("render_mode")) == 4
    assert len(env.get_attr("render_mode", indices=[0, 2])) == 2
    assert len(env.get_attr("render_mode", indices=1)) == 1
    with pytest.raises(IndexError):
        env.get_attr("render_mode", indices=[0, 9])
    env.close()


def test_set_attr_and_env_method_fail_clearly(layout):
    env, _ = make_env(layout)
    with pytest.raises(AttributeError, match="cannot be set"):
        env.set_attr("enemy_count", 5)
    with pytest.raises(AttributeError, match="no 'render' to call"):
        env.env_method("render")
    env.close()


def test_nothing_is_wrapped_including_the_time_limit(layout):
    """The budget lives in the simulation, so there is no wrapper to find."""
    from gymnasium.wrappers import TimeLimit

    env, _ = make_env(layout, envs=3)
    assert env.env_is_wrapped(TimeLimit) == [False, False, False]
    assert env.env_is_wrapped(TimeLimit, indices=[1]) == [False]
    env.close()


def test_seeding_is_deferred_to_the_next_reset(layout):
    """SB3 seeds before resetting and expects the seed to apply there."""
    env, gym = make_env(layout)
    assert gym.reset_seeds == []

    env.seed(1234)
    assert gym.reset_seeds == [], "seed() must not reset on its own"

    env.reset()
    assert gym.reset_seeds == [1234]

    # And it is spent: a second reset continues the stream rather than
    # repeating the seeded episode.
    env.reset()
    assert gym.reset_seeds == [1234, None]
    env.close()


def test_reset_infos_carry_the_new_episode_seeds(layout):
    env, gym = make_env(layout)
    assert [info["episode_seed"] for info in env.reset_infos] == list(gym.initial.seeds)
    env.reset()
    assert all("episode_seed" in info for info in env.reset_infos)
    env.close()


def test_reset_returns_frame_zero_and_clears_episode_tracking(layout):
    rewards = Rewards(survival_per_frame=1.0, death_penalty=0.0, uncontrolled_score_weight=0.0)
    env, gym = make_env(layout, rewards=rewards)
    for frame in (1, 2, 3):
        gym.queue.append(batch_of([running(frame=frame), running(frame=frame)], width=8))
        env.step_async(np.array([0, 0]))
        env.step_wait()

    env.reset()
    gym.queue.append(
        batch_of(
            [finished(frame=1, terminated=True, truncated=False), running(frame=1)],
            width=8,
            terminal={0: 1.0},
        )
    )
    env.step_async(np.array([0, 0]))
    _, _, _, infos = env.step_wait()
    assert infos[0]["episode"]["r"] == pytest.approx(0.0), "the pre-reset frames are gone"
    env.close()


def test_the_whole_action_batch_is_validated_before_anything_is_sent(layout):
    env, gym = make_env(layout)
    with pytest.raises(ValueError, match="must be one of"):
        env.step_async(np.array([0, 99]))
    assert gym.sent == [], "a refused batch must not reach the pipe"

    with pytest.raises(ValueError, match="got 1 actions"):
        env.step_async(np.array([0]))
    assert gym.sent == []
    env.close()


def test_actions_are_sent_as_the_protocols_bytes(layout):
    env, gym = make_env(layout)
    gym.queue.append(batch_of([running(), running()], width=8))
    env.step_async(np.array([3, 8]))
    env.step_wait()
    assert gym.sent == [[3, 8]]
    env.close()


def test_step_wait_without_step_async_is_an_error(layout):
    env, _ = make_env(layout)
    with pytest.raises(RuntimeError, match="without a step_async"):
        env.step_wait()
    env.close()


def test_close_is_idempotent_and_closes_the_client(layout):
    env, gym = make_env(layout)
    env.close()
    assert gym.closed
    env.close()
    env.close()


def test_the_context_manager_closes(layout):
    env, gym = make_env(layout)
    with env:
        pass
    assert gym.closed


def test_step_returns_arrays_of_the_shapes_sb3_expects(layout):
    env, gym = make_env(layout, envs=3, width=8)
    gym.queue.append(batch_of([running(), running(), running()], width=8))
    obs, rewards, dones, infos = env.step(np.array([0, 1, 2]))
    assert obs.shape == (3, 8)
    assert rewards.shape == (3,) and rewards.dtype == np.float32
    assert dones.shape == (3,) and dones.dtype == bool
    assert len(infos) == 3
    env.close()


# --- against real SB3 and a real gym --------------------------------------


def test_sb3_can_wrap_this_env(layout):
    """VecMonitor probes the env the way SB3's own machinery does."""
    from stable_baselines3.common.vec_env import VecMonitor

    env, gym = make_env(layout, envs=2, width=8)
    monitored = VecMonitor(env)
    gym.queue.append(
        batch_of(
            [finished(frame=7, terminated=True, truncated=False), running(frame=7)],
            width=8,
            terminal={0: 4.0},
        )
    )
    monitored.step_async(np.array([0, 0]))
    _, _, dones, infos = monitored.step_wait()
    assert dones[0]
    # VecMonitor writes its own `episode` record over ours, counting steps
    # since it began rather than the episode's own frames. That is its job, so
    # our authoritative length stays under its own keys where a wrapper cannot
    # overwrite it.
    assert "episode" in infos[0]
    assert infos[0]["frames"] == 7
    assert infos[0]["episode_summary"]["frames"] == 7
    monitored.close()
    assert gym.closed


def test_render_does_not_explode_when_there_is_nothing_to_render(layout):
    """`VecEnv.render` consults render_mode, which is None for a gym."""
    env, _ = make_env(layout)
    assert env.render_mode is None
    env.close()


live_gym = pytest.mark.skipif(
    not _gym_available(), reason="no gym binary; build with `cargo build --release --no-default-features`"
)


@pytest.mark.live
@live_gym
def test_a_live_env_runs_the_whole_lifecycle():
    env = RoyaleVecEnv(envs=2, seed=7, enemies=12, max_frames=4, threads=1)
    try:
        # Seeded from the start: an unseeded reset advances the stream, so a
        # baseline taken after one would not be frame zero to compare against.
        env.seed(7)
        first = env.reset()
        assert first.shape == (2, env.layout.observation_values)
        seeded_first = list(env.episode_seeds)
        for _ in range(4):
            env.step_async(np.array([2, 5]))
            obs, rewards, dones, infos = env.step_wait()
        assert all(dones), "a four-frame budget ends on the fourth step"
        for info in infos:
            assert info["TimeLimit.truncated"] is True
            assert "terminal_observation" in info
            assert info["frames"] == 4

        # The same seed reproduces frame zero exactly, episode counters and all.
        env.seed(7)
        again = env.reset()
        assert env.episode_seeds == seeded_first
        assert np.array_equal(again, first)
    finally:
        env.close()


@pytest.mark.live
@live_gym
def test_a_live_env_survives_an_sb3_rollout():
    """One real PPO rollout, against the real gym.

    Not a training check: this is the contract check that only SB3 itself can
    make, and it is the cheapest way to find out that some corner of VecEnv is
    wrong before a long run does.
    """
    from stable_baselines3 import PPO

    env = RoyaleVecEnv(envs=2, seed=3, enemies=8, max_frames=32, threads=1)
    try:
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=8,
            batch_size=8,
            n_epochs=1,
            policy_kwargs={"net_arch": [16]},
            device="cpu",
        )
        model.learn(total_timesteps=16)
    finally:
        env.close()
