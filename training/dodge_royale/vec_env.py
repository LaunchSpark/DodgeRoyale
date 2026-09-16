"""`RoyaleVecEnv`: the gym batch, wearing SB3's `VecEnv` interface.

The gym is already a batch of arenas behind one pipe, so this is a thin
adapter, not a fan-out. **Do not wrap it in `SubprocVecEnv` or `DummyVecEnv`**:
that would launch N gyms of N envs each and multiply the work by N.

Everything hard here comes from one fact: the gym **auto-resets**. When an
episode ends, the same message carries the finished episode's last observation
*and* the replacement episode's first one, and the two describe different
worlds. Which of them goes where, and which episode the metadata is about,
decides whether a learner trains on history or on fiction:

* ``obs`` is the reset episode. That is what the agent acts on next.
* ``infos[i]["terminal_observation"]`` is the finished episode. That is what a
  value function bootstraps from when the ending was a timeout.
* every other number in ``infos[i]`` describes the **finished** episode, not
  the zero-length one that replaced it.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn

from .protocol import GymClient, Layout, StepBatch, Transition
from .rewards import Rewards
from .telemetry import EpisodeTracker, TrainingEvents

__all__ = ["RoyaleVecEnv", "observation_space", "action_space"]

#: Attributes a caller may read per env. Anything else raises AttributeError,
#: because SB3 probes for attributes and only catches that.
_READABLE: frozenset[str] = frozenset(
    {"render_mode", "spec", "observation_space", "action_space", "reward_range"}
)


def observation_space(layout: Layout) -> spaces.Box:
    """Per-element bounds, from the layout the gym announced.

    Not a flat +/-1 box: the grid and the player's velocity are clipped into
    [-1, 1] by the encoder, but paths are deliberately not clipped -- a path
    may leave the window, and the policy samples it with border padding. A box
    that claimed otherwise would misdescribe the data, and anything that
    normalises against these bounds would be working from a false range.
    """
    low = np.full(layout.observation_values, -1.0, dtype=np.float32)
    high = np.full(layout.observation_values, 1.0, dtype=np.float32)
    paths = layout.path_section
    low[paths.offset : paths.stop] = -np.inf
    high[paths.offset : paths.stop] = np.inf
    return spaces.Box(low=low, high=high, dtype=np.float32)


def action_space(layout: Layout) -> spaces.Discrete:
    return spaces.Discrete(len(layout.actions))


class RoyaleVecEnv(VecEnv):
    """One `dodge-royale gym` process, as an SB3 vectorised environment."""

    def __init__(
        self,
        *,
        envs: int = 8,
        seed: int = 0,
        enemies: int = 100,
        max_frames: int = 3600,
        hold_frames: int = 24,
        threads: int = 2,
        binary: str | None = None,
        rewards: Rewards | None = None,
        client: GymClient | None = None,
    ) -> None:
        self._client = client or GymClient(
            envs=envs,
            seed=seed,
            enemies=enemies,
            max_frames=max_frames,
            hold_frames=hold_frames,
            threads=threads,
            binary=binary,
        )
        self.layout = self._client.layout
        self.rewards = rewards or Rewards.load()
        self._closed = False

        # Frame zero arrived with the handshake, so the first reset has
        # something to return without asking the gym for an episode it is
        # already running.
        self._observations = self._client.initial.observations
        self._seeds_now = list(self._client.initial.seeds)
        self._trackers = [EpisodeTracker() for _ in range(self._client.envs)]
        self._pending: np.ndarray | None = None
        #: A seed asked for before the next reset, applied there and cleared.
        self._deferred_seed: int | None = None

        # VecEnv.__init__ probes get_attr("render_mode"), so everything it
        # touches has to exist before this call, not after it.
        super().__init__(
            num_envs=self._client.envs,
            observation_space=observation_space(self.layout),
            action_space=action_space(self.layout),
        )
        self.reset_infos = [{"episode_seed": seed} for seed in self._seeds_now]

    # --- the batch ------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Restart every env, applying any seed `seed()` asked for.

        A seeded reset reproduces frame zero exactly; an unseeded one advances
        each env's episode counter and continues the stream from the same root.
        """
        batch = self._client.reset(self._deferred_seed)
        self._deferred_seed = None
        self._observations = batch.observations
        self._seeds_now = list(batch.seeds)
        for tracker in self._trackers:
            tracker.reset()
        self._pending = None
        self.reset_infos = [{"episode_seed": seed} for seed in batch.seeds]
        return self._observations

    def step_async(self, actions: np.ndarray) -> None:
        """Hold the actions. The gym steps synchronously, in `step_wait`.

        Validating here rather than there keeps a bad batch from being half
        sent: the gym refuses a malformed STEP by ending the session, so an
        action out of range must never reach the pipe.
        """
        chosen = np.asarray(actions).reshape(-1)
        if chosen.size != self.num_envs:
            raise ValueError(
                f"this env has {self.num_envs} envs, got {chosen.size} actions"
            )
        count = len(self.layout.actions)
        if not np.all((chosen >= 0) & (chosen < count)):
            raise ValueError(f"every action must be one of the {count} actions: {chosen}")
        self._pending = chosen.astype(np.uint8, copy=True)

    def step_wait(self) -> VecEnvStepReturn:
        if self._pending is None:
            raise RuntimeError("step_wait() without a step_async()")
        actions, self._pending = self._pending, None
        batch = self._client.step(actions.tolist())
        return self._interpret(batch)

    def _interpret(self, batch: StepBatch) -> VecEnvStepReturn:
        """Turn one STEP response into what SB3 expects of a vector env."""
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos: list[dict[str, Any]] = []

        for index, transition in enumerate(batch.transitions):
            reward = self.rewards.for_step(
                terminated=transition.terminated,
                truncated=transition.truncated,
                enemy_deaths=transition.enemy_deaths,
            )
            rewards[index] = reward
            dones[index] = transition.done

            tracker = self._trackers[index]
            tracker.record(
                enemy_deaths=transition.enemy_deaths,
                reward=reward,
                frame=transition.frame,
            )
            infos.append(self._info(index, transition, tracker, batch))

        self._observations = batch.observations
        # Reported seeds follow the env: only an env that finished has a new
        # episode, and its neighbour's stream is untouched.
        for index, transition in enumerate(batch.transitions):
            if transition.reset_seed is not None:
                self._seeds_now[index] = transition.reset_seed
        return self._observations, rewards, dones, infos

    def _info(
        self,
        index: int,
        transition: Transition,
        tracker: EpisodeTracker,
        batch: StepBatch,
    ) -> dict[str, Any]:
        """One env's info for one step.

        Everything here describes the transition's own episode. For an env that
        just finished, that is the episode in the terminal section, not the one
        `observations` now holds.
        """
        events = TrainingEvents(
            # A frame the player did not survive earns no survival frame, which
            # is what keeps a death from charting as a frame of life.
            survival_frames=0 if transition.terminated else 1,
            deaths=1 if transition.terminated else 0,
            lives_spent=0,
            enemies_destroyed=transition.enemy_deaths,
        )
        info: dict[str, Any] = {
            "episode_seed": transition.episode_seed,
            "frames": transition.frame,
            "run_frames": transition.frame,
            "enemy_deaths": transition.enemy_deaths,
            "training_events": events.as_dict(),
        }

        if not transition.done:
            # A live transition must never carry a terminal observation: SB3
            # reads that key as "this episode ended".
            return info

        # SB3's contract: truncation alone means the state was still alive, so
        # a value function should bootstrap from it. A death means there is no
        # future to bootstrap towards, and both flags can be set on the same
        # frame when the player dies as the budget expires -- then the death
        # wins.
        info["TimeLimit.truncated"] = transition.time_limited
        terminal = batch.terminal.get(index)
        if terminal is not None:
            info["terminal_observation"] = terminal
        info["run_over"] = True
        info["episode"] = {
            # The keys SB3's Monitor uses, so standard logging keeps working.
            "r": tracker.reward,
            "l": transition.frame,
            "t": transition.frame / 60.0,
        }
        info["episode_summary"] = tracker.summary(terminated=transition.terminated)
        if transition.reset_seed is not None:
            info["reset_seed"] = transition.reset_seed
        # The arena is already running the replacement, so this env's counters
        # start again here. Only this env's: its neighbours are mid-episode.
        tracker.reset()
        return info

    # --- the rest of the VecEnv contract --------------------------------

    def seed(self, seed: int | None = None) -> Sequence[int | None]:
        """Ask for a seed, applied at the next `reset`.

        Deferred rather than immediate because SB3 calls `seed()` before
        `reset()` and expects the seed to take effect there. Resetting here
        would throw away an episode the caller had not finished with.
        """
        self._deferred_seed = seed
        return [seed] * self.num_envs

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._client.close()

    def get_attr(self, attr_name: str, indices: Any = None) -> list[Any]:
        """Read a per-env attribute.

        `VecEnv.__init__` probes `render_mode` and catches only
        `AttributeError`, so an unknown name must raise exactly that. Returning
        `None` instead would make every probe look like a defined attribute.
        """
        if attr_name not in _READABLE:
            raise AttributeError(
                f"a Royale env has no {attr_name!r}; the gym exposes "
                f"{sorted(_READABLE)}"
            )
        value = {
            "render_mode": None,
            "spec": None,
            "reward_range": (-float("inf"), float("inf")),
            "observation_space": self.observation_space,
            "action_space": self.action_space,
        }[attr_name]
        return [value for _ in self._indices(indices)]

    def set_attr(self, attr_name: str, value: Any, indices: Any = None) -> None:
        """Refuse: an arena's state lives in the gym, not in this object.

        Silently accepting would leave a caller believing it had changed
        something in the simulation.
        """
        raise AttributeError(
            f"a Royale env's {attr_name!r} cannot be set from Python; the arenas "
            "live in the gym process"
        )

    def env_method(
        self, method_name: str, *args: Any, indices: Any = None, **kwargs: Any
    ) -> list[Any]:
        raise AttributeError(
            f"a Royale env has no {method_name!r} to call; the gym speaks only the "
            "protocol's STEP, RESET and CLOSE"
        )

    def env_is_wrapped(self, wrapper_class: type, indices: Any = None) -> list[bool]:
        """Nothing is wrapped: the arenas are behind a pipe, not a Python stack.

        Including `TimeLimit`. The frame budget is enforced inside the
        simulation, so there is no wrapper to find, and claiming one would
        invite SB3 to look for state that does not exist.
        """
        return [False for _ in self._indices(indices)]

    def _indices(self, indices: Any) -> list[int]:
        """SB3's index convention: None is all of them, an int is one."""
        if indices is None:
            return list(range(self.num_envs))
        if isinstance(indices, int):
            chosen = [indices]
        elif isinstance(indices, Iterable):
            chosen = [int(index) for index in indices]
        else:
            raise TypeError(f"indices must be None, an int or an iterable, not {indices!r}")
        for index in chosen:
            if not 0 <= index < self.num_envs:
                raise IndexError(f"env {index} is out of range for {self.num_envs} envs")
        return chosen

    # --- extras ---------------------------------------------------------

    @property
    def episode_seeds(self) -> list[int]:
        """The seed each env's current episode is running on."""
        return list(self._seeds_now)

    def __enter__(self) -> "RoyaleVecEnv":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
