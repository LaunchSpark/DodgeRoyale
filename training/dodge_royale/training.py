"""The PPO session: build it, resume it, and always close the gym behind it.

One configuration and one env factory serve both entry points, so the CLI and
the dashboard cannot drift into training subtly different things.

Two rules run through this module.

**The gym is a child process, so every path closes it.** A failed model build,
a rejected checkpoint, a staged config restart, an interrupt: each one leaves a
`dodge-royale gym` holding pipes unless something closes it. That is what
`session` is for, and why nothing here returns a live env to a caller who has
not taken responsibility for it.

**The discounts come from the architecture table on both paths.** SB3 restores
whatever a checkpoint was saved with, so a resumed run would keep an old
`gae_lambda` -- silently, and for the rest of the run.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from .policies import ARCHITECTURES, ROYALE_ARCHITECTURE, Architecture, require_loadable
from .protocol import GymError, Layout
from .rewards import Rewards
from .telemetry import EpisodeLog
from .vec_env import RoyaleVecEnv

__all__ = [
    "DEFAULTS",
    "EpisodeRecorder",
    "SessionConfig",
    "build_model",
    "check_env",
    "make_env",
    "minibatch_size",
    "session",
]


@dataclass(frozen=True)
class SessionConfig:
    """Everything a run needs, in one place both entry points read.

    The defaults are the design's: eight envs, 1,024 rollout steps, 100
    enemies, a 3,600-frame budget, a 24-frame prediction hold, and two Rust
    worker threads.
    """

    envs: int = 8
    seed: int = 0
    enemies: int = 100
    max_frames: int = 3600
    #: Frames each predicted path holds its action before coasting. This is
    #: *not* an action repeat: the policy still decides every frame. Confusing
    #: the two would quarter the decision rate while leaving every shape intact.
    hold_frames: int = 24
    threads: int = 2

    n_steps: int = 1024
    #: Upper bound on a minibatch, independent of how large the rollout is.
    #: Rollout size scales with envs times steps; a minibatch that scaled with
    #: it would grow until a device ran out of memory. Measured at 8 envs and
    #: 1,024 steps on one machine, so it is a bound and not a guarantee: a
    #: smaller device can still exhaust memory inside it.
    minibatch_cap: int = 128
    n_epochs: int = 10
    learning_rate: float = 3e-4
    total_timesteps: int = 1_000_000
    device: str = "auto"

    binary: str | None = None
    rewards_path: str | None = None
    checkpoint_dir: str = "checkpoints"
    run_name: str = "royale"
    architecture: str = ROYALE_ARCHITECTURE

    def rollout_samples(self) -> int:
        return self.envs * self.n_steps

    def resolved_architecture(self) -> Architecture:
        try:
            return ARCHITECTURES[self.architecture]
        except KeyError:
            known = ", ".join(sorted(ARCHITECTURES))
            raise ValueError(
                f"unknown architecture {self.architecture!r}; this trainer builds {known}"
            ) from None

    def validate(self) -> None:
        """Refuse a configuration that cannot produce a run.

        Checked before a gym is launched, so an impossible setting costs a
        message rather than a child process and a confusing failure inside it.
        """
        for name in ("envs", "n_steps", "enemies", "max_frames", "threads", "n_epochs"):
            value = getattr(self, name)
            if value < 0 or (name != "enemies" and value <= 0):
                raise ValueError(f"{name} must be positive, got {value}")
        if self.hold_frames <= 0:
            raise ValueError(f"hold_frames must be positive, got {self.hold_frames}")
        if self.minibatch_cap <= 0:
            raise ValueError(f"minibatch_cap must be positive, got {self.minibatch_cap}")
        if self.total_timesteps <= 0:
            raise ValueError(f"total_timesteps must be positive, got {self.total_timesteps}")
        self.resolved_architecture()


#: The defaults, exposed so the CLI and dashboard show the same numbers.
DEFAULTS = SessionConfig()


DEFAULT_REWARDS = Rewards()


def minibatch_size(rollout: int, cap: int = DEFAULTS.minibatch_cap) -> int:
    """The largest minibatch that is at most `cap` and divides the rollout.

    Two requirements at once. PPO warns and silently truncates when the
    rollout does not divide evenly, so the size has to be a divisor. And it
    must not scale with the rollout, or a 64-env run would build minibatches
    eight times the size of an 8-env run and exhaust the device that was fine
    yesterday.

    A small smoke rollout gets a valid size rather than the cap: at 64 samples
    the answer is 64, not 128, because a minibatch larger than the rollout is
    not a minibatch.
    """
    if rollout <= 0:
        raise ValueError(f"a rollout must hold at least one sample, got {rollout}")
    if cap <= 0:
        raise ValueError(f"the minibatch cap must be positive, got {cap}")
    for size in range(min(cap, rollout), 0, -1):
        if rollout % size == 0:
            return size
    return 1  # pragma: no cover - unreachable: 1 divides everything


def make_env(config: SessionConfig, rewards: Rewards | None = None) -> RoyaleVecEnv:
    """The one place an env is constructed.

    Both entry points come through here, so the CLI and the dashboard cannot
    end up training against differently configured arenas.
    """
    config.validate()
    return RoyaleVecEnv(
        envs=config.envs,
        seed=config.seed,
        enemies=config.enemies,
        max_frames=config.max_frames,
        hold_frames=config.hold_frames,
        threads=config.threads,
        binary=config.binary,
        rewards=rewards or Rewards.load(config.rewards_path),
    )


@contextlib.contextmanager
def session(config: SessionConfig, rewards: Rewards | None = None) -> Iterator[RoyaleVecEnv]:
    """An env that is closed however the block ends.

    The gym is a child process. Leaving one running after a failed build or an
    interrupt costs a CPU core until someone notices, and on a restart the next
    run competes with it.
    """
    env = make_env(config, rewards)
    try:
        yield env
    finally:
        env.close()


def check_env(env: RoyaleVecEnv) -> None:
    """A smoke check appropriate to a batched env.

    Gymnasium's `check_env` expects a single scalar environment and would
    reject this one for being batched, which says nothing about whether it
    works. This drives the real contract instead: reset, step, shapes, dtypes,
    and finite observations.
    """
    observations = env.reset()
    width = env.layout.observation_values
    if observations.shape != (env.num_envs, width):
        raise ValueError(
            f"reset produced {observations.shape}, expected {(env.num_envs, width)}"
        )
    if observations.dtype != np.float32:
        raise ValueError(f"observations must be float32, got {observations.dtype}")

    actions = np.zeros(env.num_envs, dtype=np.int64)
    stepped, rewards, dones, infos = env.step(actions)
    if stepped.shape != observations.shape:
        raise ValueError(f"step produced {stepped.shape}, expected {observations.shape}")
    if not np.isfinite(stepped).all():
        raise ValueError("a stepped observation is not finite")
    if rewards.shape != (env.num_envs,) or dones.shape != (env.num_envs,):
        raise ValueError("rewards and dones must be one value per env")
    if len(infos) != env.num_envs:
        raise ValueError(f"expected {env.num_envs} infos, got {len(infos)}")


def build_model(
    config: SessionConfig, env: RoyaleVecEnv, resume: str | Path | None = None
) -> PPO:
    """A fresh model, or one resumed from a checkpoint.

    The live layout goes into a new model so the extractor is built for the
    observations it will actually receive. A resumed one is checked against
    that layout **before** the env is attached, because a checkpoint whose
    channels mean something else would otherwise train on happily.
    """
    architecture = config.resolved_architecture()
    batch = minibatch_size(config.rollout_samples(), config.minibatch_cap)
    shared: dict[str, Any] = {
        "n_steps": config.n_steps,
        "batch_size": batch,
        "n_epochs": config.n_epochs,
        "learning_rate": config.learning_rate,
        "device": config.device,
    }

    if resume is None:
        return PPO(**architecture.ppo_kwargs(env.layout), env=env, **shared)

    path = Path(resume)
    if not path.exists():
        raise GymError(f"no checkpoint at {path}")
    # Before `env=` is passed, so a mismatch is a refusal rather than a run.
    require_loadable(path, env.layout)
    return PPO.load(path, env=env, **architecture.resume_kwargs(), **shared)


class EpisodeRecorder(BaseCallback):
    """Collect finished episodes from the infos Task 9 fills in.

    Reads `episode_summary`, which describes the episode that *ended* rather
    than the one that replaced it, so a chart of survival time is a chart of
    episodes and not of auto-resets.
    """

    def __init__(self, log: EpisodeLog | None = None, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.log = log or EpisodeLog()

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", ()):
            summary = info.get("episode_summary")
            if summary is not None:
                self.log.record(summary)
        return True

    @property
    def episodes(self) -> list[dict[str, Any]]:
        return self.log.episodes


@dataclass
class CheckpointWriter:
    """Where a run's checkpoints go, and what they are called."""

    directory: Path
    run_name: str
    saved: list[Path] = field(default_factory=list)

    @classmethod
    def for_config(cls, config: SessionConfig) -> "CheckpointWriter":
        return cls(directory=Path(config.checkpoint_dir), run_name=config.run_name)

    def save(self, model: PPO, step: int | None = None) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        suffix = "final" if step is None else f"{step:09d}"
        path = self.directory / f"{self.run_name}-{suffix}.zip"
        model.save(path)
        self.saved.append(path)
        return path

    def latest(self) -> Path | None:
        if not self.directory.exists():
            return None
        found = sorted(self.directory.glob(f"{self.run_name}-*.zip"))
        return found[-1] if found else None


def train(
    config: SessionConfig,
    *,
    resume: str | Path | None = None,
    callback: BaseCallback | None = None,
) -> Path:
    """Run one training session end to end, and save what it produced.

    Returns the final checkpoint's path. The gym is closed whatever happens,
    including a failed build and an interrupt.
    """
    config.validate()
    with session(config) as env:
        model = build_model(config, env, resume=resume)
        writer = CheckpointWriter.for_config(config)
        try:
            model.learn(total_timesteps=config.total_timesteps, callback=callback)
        except KeyboardInterrupt:
            # An interrupted run still has weights worth keeping; losing them
            # would make stopping a run expensive enough to avoid.
            path = writer.save(model)
            raise GymError(f"interrupted; the model was saved to {path}") from None
        return writer.save(model)


def describe(config: SessionConfig, layout: Layout | None = None) -> str:
    """A one-screen summary of what is about to run."""
    architecture = config.resolved_architecture()
    rollout = config.rollout_samples()
    lines = [
        f"architecture   {architecture.name}",
        f"envs           {config.envs}",
        f"rollout        {config.n_steps} steps x {config.envs} envs = {rollout} samples",
        f"minibatch      {minibatch_size(rollout, config.minibatch_cap)} "
        f"(cap {config.minibatch_cap})",
        f"gamma          {architecture.gamma:.6f}",
        f"gae_lambda     {architecture.gae_lambda:.6f}",
        f"arena          {config.enemies} enemies, {config.max_frames} frame budget",
        f"prediction     {config.hold_frames} frame hold (one action per frame)",
        f"gym threads    {config.threads}",
    ]
    if layout is not None:
        lines.append(f"observation    {layout.observation_values} values, layout v{layout.version}")
    return "\n".join(lines)
