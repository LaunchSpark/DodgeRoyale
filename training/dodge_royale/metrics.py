"""What a run reports about itself, in one place both entry points read.

The CLI prints these and the dashboard charts them. Keeping the definitions
here rather than in either one means a metric added for a chart is also
available to `--dry-run`, and that neither can quietly disagree with the other
about what "survival" means.

Adding a metric is adding a :class:`MetricDefinition` to :data:`METRICS` and a
field to :class:`Snapshot`. Nothing else has to change: the dashboard renders
whatever the registry lists, so a new metric appears without editing the
notebook.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Literal

from stable_baselines3.common.callbacks import BaseCallback

from .telemetry import FRAMES_PER_SECOND

__all__ = ["METRICS", "MetricDefinition", "MetricsCollector", "Snapshot"]

#: How many finished episodes the rolling averages look back over. Bounded so
#: a long run neither grows without limit nor reports a number dominated by
#: how the policy behaved an hour ago.
EPISODE_WINDOW = 100


@dataclass(frozen=True)
class MetricDefinition:
    """One number a run reports, and how to show it."""

    key: str
    label: str
    unit: str
    description: str
    #: `episode` numbers come from finished episodes, `throughput` from the
    #: clock, `optimizer` from PPO's own logger, `run` from the session.
    group: Literal["episode", "throughput", "optimizer", "run"]
    #: How to render one value. Kept with the definition so the dashboard does
    #: not carry a parallel table of format strings that can drift.
    format: Callable[[Any], str] = str


def _seconds(value: Any) -> str:
    return "—" if value is None else f"{float(value):.2f} s"


def _number(value: Any) -> str:
    return "—" if value is None else f"{float(value):+.3f}"


def _count(value: Any) -> str:
    return "—" if value is None else f"{int(value):,}"


def _rate(value: Any) -> str:
    return "—" if value is None else f"{float(value):,.0f}/s"


def _plain(value: Any) -> str:
    return "—" if value is None else str(value)


#: Every metric a run reports, in display order.
METRICS: tuple[MetricDefinition, ...] = (
    MetricDefinition(
        "survival_seconds",
        "Survival",
        "s",
        f"Mean length of the last {EPISODE_WINDOW} finished episodes. One action "
        f"per {FRAMES_PER_SECOND:.0f} Hz frame, so 60 frames is one second.",
        "episode",
        _seconds,
    ),
    MetricDefinition(
        "episode_return",
        "Return",
        "",
        f"Mean reward of the last {EPISODE_WINDOW} finished episodes.",
        "episode",
        _number,
    ),
    MetricDefinition(
        "deaths",
        "Deaths",
        "",
        "Episodes that ended because the player was hit.",
        "episode",
        _count,
    ),
    MetricDefinition(
        "timeouts",
        "Timeouts",
        "",
        "Episodes that ended because the frame budget ran out. The player was "
        "alive, which is why a learner bootstraps from these and not from deaths.",
        "episode",
        _count,
    ),
    MetricDefinition(
        "episodes",
        "Episodes",
        "",
        "Finished episodes this run.",
        "episode",
        _count,
    ),
    MetricDefinition(
        "steps_per_second",
        "Throughput",
        "steps/s",
        "Environment steps per wall-clock second, measured over the whole run.",
        "throughput",
        _rate,
    ),
    MetricDefinition(
        "timesteps",
        "Steps",
        "",
        "Environment steps taken.",
        "throughput",
        _count,
    ),
    MetricDefinition(
        "value_loss", "Value loss", "", "PPO critic loss.", "optimizer", _number
    ),
    MetricDefinition(
        "policy_loss",
        "Policy loss",
        "",
        "PPO policy gradient loss.",
        "optimizer",
        _number,
    ),
    MetricDefinition(
        "entropy_loss",
        "Entropy loss",
        "",
        "Negative policy entropy. Rising toward zero means the policy is "
        "committing to its choices.",
        "optimizer",
        _number,
    ),
    MetricDefinition(
        "approx_kl",
        "Approx KL",
        "",
        "How far each update moved the policy.",
        "optimizer",
        _number,
    ),
    MetricDefinition(
        "clip_fraction",
        "Clip fraction",
        "",
        "Share of samples PPO's ratio clipping bound.",
        "optimizer",
        _number,
    ),
    MetricDefinition(
        "device",
        "Device",
        "",
        "The device the policy's parameters are actually on, read from the "
        "model rather than from what was asked for.",
        "run",
        _plain,
    ),
    MetricDefinition(
        "updates", "Updates", "", "PPO updates completed.", "run", _count
    ),
)

#: The keys, for a caller that wants them without walking the registry.
METRIC_KEYS: tuple[str, ...] = tuple(metric.key for metric in METRICS)


@dataclass(frozen=True)
class Snapshot:
    """One reading of every metric, safe to hand across a thread boundary.

    Frozen and built only from plain values, so the worker can publish one and
    the reader can hold it for as long as it likes without either of them
    seeing the other's mutations.
    """

    timesteps: int = 0
    episodes: int = 0
    deaths: int = 0
    timeouts: int = 0
    survival_seconds: float | None = None
    episode_return: float | None = None
    steps_per_second: float | None = None
    value_loss: float | None = None
    policy_loss: float | None = None
    entropy_loss: float | None = None
    approx_kl: float | None = None
    clip_fraction: float | None = None
    device: str | None = None
    updates: int = 0
    elapsed: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def rows(self) -> list[tuple[MetricDefinition, str]]:
        """Every metric with its value already formatted, in display order."""
        values = self.as_dict()
        return [(metric, metric.format(values.get(metric.key))) for metric in METRICS]


class MetricsCollector(BaseCallback):
    """Accumulate metrics while PPO runs, and publish them as snapshots.

    A callback rather than a wrapper because it is the only place that can see
    all three sources at once: the infos carry finished episodes, the model's
    logger carries the optimizer's numbers, and the clock carries throughput.
    """

    def __init__(self, window: int = EPISODE_WINDOW, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.window = window
        self._lengths: list[float] = []
        self._returns: list[float] = []
        self.episodes = 0
        self.deaths = 0
        self.timeouts = 0
        self.updates = 0
        self._started = time.monotonic()
        self._device: str | None = None

    def _on_training_start(self) -> None:
        self._started = time.monotonic()
        # Read the device off a parameter rather than trusting what was asked
        # for: "auto" resolves somewhere, and a run on CPU when CUDA was
        # expected should be visible here rather than inferred from its speed.
        try:
            self._device = str(next(self._model().policy.parameters()).device)
        except (StopIteration, AttributeError):  # pragma: no cover - defensive
            self._device = None

    def _on_rollout_end(self) -> None:
        self.updates += 1

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", ()):
            summary = info.get("episode_summary")
            if summary is None:
                continue
            self.episodes += 1
            if summary.get("died"):
                self.deaths += 1
            else:
                self.timeouts += 1
            self._lengths.append(float(summary.get("seconds", 0.0)))
            self._returns.append(float(summary.get("return", 0.0)))
            del self._lengths[: -self.window]
            del self._returns[: -self.window]
        return True

    # -- publishing --

    def _model(self) -> Any | None:
        """The algorithm, once SB3 has attached it.

        `BaseCallback` only gains `.model` when `learn` initialises the
        callback, so anything that can be asked for a snapshot before then --
        the worker publishes one as soon as the thread starts -- has to cope
        with its absence rather than assume it.
        """
        return getattr(self, "model", None)

    def _logged(self, name: str) -> float | None:
        """One of PPO's own numbers, if it has produced any yet.

        The logger is empty until the first update finishes, so every
        optimizer metric is None for the first rollout rather than zero. Zero
        would chart as a real reading.
        """
        logger = getattr(self._model(), "logger", None)
        if logger is None:
            return None
        value = logger.name_to_value.get(name)
        return None if value is None else float(value)

    def snapshot(self) -> Snapshot:
        elapsed = max(time.monotonic() - self._started, 1e-9)
        timesteps = int(getattr(self._model(), "num_timesteps", 0) or 0)
        return Snapshot(
            timesteps=timesteps,
            episodes=self.episodes,
            deaths=self.deaths,
            timeouts=self.timeouts,
            survival_seconds=(
                sum(self._lengths) / len(self._lengths) if self._lengths else None
            ),
            episode_return=(
                sum(self._returns) / len(self._returns) if self._returns else None
            ),
            steps_per_second=timesteps / elapsed if timesteps else None,
            value_loss=self._logged("train/value_loss"),
            policy_loss=self._logged("train/policy_gradient_loss"),
            entropy_loss=self._logged("train/entropy_loss"),
            approx_kl=self._logged("train/approx_kl"),
            clip_fraction=self._logged("train/clip_fraction"),
            device=self._device,
            updates=self.updates,
            elapsed=elapsed,
        )
