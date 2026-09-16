"""Per-step events the dashboard charts, and the episode bookkeeping behind them.

Two rules run through all of this.

**Counts are per step, never cumulative.** The dashboard sums what it is given
over whatever window it is drawing, so handing it a running total would have it
sum the totals and chart a curve that climbs with the square of the run.

**An episode's numbers describe the episode that ended**, not the one that
replaced it. The gym auto-resets, so the frame the learner is told about and
the frame the arena is now on are different episodes; reporting the new one's
zero would erase the very episode that just finished.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["FRAMES_PER_SECOND", "EpisodeTracker", "TrainingEvents"]

#: The simulation's fixed step. One action per frame, so a 60-frame episode is
#: one second of survival, and the dashboard converts with this and nothing
#: else.
FRAMES_PER_SECOND = 60.0


@dataclass(frozen=True)
class TrainingEvents:
    """What happened during one step, in the keys the charts read."""

    #: Frames survived during this step: one, or zero if the player died on it.
    survival_frames: int
    #: Deaths during this step. Royale has one life, so this is 0 or 1.
    deaths: int
    #: Royale has no extra lives, so nothing is ever spent.
    lives_spent: int
    #: Enemy-on-enemy kills during this step.
    enemies_destroyed: int

    def as_dict(self) -> dict[str, int]:
        return {
            "survival_frames": self.survival_frames,
            "deaths": self.deaths,
            "lives_spent": self.lives_spent,
            "enemies_destroyed": self.enemies_destroyed,
        }


@dataclass
class EpisodeTracker:
    """One env's running episode, for the metadata a transition carries.

    `frames` and `run_frames` are equal in Royale: a run is one episode,
    because there are no extra lives to carry a run across deaths. They are
    both reported because the charts read both, and a run-aware game would make
    them differ.
    """

    frames: int = 0
    enemies_destroyed: int = 0
    reward: float = 0.0

    def record(self, *, enemy_deaths: int, reward: float, frame: int) -> None:
        """Fold one transition in.

        The frame number comes from the gym rather than being counted here.
        The arena is the authority on how long its episode has run, and a
        counter kept on this side would drift the moment a message was
        retried, reordered or dropped.
        """
        self.frames = frame
        self.enemies_destroyed += enemy_deaths
        self.reward += reward

    def summary(self, *, terminated: bool) -> dict[str, float | int | bool]:
        """The finished episode, described after it has ended."""
        return {
            "frames": self.frames,
            "run_frames": self.frames,
            "seconds": self.frames / FRAMES_PER_SECOND,
            "enemies_destroyed": self.enemies_destroyed,
            "return": self.reward,
            "died": terminated,
        }

    def reset(self) -> None:
        self.frames = 0
        self.enemies_destroyed = 0
        self.reward = 0.0


@dataclass
class EpisodeLog:
    """Finished episodes, for a dashboard or a test to read back."""

    episodes: list[dict[str, float | int | bool]] = field(default_factory=list)

    def record(self, summary: dict[str, float | int | bool]) -> None:
        self.episodes.append(summary)

    @property
    def total_seconds(self) -> float:
        return sum(float(episode["seconds"]) for episode in self.episodes)

    @property
    def deaths(self) -> int:
        return sum(1 for episode in self.episodes if episode["died"])

    @property
    def enemies_destroyed(self) -> int:
        return sum(int(episode["enemies_destroyed"]) for episode in self.episodes)
