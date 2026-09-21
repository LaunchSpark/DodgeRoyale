"""What a frame of DodgeRoyale is worth.

Reward is computed here, in Python, rather than in the simulation, so that
tuning it is a config edit and not a rebuild. The gym reports what happened;
this decides what it was worth.

Only the controls that mean something in Royale are exposed. `edge_penalty`
and `score_weight` are carried in the file for compatibility with the reward
sheets this scheme was adapted from, but Royale has no walls to be pushed
against and no score to earn, so applying them would be inventing a signal the
simulation never produces.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Rewards", "DEFAULT_REWARDS_PATH", "DIRECTION_EPSILON"]


def _heading(direction) -> tuple[float, float] | None:
    """A direction as a unit heading, or None if it points nowhere."""
    if direction is None:
        return None
    x, y = (float(direction[0]), float(direction[1]))
    length = math.hypot(x, y)
    if not math.isfinite(length) or length < DIRECTION_EPSILON:
        return None
    return x / length, y / length

#: Where `dodge_royale.train` looks for reward settings by default.
DEFAULT_REWARDS_PATH = Path(__file__).resolve().parent.parent / "rewards.json"

#: Each enemy-on-enemy kill is credited at half an enemy, matching the scheme
#: this was adapted from: the agent did not cause the kill, so it earns the
#: uncontrolled share of it.
UNCONTROLLED_KILL_SHARE = 0.5

#: Below this the direction is the simulation's idle, and has no heading. Two
#: zero vectors are not a turn, and a heading read off a vector this short is
#: the noise in its two components rather than a decision.
DIRECTION_EPSILON = 1e-6


@dataclass(frozen=True)
class Rewards:
    """The reward scheme, and the one place it is applied.

    Keeping :meth:`for_step` here rather than in the environment means the
    arithmetic can be tested on its own, without a gym, a client or a batch.
    """

    #: Earned for every frame the player is alive at the end of.
    survival_per_frame: float = 0.02
    #: Subtracted once, on the frame the player is hit.
    death_penalty: float = 2.0
    #: Share of an enemy-on-enemy kill the agent is credited with.
    uncontrolled_score_weight: float = 0.05
    #: Charged for reversing hard between one decision and the next, at a full
    #: about-face. Small on purpose: a fifth of a tenth of one decision's
    #: survival pay, so it shades between two headings that are otherwise
    #: equally safe and never argues with staying alive. A dodge that needs a
    #: reversal should still be worth making.
    turn_penalty: float = 0.01
    #: Turning less than this is free. Forty-five degrees is one step around
    #: the compass the nine candidate paths are drawn on, so moving to an
    #: adjacent heading costs nothing and only skipping one does.
    turn_free_degrees: float = 45.0

    def turn_cost(self, previous, direction) -> float:
        """What this decision's change of heading cost.

        Zero up to the free angle, then straight-line to the full penalty at a
        complete about-face. Graded rather than a cliff at the threshold: a
        step charge would make 46 degrees as expensive as 180, and the agent
        would learn that once a turn is worth paying for it may as well be the
        biggest one available.

        Starting or stopping is not a turn. Neither has two headings to be
        between, and charging for them would tax the decision to stand still --
        which is the one the idle floor exists to make available.
        """
        if not self.turn_penalty:
            return 0.0
        before, after = _heading(previous), _heading(direction)
        if before is None or after is None:
            return 0.0
        dot = max(-1.0, min(1.0, before[0] * after[0] + before[1] * after[1]))
        degrees = math.degrees(math.acos(dot))
        free = max(0.0, min(self.turn_free_degrees, 180.0))
        if degrees <= free or degrees <= 0.0:
            return 0.0
        return self.turn_penalty * (degrees - free) / (180.0 - free)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Rewards":
        """Read a reward file, falling back to the defaults if there is none.

        Unknown keys are ignored rather than refused: the file format is shared
        with schemes that have controls Royale does not use, and failing on
        `edge_penalty` would make those files unusable for no benefit.
        """
        source = Path(path) if path is not None else DEFAULT_REWARDS_PATH
        if not source.exists():
            return cls()
        raw = json.loads(source.read_text(encoding="utf-8"))
        known = {field for field in cls.__dataclass_fields__}
        return cls(**{key: float(value) for key, value in raw.items() if key in known})

    def for_step(
        self,
        *,
        terminated: bool,
        truncated: bool,
        enemy_deaths: int,
        previous_direction=None,
        direction=None,
    ) -> float:
        """What one transition earned.

        Survival is paid for a frame the player finished alive, so a death does
        not also collect it. Truncation is not a death: the budget ran out
        while the player was in perfectly good health, and penalising that
        would teach the agent that the end of an episode is dangerous.

        `truncated` is accepted so that this reads as the whole rule rather
        than half of it, and so the same-frame case -- both flags set, because
        a player can die on the frame the budget expires -- is decided here
        and not by the caller.
        """
        died = terminated
        reward = 0.0 if died else self.survival_per_frame
        if died:
            reward -= self.death_penalty
        reward += self.uncontrolled_score_weight * UNCONTROLLED_KILL_SHARE * enemy_deaths
        # Charged even on the frame the player dies: the reversal was made
        # before the outcome was known, and a turn that happens to end an
        # episode is not thereby free.
        if previous_direction is not None and direction is not None:
            reward -= self.turn_cost(previous_direction, direction)
        return reward

    def as_dict(self) -> dict[str, float]:
        return {
            "survival_per_frame": self.survival_per_frame,
            "death_penalty": self.death_penalty,
            "uncontrolled_score_weight": self.uncontrolled_score_weight,
        }
