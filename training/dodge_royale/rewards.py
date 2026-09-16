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
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Rewards", "DEFAULT_REWARDS_PATH"]

#: Where `dodge_royale.train` looks for reward settings by default.
DEFAULT_REWARDS_PATH = Path(__file__).resolve().parent.parent / "rewards.json"

#: Each enemy-on-enemy kill is credited at half an enemy, matching the scheme
#: this was adapted from: the agent did not cause the kill, so it earns the
#: uncontrolled share of it.
UNCONTROLLED_KILL_SHARE = 0.5


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

    def for_step(self, *, terminated: bool, truncated: bool, enemy_deaths: int) -> float:
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
        return reward

    def as_dict(self) -> dict[str, float]:
        return {
            "survival_per_frame": self.survival_per_frame,
            "death_penalty": self.death_penalty,
            "uncontrolled_score_weight": self.uncontrolled_score_weight,
        }
