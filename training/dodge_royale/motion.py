"""The player's movement rules, restated in Python and pinned to Rust's.

The simulation owns movement. This module exists because the trainer has to
*predict* it: to align a spatial memory from one frame to the next, it needs to
know how far the player moved, and nothing in the gym wire format carries that.
So the rules are implemented twice, and `tests/fixtures/player-motion/` is what
keeps the two copies honest -- Rust emits positions and velocities from the
real `advance_motion`, and this module has to land on the same numbers.

**Operation order is copied, not just the formula.** Several steps here could be
written more naturally and would still be algebraically right: the velocity
blend as ``target + (v - target) * decay``, the normalisation as ``v / length``.
Rust reaches those values through `glam`'s own spelling -- a two-term lerp, a
multiply by a reciprocal -- and float arithmetic is not associative, so a
different spelling lands a unit or two away. That is far inside the fixture
tolerance, but matching the order costs nothing and makes a real disagreement
visible instead of hiding under the noise of a rewrite.

Everything is `float32`, for the same reason. The simulation is `f32`, and
computing the reference in `float64` would make this module *more* accurate than
the thing it is supposed to reproduce, which is not what a reference is for.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "MOTION_CONTRACT_ID",
    "MotionContract",
    "MotionStep",
    "advance_motion",
    "intent_from_command",
    "load_contract",
    "normalize_or_zero",
    "wrap_position",
]

#: The movement rules this module implements. A fixture or checkpoint carrying
#: a different identifier was produced under different physics, and its
#: displacements do not describe this build.
MOTION_CONTRACT_ID = "royale-motion-1"

F32 = np.float32


@dataclass(frozen=True)
class MotionContract:
    """Every constant the two implementations have to agree on."""

    id: str
    top_speed: np.float32
    movement_response: np.float32
    max_frame_seconds: np.float32
    idle_threshold: np.float32
    seconds_per_frame: np.float32
    world_units_per_pixel: np.float32
    world_half_extents: np.ndarray

    @classmethod
    def from_manifest(cls, raw: dict) -> "MotionContract":
        """Read a contract out of a fixture manifest.

        Taken from `contract_bits` rather than from the decimals beside them.
        The decimals do round-trip -- they are written shortest-round-trip for
        `f32` -- but a bit pattern cannot be wrong, and a reader that compares
        against the wrong constant by one unit in the last place would produce
        a failure whose cause is invisible.
        """
        bits = raw["contract_bits"]
        contract = cls(
            id=str(raw["contract"]["id"]),
            top_speed=_from_bits(bits["top_speed"]),
            movement_response=_from_bits(bits["movement_response"]),
            max_frame_seconds=_from_bits(bits["max_frame_seconds"]),
            idle_threshold=_from_bits(bits["idle_threshold"]),
            seconds_per_frame=_from_bits(bits["seconds_per_frame"]),
            world_units_per_pixel=_from_bits(bits["world_units_per_pixel"]),
            world_half_extents=np.array(
                [_from_bits(value) for value in bits["world_half_extents"]], dtype=F32
            ),
        )
        contract.require_known()
        return contract

    def require_known(self) -> None:
        """Refuse a contract this module does not implement.

        Not a warning. Every displacement this module predicts would be wrong
        by an amount too small to look like a bug and too large to ignore.
        """
        if self.id != MOTION_CONTRACT_ID:
            raise ValueError(
                f"this module implements the motion contract {MOTION_CONTRACT_ID!r}, "
                f"and was given {self.id!r}"
            )


def _from_bits(bits: int) -> np.float32:
    return np.array([int(bits)], dtype=np.uint32).view(F32)[0]


def load_contract(manifest: str | Path) -> MotionContract:
    """The contract recorded beside the committed motion fixtures."""
    raw = json.loads(Path(manifest).read_text(encoding="utf-8"))
    return MotionContract.from_manifest(raw)


@dataclass(frozen=True)
class MotionStep:
    """What one simulation frame did."""

    #: Where the player is now, wrapped into the arena.
    position: np.ndarray
    #: The velocity it ends the frame with.
    velocity: np.ndarray
    #: How far it travelled, before wrapping. A seam crossing does not change
    #: this; only the coordinate jumps.
    displacement: np.ndarray
    #: The command after the idle floor, which is what actually steered.
    intent: np.ndarray


def normalize_or_zero(vector: np.ndarray) -> np.ndarray:
    """A unit vector, or zero when there is no direction to take.

    `glam` multiplies by the reciprocal of the length rather than dividing by
    it, and rejects a reciprocal that is not finite and positive -- which is
    what makes a zero vector, and an overflowing one, come back as zero.
    """
    values = np.asarray(vector, dtype=F32)
    length = np.sqrt(np.sum(values * values, axis=-1, keepdims=True), dtype=F32)
    with np.errstate(divide="ignore", invalid="ignore"):
        reciprocal = F32(1.0) / length
        # Rust takes the fallback for the whole vector rather than scaling by a
        # bad reciprocal, so the scaled value is selected, not the factor. A
        # vector holding an infinity has a reciprocal of zero -- finite, but not
        # positive -- and must come back as nothing rather than as a NaN.
        usable = np.isfinite(reciprocal) & (reciprocal > F32(0.0))
        scaled = values * reciprocal
    return np.where(usable, scaled, F32(0.0))


def intent_from_command(command: np.ndarray, contract: MotionContract) -> np.ndarray:
    """A commanded direction as a player intent, with short commands idling.

    Compared as squared lengths, the way the simulation does it, so a command
    exactly on the floor falls the same side in both.
    """
    values = np.asarray(command, dtype=F32)
    squared = np.sum(values * values, axis=-1, keepdims=True, dtype=F32)
    floor = F32(contract.idle_threshold) * F32(contract.idle_threshold)
    return np.where(squared < floor, F32(0.0), values)


def wrap_position(position: np.ndarray, half_extents: np.ndarray) -> np.ndarray:
    """Bring a position back inside the world, wrapping at every edge."""
    values = np.asarray(position, dtype=F32)
    half = np.asarray(half_extents, dtype=F32)
    size = half * F32(2.0)
    if not np.isfinite(size).all() or bool((size <= F32(0.0)).any()):
        return values
    if not np.isfinite(values).all():
        return values
    # Rust's `rem_euclid`: a truncating remainder, then one size added back if
    # it came out negative. `np.fmod` truncates the way Rust's `%` does;
    # `np.remainder` would floor instead and disagree on negatives.
    shifted = values + half
    remainder = np.fmod(shifted, size, dtype=F32)
    remainder = np.where(remainder < F32(0.0), remainder + np.abs(size), remainder)
    return remainder - half


def advance_motion(
    position: np.ndarray,
    velocity: np.ndarray,
    command: np.ndarray,
    contract: MotionContract,
    seconds: float | None = None,
) -> MotionStep:
    """Advance one frame, returning everything the caller might need.

    `seconds` defaults to one simulation frame, which is what a gym STEP
    advances: exactly one, on exactly one command. The cap matches the
    simulation's, so a caller that hands over a long frame gets the same
    truncation rather than a longer journey.
    """
    elapsed = F32(contract.seconds_per_frame if seconds is None else seconds)
    elapsed = F32(np.clip(elapsed, F32(0.0), F32(contract.max_frame_seconds)))

    previous_position = np.asarray(position, dtype=F32)
    previous_velocity = np.asarray(velocity, dtype=F32)
    intent = intent_from_command(command, contract)
    target = normalize_or_zero(intent) * F32(contract.top_speed)

    response = F32(contract.movement_response)
    blend = F32(1.0) - F32(np.exp(F32(-response * elapsed), dtype=F32))
    # The two-term lerp `glam` uses, not `previous + (target - previous) * blend`.
    next_velocity = previous_velocity * (F32(1.0) - blend) + target * blend
    # The integral of the exponential velocity curve, so travel and acceleration
    # agree however the frame is divided. `velocity * elapsed` is not the same
    # number while the player is still speeding up.
    displacement = target * elapsed + (previous_velocity - next_velocity) / response
    next_position = wrap_position(
        previous_position + displacement, contract.world_half_extents
    )
    return MotionStep(
        position=next_position,
        velocity=next_velocity,
        displacement=displacement,
        intent=intent,
    )
