"""The Python motion reference, against Rust's own committed answers.

Nothing here computes an expectation. Every expected position, velocity and
displacement was produced by `advance_motion` in the simulation and committed;
this replays the same commands and has to arrive at them. A fixture generated
from the formula under test would agree with it by construction, and would go
on agreeing through a shared mistake -- which is the whole failure this is
supposed to catch.

Displacement is reached from opposite directions on the two sides. Rust
measures it from the positions it produced, across a seam where there is one;
Python computes it from the velocity curve. They agree only if the position
integration and the velocity ramp both agree.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dodge_royale.motion import (
    MOTION_CONTRACT_ID,
    MotionContract,
    advance_motion,
    intent_from_command,
    load_contract,
    normalize_or_zero,
    wrap_position,
)

FIXTURES = Path(__file__).resolve().parent.parent.parent / "tests" / "fixtures" / "player-motion"

#: Per step, in reference pixels, for displacement only. The two sides reach
#: it by different routes -- Rust from the positions it produced, Python from
#: the velocity curve -- so they agree to within float32's granularity at the
#: coordinates involved rather than exactly. Measured worst case is 5.6e-5,
#: from the seam scenarios where coordinates are near three thousand world
#: units and one unit in the last place is already 4e-5 reference pixels.
#:
#: A failure here is a disagreement to investigate, not a number to raise.
STEP_TOLERANCE = 1e-4

#: Accumulated over the six-hundred-step trace, in reference pixels.
TRACE_TOLERANCE = 0.02


@pytest.fixture(scope="module")
def manifest() -> dict:
    path = FIXTURES / "manifest.json"
    if not path.exists():
        pytest.skip(
            "no motion fixtures; generate them with `cargo run --locked "
            "--no-default-features --example motion_fixtures`"
        )
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def contract(manifest) -> MotionContract:
    return MotionContract.from_manifest(manifest)


def steps_of(manifest: dict, name: str) -> np.ndarray:
    """One scenario's recorded steps, as `(steps, 10)` float32."""
    entry = next(item for item in manifest["scenarios"] if item["name"] == name)
    raw = np.frombuffer((FIXTURES / entry["file"]).read_bytes(), dtype="<f4")
    columns = len(manifest["record"])
    assert raw.size == entry["steps"] * columns, entry["file"]
    return raw.reshape(entry["steps"], columns)


def start_of(manifest: dict, name: str) -> tuple[np.ndarray, np.ndarray]:
    entry = next(item for item in manifest["scenarios"] if item["name"] == name)
    position = np.array([sample["value"] for sample in entry["start_position"]], dtype=np.float32)
    velocity = np.array([sample["value"] for sample in entry["start_velocity"]], dtype=np.float32)
    return position, velocity


def scenario_names(manifest: dict) -> list[str]:
    return [item["name"] for item in manifest["scenarios"]]


def replay(manifest: dict, contract: MotionContract, name: str):
    """Walk one scenario, yielding what Rust recorded and what Python computed."""
    recorded = steps_of(manifest, name)
    position, velocity = start_of(manifest, name)
    for index, row in enumerate(recorded):
        command = row[0:2]
        step = advance_motion(position, velocity, command, contract)
        yield index, row, step
        position, velocity = step.position, step.velocity


def as_pixels(world: np.ndarray, contract: MotionContract) -> np.ndarray:
    """World units as reference pixels, which is what the tolerances are in."""
    return np.asarray(world, dtype=np.float64) / float(contract.world_units_per_pixel)


# --- the contract ----------------------------------------------------------


def test_the_fixtures_were_made_under_the_rules_this_module_implements(contract):
    assert contract.id == MOTION_CONTRACT_ID


def test_the_recorded_decimals_and_bits_describe_the_same_constants(manifest, contract):
    """Belt and braces: the manifest carries each constant twice."""
    decimals = manifest["contract"]
    assert np.float32(decimals["top_speed"]) == contract.top_speed
    assert np.float32(decimals["movement_response"]) == contract.movement_response
    assert np.float32(decimals["idle_threshold"]) == contract.idle_threshold
    assert np.float32(decimals["seconds_per_frame"]) == contract.seconds_per_frame
    assert np.float32(decimals["max_frame_seconds"]) == contract.max_frame_seconds
    assert np.float32(decimals["world_units_per_pixel"]) == contract.world_units_per_pixel


def test_a_contract_from_other_physics_is_refused(manifest):
    """Loudly, because every prediction under it would be quietly wrong."""
    foreign = json.loads(json.dumps(manifest))
    foreign["contract"]["id"] = "some-other-game-2"
    with pytest.raises(ValueError, match="motion contract"):
        MotionContract.from_manifest(foreign)


def test_one_step_is_one_frame(contract):
    """The cadence the whole plan rests on. A STEP advances this much time."""
    assert float(contract.seconds_per_frame) == pytest.approx(1.0 / 60.0, abs=1e-7)


# --- the scenarios ---------------------------------------------------------


def test_every_committed_scenario_is_replayed(manifest):
    """A scenario nobody replays is a fixture that proves nothing."""
    covered = set(scenario_names(manifest))
    assert covered == {
        "rest",
        "acceleration",
        "braking",
        "reversal",
        "diagonal",
        "off-compass",
        "off-compass-negative",
        "over-unit-command",
        "idle-floor",
        "seam-x",
        "seam-y",
        "long-trace",
    }


@pytest.mark.parametrize(
    "name",
    [
        "rest",
        "acceleration",
        "braking",
        "reversal",
        "diagonal",
        "off-compass",
        "off-compass-negative",
        "over-unit-command",
        "idle-floor",
        "seam-x",
        "seam-y",
        "long-trace",
    ],
)
def test_python_reproduces_the_recorded_motion(manifest, contract, name):
    """Bit for bit, for everything the two sides compute the same way.

    Not "within a tolerance": intent, position and velocity come out of the
    same arithmetic in the same order and the same precision, so they are
    identical, and asserting only closeness would let a real divergence hide
    under a tolerance that was never needed. `numpy`'s float32 `exp` even
    agrees with Rust's to the last bit, which is what makes this possible.

    Displacement is the exception, and deliberately so -- see STEP_TOLERANCE.
    """
    for index, row, step in replay(manifest, contract, name):
        for label, got, want in (
            ("intent", step.intent, row[2:4]),
            ("position", step.position, row[4:6]),
            ("velocity", step.velocity, row[6:8]),
        ):
            assert np.array_equal(np.asarray(got, dtype=np.float32), want), (
                f"{name} step {index}: {label} is {got}, recorded {want}"
            )

        error = np.abs(as_pixels(step.displacement, contract) - as_pixels(row[8:10], contract))
        assert error.max() < STEP_TOLERANCE, (
            f"{name} step {index}: displacement out by {error.max()} reference px "
            f"(got {step.displacement}, recorded {row[8:10]})"
        )


def test_the_long_trace_does_not_drift(manifest, contract):
    """Per-step agreement is not enough on its own: a formula that is nearly
    right passes every step and arrives somewhere else after six hundred.

    The bound is the plan's, and it is enormously slack for what this actually
    does -- position never diverges at all, because each step is exact and an
    exact step from an exact state stays exact. Keeping the looser bound named
    here says what would still be acceptable if that ever stopped being true.
    """
    worst = 0.0
    for _index, row, step in replay(manifest, contract, "long-trace"):
        error = np.abs(as_pixels(step.position, contract) - as_pixels(row[4:6], contract))
        worst = max(worst, float(error.max()))
    assert worst < TRACE_TOLERANCE, f"drifted {worst} reference px over the trace"
    assert worst == 0.0, "a step that was exact yesterday has started rounding"


def test_the_trace_actually_travels_far_enough_to_matter(manifest, contract):
    """A drift bound over a trace that barely moves would prove nothing."""
    recorded = steps_of(manifest, "long-trace")
    travelled = np.abs(recorded[:, 8:10]).sum(axis=0)
    assert as_pixels(travelled, contract).min() > 100.0


# --- the properties the fixtures were chosen to pin -------------------------


def test_a_commands_length_buys_no_speed(manifest, contract):
    """`over-unit-command` is five times unit length and must match the unit
    one step for step. Length is a heading's magnitude, not a throttle."""
    unit = steps_of(manifest, "acceleration")
    over = steps_of(manifest, "over-unit-command")
    assert np.array_equal(unit[:, 4:10], over[:, 4:10])


def test_a_diagonal_travels_at_an_axis_speed(manifest, contract):
    """Without normalisation the corner would be root two times faster."""
    axis = steps_of(manifest, "acceleration")
    diagonal = steps_of(manifest, "diagonal")
    reached = min(len(axis), len(diagonal)) - 1
    axis_speed = np.hypot(*axis[reached, 6:8])
    diagonal_speed = np.hypot(*diagonal[reached, 6:8])
    assert diagonal_speed == pytest.approx(axis_speed, rel=1e-5)


def test_the_idle_floor_zeroes_a_short_command_and_passes_a_long_one(manifest, contract):
    """Both directions across the floor, which is why the sweep rises and falls."""
    recorded = steps_of(manifest, "idle-floor")
    lengths = np.hypot(recorded[:, 0], recorded[:, 1])
    intents = np.hypot(recorded[:, 2], recorded[:, 3])
    floor = float(contract.idle_threshold)

    below = lengths < floor
    assert below.any() and (~below).any(), "the sweep must cross the floor"
    assert (intents[below] == 0.0).all(), "a short command means standing still"
    assert (intents[~below] > 0.0).all(), "a long enough one steers"
    # And the intent is the command itself, not a normalised version of it:
    # normalising here would make the floor invisible to everything downstream.
    steering = ~below
    assert np.allclose(intents[steering], lengths[steering], rtol=1e-6)


def test_python_agrees_about_which_commands_idle(manifest, contract):
    recorded = steps_of(manifest, "idle-floor")
    commands = recorded[:, 0:2]
    assert np.array_equal(
        np.asarray(intent_from_command(commands, contract), dtype=np.float32),
        recorded[:, 2:4],
    )


@pytest.mark.parametrize("name", ["seam-x", "seam-y"])
def test_a_seam_crossing_moves_a_little_and_jumps_the_coordinate(manifest, contract, name):
    entry = next(item for item in manifest["scenarios"] if item["name"] == name)
    assert entry["crosses_seam"], f"{name} was supposed to leave the world"

    recorded = steps_of(manifest, name)
    half = np.asarray(contract.world_half_extents, dtype=np.float64)
    # Every recorded position stays inside the arena...
    assert (np.abs(recorded[:, 4:6]) <= half + 1e-3).all()
    # ...and no single step travels more than a top-speed frame, however far
    # apart two consecutive coordinates look.
    per_step = np.hypot(recorded[:, 8], recorded[:, 9])
    limit = float(contract.top_speed) * float(contract.seconds_per_frame) * 1.01
    assert per_step.max() <= limit, f"a step travelled {per_step.max()} world units"
    # The naive difference of coordinates does exceed it, which is the mistake
    # these two scenarios exist to catch.
    naive = np.abs(np.diff(recorded[:, 4:6], axis=0)).max()
    assert naive > limit


def test_a_still_player_stays_exactly_still(manifest, contract):
    recorded = steps_of(manifest, "rest")
    assert not recorded[:, 4:10].any(), "no command, no drift"


# --- the pieces, on their own ----------------------------------------------


def test_normalising_nothing_gives_nothing_rather_than_a_nan():
    for empty in ([0.0, 0.0], [0.0, -0.0]):
        assert np.array_equal(normalize_or_zero(np.array(empty, dtype=np.float32)), [0.0, 0.0])


def test_normalising_something_unmeasurable_gives_nothing():
    """A reciprocal that is not finite and positive falls back to zero, so an
    infinity cannot become a heading."""
    for broken in ([np.inf, 0.0], [np.nan, 1.0]):
        result = normalize_or_zero(np.array(broken, dtype=np.float32))
        assert np.isfinite(result).all() and not result.any()


def test_wrapping_is_stable_and_symmetric(contract):
    half = contract.world_half_extents
    for point in ([0.0, 0.0], [2999.0, -1999.0], [3010.0, 2010.0], [-9000.0, 7000.0]):
        once = wrap_position(np.array(point, dtype=np.float32), half)
        assert np.array_equal(wrap_position(once, half), once), "wrapping twice changes nothing"
        assert (np.abs(once) <= np.asarray(half, dtype=np.float32)).all()


def test_a_long_frame_is_capped_the_way_the_simulation_caps_it(contract):
    """A backgrounded tab must not teleport the player, on either side."""
    position = np.zeros(2, dtype=np.float32)
    velocity = np.zeros(2, dtype=np.float32)
    command = np.array([1.0, 0.0], dtype=np.float32)
    capped = advance_motion(position, velocity, command, contract, seconds=10.0)
    at_cap = advance_motion(
        position, velocity, command, contract, seconds=float(contract.max_frame_seconds)
    )
    assert np.array_equal(capped.position, at_cap.position)
    assert np.array_equal(capped.velocity, at_cap.velocity)


def test_the_reference_works_on_a_batch_of_players(manifest, contract):
    """Task 2 aligns one state per env, so this has to answer for all of them
    at once and give each the same answer it would alone."""
    recorded = steps_of(manifest, "off-compass")
    commands = np.stack([recorded[0, 0:2], np.zeros(2, dtype=np.float32)])
    positions = np.zeros((2, 2), dtype=np.float32)
    velocities = np.zeros((2, 2), dtype=np.float32)

    together = advance_motion(positions, velocities, commands, contract)
    assert together.position.shape == (2, 2)
    for index in range(2):
        alone = advance_motion(positions[index], velocities[index], commands[index], contract)
        assert np.array_equal(together.position[index], alone.position)
        assert np.array_equal(together.velocity[index], alone.velocity)
