# Player-motion fixtures

The simulation owns movement, but the trainer has to **predict** it. Aligning a
spatial memory from one frame to the next means knowing how far the player
moved, and nothing in the gym wire format carries the movement constants. So the
rules end up implemented twice — once in `src/motion.rs`, once in
`training/dodge_royale/motion.py` — and these fixtures are what keep the two
copies honest.

Regenerate after an intentional change to the movement rules:

```sh
cargo run --locked --no-default-features --example motion_fixtures
```

That rewrites every file here. It is a maintenance action, not a test step —
see [`examples/motion_fixtures.rs`](../../../examples/motion_fixtures.rs).

## Why the expectations are not computed in Python

A fixture generated from the formula under test agrees with it by construction,
and goes on agreeing through a shared mistake. So every position and velocity
here came out of the real `advance_motion`.

Displacement is reached from opposite ends. Rust measures it from the positions
it produced, across a seam where there is one; Python computes it from the
velocity curve. They match only if the position integration and the velocity
ramp both match.

## The contract

`manifest.json` opens with the constants both implementations have to agree on,
and `contract_bits` carries each one again as a bit pattern for a reader that
would rather not trust a decimal. `id` names the rules: a fixture or checkpoint
carrying a different one was produced under different physics, and
`MotionContract.require_known` in Python refuses it rather than predicting
displacements that are wrong by an amount too small to look like a bug.

`tests/motion_fixtures.rs` compares the recorded contract against the live one
field by field, so a constant that changes without a regeneration fails on this
side too. That is the dangerous case: the numbers would still be self-consistent
and Python would still agree with them, and both would be describing a game that
no longer exists.

## The record

Each `.bin` is a flat sequence of little-endian `f32`, ten per step:

| Column | Meaning |
|---|---|
| `command_x`, `command_y` | What was commanded, before the idle floor. |
| `intent_x`, `intent_y` | The command after the floor. Zero means standing still. |
| `position_x`, `position_y` | Where the player is after this step, wrapped into the arena. |
| `velocity_x`, `velocity_y` | The velocity it ends the step with. |
| `displacement_x`, `displacement_y` | How far it travelled during the step, measured across a seam rather than by subtracting wrapped coordinates. |

A reader replays the commands from column one, starting at the scenario's
`start_position` and `start_velocity`, and has to reproduce the other eight.

## The scenarios

| File | What it pins |
|---|---|
| `rest.bin` | No command from rest: nothing accelerates, nothing drifts. |
| `acceleration.bin` | The exponential ramp from rest, and the integrated travel during it. |
| `braking.bin` | Coasting after the command is released — the curve a stopping-offset lookahead has to predict. |
| `reversal.bin` | Full speed one way, then the opposite command. The hardest case for a predictor that assumes velocity follows the command at once. |
| `diagonal.bin` | `(1, 1)` has length root two; normalisation makes it travel at an axis's speed, not faster. |
| `off-compass.bin` | Fifteen degrees — a heading the old nine-way action space could not express. |
| `off-compass-negative.bin` | Two hundred and fifty-five degrees, so both components are negative and a dropped sign cannot pass. |
| `over-unit-command.bin` | A command five times unit length, which must match `acceleration.bin` step for step: length is a heading's magnitude, not a throttle. |
| `idle-floor.bin` | A command length that rises through the idle floor and falls back through it, so the boundary is crossed in both directions. |
| `seam-x.bin`, `seam-y.bin` | Leaving one edge and arriving at the opposite one. The two extents differ, so an axis mix-up lands somewhere else. |
| `long-trace.bin` | Six hundred steps on a continuously turning heading, for the accumulated-error bound. A formula that is nearly right passes every single step and arrives somewhere else. |

## Tolerances

Python reproduces the recorded **intent, position and velocity exactly** -- bit
for bit, at every step of every scenario. The two sides run the same arithmetic
in the same order at the same precision, and `numpy`'s float32 `exp` agrees with
Rust's to the last bit, so there is nothing left to round differently.

**Displacement** is the exception, and deliberately: Rust measures it from the
positions it produced and Python computes it from the velocity curve, so the two
agree only to float32's granularity at the coordinates involved. The bound is
**1e-4 reference pixels**; the measured worst case is 5.6e-5, in the seam
scenarios, where coordinates near three thousand world units make a single unit
in the last place worth 4e-5 reference pixels already.

A failure is a disagreement to investigate, not a number to raise.

Both sides work in `float32` and copy each other's operation order — a two-term
lerp for the velocity blend, a multiply by a reciprocal for the normalisation.
Those spellings are algebraically interchangeable and numerically are not, and
matching them costs nothing while making a real disagreement visible instead of
hiding it under the noise of a rewrite.

## What checks these

* [`tests/motion_fixtures.rs`](../../motion_fixtures.rs) replays every scenario
  through `advance_motion` and compares the contract against this build.
* `training/tests/test_motion_reference.py` replays them through the Python
  reference, which is the copy the trainer actually uses.
