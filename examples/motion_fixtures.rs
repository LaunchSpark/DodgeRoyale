//! Emit the committed player-motion fixtures under `tests/fixtures/player-motion/`.
//!
//! The Python trainer has to predict where the player will be in order to align
//! its own spatial state, and nothing in the gym wire format carries the
//! movement constants. So the two implementations agree only if something makes
//! them: these are that something. Rust produces the expected positions,
//! velocities and displacements; Python replays the same commands and has to
//! land on the same numbers.
//!
//! Regenerate after an intentional change to the movement rules:
//!
//! ```sh
//! cargo run --locked --no-default-features --example motion_fixtures
//! ```
//!
//! The expected values are never regenerated from the Python formula under
//! test. A fixture derived from the thing it checks agrees with it by
//! construction, and would go on agreeing through a shared mistake.
//!
//! Nothing here restates the movement arithmetic either. Positions and
//! velocities come from `advance_motion` itself, and each step's displacement
//! is measured from the positions it produced. Python arrives at that same
//! displacement from the velocity curve instead, so the two sides reach the
//! recorded number by different routes.

use std::fs;
use std::path::{Path, PathBuf};

use bevy::math::Vec2;
use serde::Serialize;

use dodge_royale::motion::advance_motion;
use dodge_royale::scale::WORLD_HALF_EXTENTS;
use dodge_royale::simulation::{
    IDLE_THRESHOLD, MOTION_CONTRACT_ID, MotionContract, intent_from_command, motion_contract,
};
use dodge_royale::torus::wrapped_delta;

/// Where the fixtures live, relative to the manifest directory.
const FIXTURE_DIR: &str = "tests/fixtures/player-motion";

/// Values recorded per step, in this order. Ten `f32`, little endian.
const RECORD: [&str; 10] = [
    "command_x",
    "command_y",
    "intent_x",
    "intent_y",
    "position_x",
    "position_y",
    "velocity_x",
    "velocity_y",
    "displacement_x",
    "displacement_y",
];

/// Bytes one recorded step occupies.
const STEP_BYTES: usize = RECORD.len().saturating_mul(4);

/// How a scenario chooses its command on each step.
enum Commands {
    /// The same command every step.
    Held(Vec2),
    /// One command, then another from `at` onward.
    Switch { first: Vec2, then: Vec2, at: usize },
    /// A command whose length rises through the idle floor and falls back.
    Sweep { heading: Vec2, peak: f32 },
    /// A heading that turns a fixed number of degrees each step.
    Turning { degrees_per_step: f32 },
}

impl Commands {
    #[expect(
        clippy::arithmetic_side_effects,
        reason = "Bounded step counts and finite headings"
    )]
    fn at(&self, step: usize, steps: usize) -> Vec2 {
        match *self {
            Self::Held(command) => command,
            Self::Switch { first, then, at } => {
                if step < at {
                    first
                } else {
                    then
                }
            }
            Self::Sweep { heading, peak } => {
                // A triangle wave: up to `peak` and back down, so the idle floor
                // is crossed in both directions rather than only upward.
                let half = steps.max(2) / 2;
                let rise = if step < half {
                    ratio(step, half)
                } else {
                    1.0 - ratio(step.saturating_sub(half), steps.saturating_sub(half))
                };
                heading * (peak * rise)
            }
            Self::Turning { degrees_per_step } => {
                Vec2::from_angle((degrees_per_step * count(step)).to_radians())
            }
        }
    }
}

/// A step index as a float, without an `as` cast.
fn count(step: usize) -> f32 {
    f32::from(u16::try_from(step).unwrap_or(u16::MAX))
}

/// `numerator / denominator`, with a denominator of at least one.
fn ratio(numerator: usize, denominator: usize) -> f32 {
    count(numerator) / count(denominator.max(1))
}

struct Scenario {
    name: &'static str,
    file: &'static str,
    purpose: &'static str,
    steps: usize,
    start_position: Vec2,
    start_velocity: Vec2,
    commands: Commands,
}

/// The cases the two implementations have to agree on.
///
/// One list rather than several, because what matters about it is the coverage
/// as a whole: every scenario here is one way the motion model can be got
/// subtly wrong, and a reader checking that the set is complete wants them in
/// front of them rather than spread across helpers.
#[expect(
    clippy::too_many_lines,
    reason = "A table of cases, and a table reads better whole"
)]
fn scenarios() -> Vec<Scenario> {
    // Headings at fifteen and two hundred and fifty-five degrees. Neither is on
    // the nine-way compass the observation's candidate paths use, so a reader
    // that quietly snapped a command to the nearest of them would travel
    // somewhere else, and fail here rather than in a training run.
    let fifteen = Vec2::from_angle(15.0_f32.to_radians());
    let two_fifty_five = Vec2::from_angle(255.0_f32.to_radians());
    let near_x = WORLD_HALF_EXTENTS.x - 20.0;
    let near_y = WORLD_HALF_EXTENTS.y - 20.0;

    vec![
        Scenario {
            name: "rest",
            file: "rest.bin",
            purpose: "No command from rest: nothing accelerates and nothing drifts.",
            steps: 60,
            start_position: Vec2::ZERO,
            start_velocity: Vec2::ZERO,
            commands: Commands::Held(Vec2::ZERO),
        },
        Scenario {
            name: "acceleration",
            file: "acceleration.bin",
            purpose: "Rest to top speed along +x. Pins the exponential ramp, and \
                      the integrated displacement during it.",
            steps: 120,
            start_position: Vec2::ZERO,
            start_velocity: Vec2::ZERO,
            commands: Commands::Held(Vec2::X),
        },
        Scenario {
            name: "braking",
            file: "braking.bin",
            purpose: "Accelerate, then release. Coasting is the same curve run \
                      toward a zero target, and it is what a stopping-offset \
                      lookahead has to predict.",
            steps: 120,
            start_position: Vec2::ZERO,
            start_velocity: Vec2::ZERO,
            commands: Commands::Switch {
                first: Vec2::X,
                then: Vec2::ZERO,
                at: 60,
            },
        },
        Scenario {
            name: "reversal",
            file: "reversal.bin",
            purpose: "Full speed one way, then the opposite command. The hardest \
                      case for a predictor that assumes velocity follows the \
                      command immediately.",
            steps: 120,
            start_position: Vec2::ZERO,
            start_velocity: Vec2::ZERO,
            commands: Commands::Switch {
                first: Vec2::X,
                then: Vec2::NEG_X,
                at: 60,
            },
        },
        Scenario {
            name: "diagonal",
            file: "diagonal.bin",
            purpose: "The command (1, 1) has length root two. Normalisation makes \
                      it travel at an axis's speed, not faster.",
            steps: 90,
            start_position: Vec2::ZERO,
            start_velocity: Vec2::ZERO,
            commands: Commands::Held(Vec2::ONE),
        },
        Scenario {
            name: "off-compass",
            file: "off-compass.bin",
            purpose: "Fifteen degrees: a heading the nine-way action space could \
                      not express.",
            steps: 90,
            start_position: Vec2::ZERO,
            start_velocity: Vec2::ZERO,
            commands: Commands::Held(fifteen),
        },
        Scenario {
            name: "off-compass-negative",
            file: "off-compass-negative.bin",
            purpose: "Two hundred and fifty-five degrees, so both components are \
                      negative and a dropped sign cannot pass.",
            steps: 90,
            start_position: Vec2::ZERO,
            start_velocity: Vec2::ZERO,
            commands: Commands::Held(two_fifty_five),
        },
        Scenario {
            name: "over-unit-command",
            file: "over-unit-command.bin",
            purpose: "A command five times unit length. Length sets no speed, so \
                      this matches `acceleration` step for step.",
            steps: 120,
            start_position: Vec2::ZERO,
            start_velocity: Vec2::ZERO,
            commands: Commands::Held(Vec2::new(5.0, 0.0)),
        },
        Scenario {
            name: "idle-floor",
            file: "idle-floor.bin",
            purpose: "A command whose length rises through the idle floor and \
                      falls back through it. Below the floor the intent is zero \
                      and the player coasts; above it the full target speed \
                      applies, however short the command.",
            steps: 120,
            start_position: Vec2::ZERO,
            start_velocity: Vec2::ZERO,
            commands: Commands::Sweep {
                heading: Vec2::X,
                peak: IDLE_THRESHOLD * 2.0,
            },
        },
        Scenario {
            name: "seam-x",
            file: "seam-x.bin",
            purpose: "Driving off the +x edge and arriving at the -x one. The \
                      displacement stays small; only the coordinate jumps.",
            steps: 90,
            start_position: Vec2::new(near_x, 0.0),
            start_velocity: Vec2::ZERO,
            commands: Commands::Held(Vec2::X),
        },
        Scenario {
            name: "seam-y",
            file: "seam-y.bin",
            purpose: "The same across the +y edge. The two extents differ, so an \
                      axis mix-up lands somewhere else entirely.",
            steps: 90,
            start_position: Vec2::new(0.0, near_y),
            start_velocity: Vec2::ZERO,
            commands: Commands::Held(Vec2::Y),
        },
        Scenario {
            name: "long-trace",
            file: "long-trace.bin",
            purpose: "Six hundred steps on a continuously turning heading, for the \
                      accumulated-error bound. Every step compounds, so a formula \
                      that is nearly right fails here and nowhere else.",
            steps: 600,
            start_position: Vec2::ZERO,
            start_velocity: Vec2::ZERO,
            commands: Commands::Turning {
                degrees_per_step: 1.7,
            },
        },
    ]
}

fn main() -> Result<(), Box<dyn core::error::Error>> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(FIXTURE_DIR);
    fs::create_dir_all(&root)?;

    let contract = motion_contract();
    let seconds = contract.seconds_per_frame;
    let mut emitted = Vec::new();
    for scenario in scenarios() {
        emitted.push(emit(&root, &scenario, seconds)?);
    }

    let manifest = Manifest {
        contract_bits: Bits::of(&contract),
        contract,
        regenerate: "cargo run --locked --no-default-features --example motion_fixtures",
        comparison: "Every `value` is the f32 widened to f64, so a reader that parses JSON \
                     numbers as doubles compares equal exactly. `bits` is the same value's \
                     little-endian u32, for a reader that would rather not trust that.",
        record: RECORD,
        step_bytes: STEP_BYTES,
        note: "Each step's `position` and `velocity` are the values after that step. \
               `displacement` is how far the player travelled during it, measured \
               across a seam rather than by subtracting wrapped coordinates. \
               `intent` is the command after the idle floor.",
        scenarios: emitted,
    };

    let path = root.join("manifest.json");
    let mut json = serde_json::to_string_pretty(&manifest)?;
    json.push('\n');
    fs::write(&path, json)?;
    println!("wrote {}", path.display());
    Ok(())
}

/// Run one scenario and write its steps.
fn emit(
    root: &Path,
    scenario: &Scenario,
    seconds: f32,
) -> Result<Emitted, Box<dyn core::error::Error>> {
    let mut position = scenario.start_position;
    let mut velocity = scenario.start_velocity;
    let mut bytes = Vec::with_capacity(scenario.steps.saturating_mul(STEP_BYTES));
    let mut crosses_seam = false;

    for step in 0..scenario.steps {
        let command = scenario.commands.at(step, scenario.steps);
        let intent = intent_from_command(command);
        let before = position;
        position = advance_motion(position, &mut velocity, intent, seconds);
        // How far the player travelled, which a seam crossing must not change.
        // Subtracting the wrapped coordinates would report a whole world's
        // width, which is exactly the mistake these fixtures exist to catch.
        let displacement = wrapped_delta(before, position, WORLD_HALF_EXTENTS);
        crosses_seam = crosses_seam || seam_crossed(before, position, displacement);

        for value in [
            command.x,
            command.y,
            intent.x,
            intent.y,
            position.x,
            position.y,
            velocity.x,
            velocity.y,
            displacement.x,
            displacement.y,
        ] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }

    let path = root.join(scenario.file);
    fs::write(&path, &bytes)?;
    println!("wrote {} ({} bytes)", path.display(), bytes.len());

    Ok(Emitted {
        name: scenario.name,
        file: scenario.file,
        purpose: scenario.purpose,
        steps: scenario.steps,
        bytes: bytes.len(),
        crosses_seam,
        start_position: pair(scenario.start_position),
        start_velocity: pair(scenario.start_velocity),
    })
}

/// Whether this step left one edge and arrived at the opposite one.
#[expect(
    clippy::arithmetic_side_effects,
    reason = "Two finite positions and their finite difference"
)]
fn seam_crossed(before: Vec2, after: Vec2, travelled: Vec2) -> bool {
    // Straight subtraction disagrees with the travelled distance only when the
    // coordinate wrapped, and then it disagrees by most of a world.
    (after - before - travelled).length() > 1.0
}

fn pair(value: Vec2) -> [Sample; 2] {
    [Sample::of(value.x), Sample::of(value.y)]
}

#[derive(Serialize)]
struct Manifest {
    contract: MotionContract,
    contract_bits: Bits,
    regenerate: &'static str,
    comparison: &'static str,
    record: [&'static str; 10],
    step_bytes: usize,
    note: &'static str,
    scenarios: Vec<Emitted>,
}

/// The contract's floats as bit patterns, for a reader that would rather not
/// trust a decimal at all.
#[derive(Serialize)]
struct Bits {
    id: &'static str,
    top_speed: u32,
    movement_response: u32,
    max_frame_seconds: u32,
    idle_threshold: u32,
    seconds_per_frame: u32,
    world_units_per_pixel: u32,
    world_half_extents: [u32; 2],
}

impl Bits {
    const fn of(contract: &MotionContract) -> Self {
        Self {
            id: MOTION_CONTRACT_ID,
            top_speed: contract.top_speed.to_bits(),
            movement_response: contract.movement_response.to_bits(),
            max_frame_seconds: contract.max_frame_seconds.to_bits(),
            idle_threshold: contract.idle_threshold.to_bits(),
            seconds_per_frame: contract.seconds_per_frame.to_bits(),
            world_units_per_pixel: contract.world_units_per_pixel.to_bits(),
            world_half_extents: [
                contract.world_half_extents[0].to_bits(),
                contract.world_half_extents[1].to_bits(),
            ],
        }
    }
}

#[derive(Serialize)]
struct Emitted {
    name: &'static str,
    file: &'static str,
    purpose: &'static str,
    steps: usize,
    bytes: usize,
    /// Whether the player left one edge and arrived at the opposite one.
    crosses_seam: bool,
    start_position: [Sample; 2],
    start_velocity: [Sample; 2],
}

#[derive(Serialize)]
struct Sample {
    /// The `f32` widened to `f64`, which is exact and survives JSON.
    value: f64,
    /// The same value's bit pattern.
    bits: u32,
}

impl Sample {
    fn of(value: f32) -> Self {
        Self {
            value: f64::from(value),
            bits: value.to_bits(),
        }
    }
}
