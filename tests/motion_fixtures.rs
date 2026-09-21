//! Read the committed player-motion fixtures back against this build.
//!
//! The fixtures exist for the Python trainer, which predicts player
//! displacement to align its own spatial state and reads these without a Rust
//! toolchain. That makes them a one-way promise unless something on this side
//! checks it, and the promise has two halves.
//!
//! The first is that the recorded motion is still the motion this build
//! produces. Replaying every scenario through `advance_motion` catches a
//! fixture set that a gameplay change left behind, which would otherwise show
//! up as Python failing against numbers no Rust test had looked at.
//!
//! The second is the constants themselves. A stale `MOTION_CONTRACT_ID` is the
//! dangerous case: the numbers would still be self-consistent, Python would
//! still agree with them, and both would be describing a game that no longer
//! exists. So the manifest's contract is compared against the live one field by
//! field, and a changed constant fails here rather than in a training run.
//!
//! Regenerate with:
//! `cargo run --locked --no-default-features --example motion_fixtures`

// clippy.toml allows these inside tests, but its detection only reaches
// #[test] functions; the helpers below are ordinary functions in a test-only
// binary, where a failed expectation is exactly the intended failure.
#![expect(
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    reason = "Test-only binary: a failed expectation is how a test reports, and a fixture value must match exactly rather than closely"
)]

use std::fs;
use std::path::PathBuf;

use bevy::math::Vec2;
use serde::Deserialize;

use dodge_royale::motion::advance_motion;
use dodge_royale::scale::WORLD_HALF_EXTENTS;
use dodge_royale::simulation::{
    MOTION_CONTRACT_ID, MotionContract, intent_from_command, motion_contract,
};
use dodge_royale::torus::wrapped_delta;

/// Values recorded per step. Kept here rather than read from the manifest, so
/// a manifest that renamed or dropped one is a failure and not a shrug.
const COLUMNS: usize = 10;

#[derive(Deserialize)]
struct Manifest {
    contract: MotionContract,
    contract_bits: Bits,
    record: Vec<String>,
    step_bytes: usize,
    scenarios: Vec<Scenario>,
}

#[derive(Deserialize)]
struct Bits {
    id: String,
    top_speed: u32,
    movement_response: u32,
    max_frame_seconds: u32,
    idle_threshold: u32,
    seconds_per_frame: u32,
    world_units_per_pixel: u32,
    world_half_extents: [u32; 2],
}

#[derive(Deserialize)]
struct Scenario {
    name: String,
    file: String,
    steps: usize,
    bytes: usize,
    crosses_seam: bool,
    start_position: [Sample; 2],
    start_velocity: [Sample; 2],
}

#[derive(Deserialize)]
struct Sample {
    value: f64,
    bits: u32,
}

impl Sample {
    fn as_f32(&self) -> f32 {
        let from_bits = f32::from_bits(self.bits);
        assert_eq!(
            f64::from(from_bits),
            self.value,
            "a sample's decimal and its bits disagree"
        );
        from_bits
    }
}

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/player-motion")
}

fn manifest() -> Manifest {
    let path = root().join("manifest.json");
    let text = fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("reading {}: {error}", path.display()));
    serde_json::from_str(&text)
        .unwrap_or_else(|error| panic!("parsing {}: {error}", path.display()))
}

/// One scenario's recorded steps, as rows of [`COLUMNS`] floats.
fn steps_of(scenario: &Scenario) -> Vec<[f32; COLUMNS]> {
    let path = root().join(&scenario.file);
    let bytes =
        fs::read(&path).unwrap_or_else(|error| panic!("reading {}: {error}", path.display()));
    assert_eq!(
        bytes.len(),
        scenario.bytes,
        "{} changed size",
        scenario.file
    );
    assert_eq!(
        bytes.len(),
        scenario.steps.saturating_mul(COLUMNS.saturating_mul(4)),
        "{} is not a whole number of steps",
        scenario.file
    );

    let mut rows = Vec::with_capacity(scenario.steps);
    for row in bytes.chunks_exact(COLUMNS.saturating_mul(4)) {
        let mut values = [0.0_f32; COLUMNS];
        for (slot, word) in values.iter_mut().zip(row.chunks_exact(4)) {
            let quad: [u8; 4] = word.try_into().expect("four bytes make an f32");
            *slot = f32::from_le_bytes(quad);
        }
        rows.push(values);
    }
    rows
}

#[test]
fn the_manifest_describes_this_builds_movement_rules() {
    let manifest = manifest();
    let live = motion_contract();
    assert_eq!(
        manifest.contract.id, live.id,
        "the fixtures were made under a different motion contract; regenerate them"
    );
    assert_eq!(manifest.contract, live, "a movement constant has changed");
}

#[test]
fn every_recorded_constant_carries_the_same_value_twice() {
    let manifest = manifest();
    let bits = &manifest.contract_bits;
    let contract = &manifest.contract;
    assert_eq!(bits.id, MOTION_CONTRACT_ID);
    assert_eq!(f32::from_bits(bits.top_speed), contract.top_speed);
    assert_eq!(
        f32::from_bits(bits.movement_response),
        contract.movement_response
    );
    assert_eq!(
        f32::from_bits(bits.max_frame_seconds),
        contract.max_frame_seconds
    );
    assert_eq!(f32::from_bits(bits.idle_threshold), contract.idle_threshold);
    assert_eq!(
        f32::from_bits(bits.seconds_per_frame),
        contract.seconds_per_frame
    );
    assert_eq!(
        f32::from_bits(bits.world_units_per_pixel),
        contract.world_units_per_pixel
    );
    assert_eq!(
        [
            f32::from_bits(bits.world_half_extents[0]),
            f32::from_bits(bits.world_half_extents[1]),
        ],
        contract.world_half_extents
    );
}

#[test]
fn the_record_layout_is_the_one_readers_expect() {
    let manifest = manifest();
    assert_eq!(manifest.record.len(), COLUMNS);
    assert_eq!(manifest.step_bytes, COLUMNS.saturating_mul(4));
    assert_eq!(
        manifest.record,
        [
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
        ]
    );
}

#[test]
fn every_committed_scenario_replays_to_what_was_recorded() {
    let manifest = manifest();
    let seconds = manifest.contract.seconds_per_frame;
    assert!(
        !manifest.scenarios.is_empty(),
        "no scenarios were committed"
    );

    for scenario in &manifest.scenarios {
        let rows = steps_of(scenario);
        let mut position = Vec2::new(
            scenario.start_position[0].as_f32(),
            scenario.start_position[1].as_f32(),
        );
        let mut velocity = Vec2::new(
            scenario.start_velocity[0].as_f32(),
            scenario.start_velocity[1].as_f32(),
        );

        for (index, row) in rows.iter().enumerate() {
            let command = Vec2::new(row[0], row[1]);
            let intent = intent_from_command(command);
            let before = position;
            position = advance_motion(position, &mut velocity, intent, seconds);
            let displacement = wrapped_delta(before, position, WORLD_HALF_EXTENTS);

            let expected = [
                row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9],
            ];
            let produced = [
                intent.x,
                intent.y,
                position.x,
                position.y,
                velocity.x,
                velocity.y,
                displacement.x,
                displacement.y,
            ];
            assert_eq!(
                produced, expected,
                "{} step {index} no longer replays to what was recorded",
                scenario.name
            );
        }
    }
}

#[test]
fn the_seam_scenarios_actually_leave_the_world() {
    // A scenario that claimed to cross a seam but never reached one would be a
    // fixture that tests nothing, and would keep passing after the wrap broke.
    let manifest = manifest();
    let crossing: Vec<&str> = manifest
        .scenarios
        .iter()
        .filter(|scenario| scenario.crosses_seam)
        .map(|scenario| scenario.name.as_str())
        .collect();
    assert!(
        crossing.contains(&"seam-x") && crossing.contains(&"seam-y"),
        "both seams must be covered, got {crossing:?}"
    );

    for scenario in manifest.scenarios.iter().filter(|s| s.crosses_seam) {
        let rows = steps_of(scenario);
        let travelled = rows
            .iter()
            .map(|row| Vec2::new(row[8], row[9]).length())
            .fold(0.0_f32, f32::max);
        let limit = manifest.contract.top_speed * manifest.contract.seconds_per_frame * 1.01;
        assert!(
            travelled <= limit,
            "{} travelled {travelled} world units in one step, over the {limit} a frame allows",
            scenario.name
        );
        let jump = rows
            .windows(2)
            .map(|pair| {
                let a = Vec2::new(pair[0][4], pair[0][5]);
                let b = Vec2::new(pair[1][4], pair[1][5]);
                (b - a).length()
            })
            .fold(0.0_f32, f32::max);
        assert!(
            jump > limit,
            "{} never wrapped: its coordinates never jumped",
            scenario.name
        );
    }
}
