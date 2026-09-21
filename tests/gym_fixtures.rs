//! Read the committed protocol-v2 fixtures back with the shipped codec.
//!
//! The fixtures exist for the Python client, which reads them without a Rust
//! toolchain. That makes them a one-way promise unless something on this side
//! checks it: a codec change that altered the format would leave Python
//! failing against bytes no Rust test had looked at. These read every
//! committed file with the real readers and check it against the manifest the
//! emitter wrote beside it.
//!
//! This is not the same check as the hand-written wire images in
//! `src/gym/protocol/tests.rs`. Those state what the bytes must be, from first
//! principles, and would catch a codec and a fixture that drifted together.
//! These state that what is committed still parses and still means what the
//! manifest says. Both are wanted; neither replaces the other.
//!
//! Regenerate with:
//! `cargo run --locked --no-default-features --example gym_fixtures`

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

use serde::Deserialize;

use dodge_royale::gym::protocol::{
    MAGIC, PROTOCOL_VERSION, max_payload, read_handshake, read_reset, read_step,
};
use dodge_royale::observation::{Layout, OBSERVATION_VALUES, layout as build_layout};

/// Values a whole batch of `envs` observations holds.
fn values_in(envs: u32) -> usize {
    usize::try_from(envs)
        .unwrap_or(usize::MAX)
        .saturating_mul(OBSERVATION_VALUES)
}

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gym-v2")
}

fn manifest() -> Manifest {
    let path = fixture_dir().join("manifest.json");
    let json = fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("reading {}: {error}", path.display()));
    serde_json::from_str(&json).expect("the manifest parses")
}

/// Every value the manifest recorded, checked against the decoded observation.
///
/// Both the decimal and the bit pattern, because they are what the Python side
/// compares and a fixture that satisfies only one of them would let a decoder
/// through that agrees with the manifest but not with the bytes.
fn check_samples(samples: &[Sample], observations: &[f32], file: &str, section: &str) {
    for sample in samples {
        let index = usize::try_from(sample.env)
            .ok()
            .and_then(|env| env.checked_mul(OBSERVATION_VALUES))
            .and_then(|base| base.checked_add(sample.index))
            .expect("a sample index inside the batch");
        let value = *observations
            .get(index)
            .unwrap_or_else(|| panic!("{file}: {section} sample {index} is past the batch"));
        assert!(
            f64::from(value) == sample.value,
            "{file}: {section} value at {index} is {value}, manifest says {}",
            sample.value
        );
        assert_eq!(
            value.to_bits(),
            sample.bits,
            "{file}: {section} bits at {index} disagree"
        );
    }
}

#[test]
fn the_committed_fixtures_still_parse_and_match_their_manifest() {
    let manifest = manifest();
    assert_eq!(manifest.protocol_version, PROTOCOL_VERSION);
    assert_eq!(manifest.observation_values, OBSERVATION_VALUES);

    let mut seen = 0_usize;
    for fixture in &manifest.fixtures {
        let path = fixture_dir().join(&fixture.file);
        let bytes =
            fs::read(&path).unwrap_or_else(|error| panic!("reading {}: {error}", path.display()));
        assert_eq!(
            bytes.len(),
            fixture.bytes,
            "{}: the manifest records a different length",
            fixture.file
        );
        let maximum = max_payload(fixture.envs, OBSERVATION_VALUES);

        match fixture.kind.as_str() {
            "handshake" => {
                assert_eq!(bytes.get(..8), Some(MAGIC.as_slice()));
                let handshake =
                    read_handshake(&mut bytes.as_slice()).expect("the handshake parses");
                assert_eq!(handshake.envs, fixture.envs);
                assert_eq!(
                    handshake.layout, manifest.layout,
                    "the handshake announces a different layout than the manifest records"
                );
            }
            "reset" => {
                let batch = read_reset(&mut bytes.as_slice(), maximum).expect("the reset parses");
                assert_eq!(batch.seeds, fixture.episode_seeds);
                assert_eq!(batch.observations.len(), values_in(fixture.envs));
                check_samples(
                    &fixture.samples,
                    &batch.observations,
                    &fixture.file,
                    "observation",
                );
            }
            "step" => {
                let batch = read_step(&mut bytes.as_slice(), maximum).expect("the step parses");
                assert_eq!(
                    batch.transitions.len(),
                    usize::try_from(fixture.envs).unwrap_or(usize::MAX)
                );
                for (index, expected) in fixture.transitions.iter().enumerate() {
                    let got = batch.transitions.get(index).expect("a transition");
                    assert_eq!(u32::try_from(index).unwrap_or(u32::MAX), expected.env);
                    assert_eq!(got.frame, expected.frame, "{}", fixture.file);
                    assert_eq!(got.terminated, expected.terminated, "{}", fixture.file);
                    assert_eq!(got.truncated, expected.truncated, "{}", fixture.file);
                    assert_eq!(got.enemy_deaths, expected.enemy_deaths, "{}", fixture.file);
                    assert_eq!(got.episode_seed, expected.episode_seed, "{}", fixture.file);
                    assert_eq!(got.reset_seed, expected.reset_seed, "{}", fixture.file);
                }
                let terminals: Vec<u32> = batch.terminal.iter().map(|end| end.env).collect();
                assert_eq!(terminals, fixture.terminal_envs, "{}", fixture.file);
                check_samples(
                    &fixture.samples,
                    &batch.observations,
                    &fixture.file,
                    "observation",
                );
                for end in &batch.terminal {
                    let mine: Vec<Sample> = fixture
                        .terminal_samples
                        .iter()
                        .filter(|sample| sample.env == end.env)
                        .map(|sample| Sample {
                            // The terminal array is one env's own, so the
                            // batch stride the sample was recorded with does
                            // not apply to it.
                            env: 0,
                            ..*sample
                        })
                        .collect();
                    check_samples(&mine, &end.observation, &fixture.file, "terminal");
                }
            }
            other => panic!("{}: unknown fixture kind {other}", fixture.file),
        }
        seen += 1;
    }
    assert!(seen >= 5, "the manifest should describe every fixture file");
}

#[test]
fn the_fixtures_describe_the_layout_this_build_produces() {
    // A fixture set that no longer matches the encoder is worse than none: the
    // Python side would validate against a layout this binary never sends. If
    // this fails, regenerate the fixtures and bump what the change requires.
    let manifest = manifest();
    assert_eq!(
        manifest.layout,
        build_layout(manifest.layout.hold_frames),
        "the committed fixtures were emitted by a different observation layout"
    );
}

#[test]
fn every_fixture_file_is_described_by_the_manifest() {
    // Catches a file added or renamed without the manifest following it, which
    // Python would silently never read.
    let manifest = manifest();
    for entry in fs::read_dir(fixture_dir()).expect("the fixture directory exists") {
        let entry = entry.expect("a directory entry");
        let name = entry.file_name().to_string_lossy().into_owned();
        if !name.to_ascii_lowercase().ends_with(".bin") {
            continue;
        }
        assert!(
            manifest.fixtures.iter().any(|fixture| fixture.file == name),
            "{name} is committed but the manifest does not describe it"
        );
    }
}

// --- the manifest, as this side reads it --------------------------------

#[derive(Deserialize)]
struct Manifest {
    protocol_version: u32,
    observation_values: usize,
    layout: Layout,
    fixtures: Vec<Fixture>,
}

#[derive(Deserialize)]
struct Fixture {
    file: String,
    kind: String,
    bytes: usize,
    envs: u32,
    #[serde(default)]
    episode_seeds: Vec<u64>,
    #[serde(default)]
    transitions: Vec<TransitionExpectation>,
    #[serde(default)]
    terminal_envs: Vec<u32>,
    #[serde(default)]
    samples: Vec<Sample>,
    #[serde(default)]
    terminal_samples: Vec<Sample>,
}

#[derive(Deserialize)]
struct TransitionExpectation {
    env: u32,
    frame: u32,
    terminated: bool,
    truncated: bool,
    enemy_deaths: u32,
    episode_seed: u64,
    reset_seed: Option<u64>,
}

#[derive(Deserialize, Clone, Copy)]
struct Sample {
    env: u32,
    index: usize,
    value: f64,
    bits: u32,
}
