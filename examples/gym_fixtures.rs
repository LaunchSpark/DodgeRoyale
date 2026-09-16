//! Emit the committed protocol-v1 fixtures under `tests/fixtures/gym-v1/`.
//!
//! The fixtures exist so the Python client can be tested against real protocol
//! bytes without a Rust toolchain, a build, or a running gym. They are checked
//! in; this example regenerates them:
//!
//! ```sh
//! cargo run --locked --no-default-features --example gym_fixtures
//! ```
//!
//! Two kinds of message are emitted, for two different reasons.
//!
//! The **session** fixtures come from a real `ArenaBatch` at a fixed seed, so
//! they are exactly what the server sends: a handshake, frame zero, an
//! ordinary step, and the step where both episodes hit the frame budget and
//! are replaced. That last one is the auto-reset shape -- terminal
//! observations, replacement seeds, and both `terminated` and `truncated` --
//! which is the part of the protocol a client is most likely to get wrong.
//!
//! The **scene** fixtures encode hand-placed arenas instead. A real episode
//! cannot be relied on to contain a half-grown blast or a hazard straddling
//! the seam on a chosen frame, and the encoder's coordinate contract is
//! exactly where a decoder's axes and signs go wrong, so those cases are
//! placed deliberately rather than waited for.
//!
//! Regeneration is a maintenance action, not a test. The committed bytes are
//! what both sides read, so neither side depends on this example running, nor
//! on floating point agreeing across machines.

// A dev-only generator. A failed expectation stops it, which is the intended
// reporting, and the arithmetic is over indices the loops just produced.
#![expect(
    clippy::expect_used,
    clippy::arithmetic_side_effects,
    clippy::suboptimal_flops,
    reason = "Dev-only generator: a failed expectation is how it reports, the maths is indexing, and a scene reads better as plain coordinates than as mul_add"
)]

use std::fs;
use std::path::{Path, PathBuf};

use bevy::ecs::entity::Entity;
use bevy::math::Vec2;
use serde::Serialize;

use dodge_royale::collision::Collider;
use dodge_royale::enemy_types::EnemyKind;
use dodge_royale::gym::protocol::{
    EnvTransition, Handshake, MAGIC, PROTOCOL_VERSION, ResetBatch, StepBatch, TerminalObservation,
    write_handshake, write_reset, write_step,
};
use dodge_royale::gym::workers::{ArenaBatch, BatchConfig};
use dodge_royale::observation::{
    Layout, OBSERVATION_VALUES, PLAYER_VALUES, encode, layout as build_layout,
};
use dodge_royale::scale::{PIXEL, WORLD_HALF_EXTENTS};
use dodge_royale::simulation::{ArenaView, BlastView, DEFAULT_HOLD_FRAMES, EnemyView, PlayerView};

/// Where the fixtures live, relative to the manifest directory.
const FIXTURE_DIR: &str = "tests/fixtures/gym-v1";

/// The session the real-arena fixtures come from. Small on purpose: an
/// observation is 115,128 bytes, so every extra env is another 115 KB in git.
const SESSION: BatchConfig = BatchConfig {
    envs: 2,
    workers: 1,
    root_seed: 7,
    enemy_count: 12,
    max_frames: 3,
    hold_frames: DEFAULT_HOLD_FRAMES,
};

/// Non-zero observation values recorded per env, beyond the ones always taken.
const GRID_SAMPLES: usize = 48;

fn main() -> Result<(), Box<dyn core::error::Error>> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(FIXTURE_DIR);
    fs::create_dir_all(&root)?;

    let layout = build_layout(SESSION.hold_frames);
    let mut fixtures = Vec::new();

    fixtures.push(emit_handshake(&root, &layout)?);
    fixtures.extend(emit_session(&root, &layout)?);
    fixtures.push(emit_scenes(&root, &layout)?);

    let manifest = Manifest {
        protocol_version: PROTOCOL_VERSION,
        magic: hex(&MAGIC),
        regenerate: "cargo run --locked --no-default-features --example gym_fixtures",
        comparison: "Every `value` is the f32 widened to f64, so a reader that parses JSON \
                     numbers as doubles compares equal exactly. `bits` is the same value's \
                     little-endian u32, for a reader that would rather not trust that.",
        observation_values: OBSERVATION_VALUES,
        layout,
        session: Session {
            envs: SESSION.envs,
            root_seed: SESSION.root_seed,
            enemy_count: SESSION.enemy_count,
            max_frames: SESSION.max_frames,
            hold_frames: SESSION.hold_frames,
        },
        fixtures,
    };

    let path = root.join("manifest.json");
    let mut json = serde_json::to_string_pretty(&manifest)?;
    json.push('\n');
    fs::write(&path, json)?;
    println!("wrote {}", path.display());
    Ok(())
}

// --- the real session ---------------------------------------------------

/// The opening announcement, exactly as `serve` writes it.
fn emit_handshake(root: &Path, layout: &Layout) -> Result<Fixture, Box<dyn core::error::Error>> {
    let handshake = Handshake {
        protocol_version: PROTOCOL_VERSION,
        envs: SESSION.envs,
        enemy_count: u32::try_from(SESSION.enemy_count)?,
        max_frames: SESSION.max_frames,
        root_seed: SESSION.root_seed,
        workers: SESSION.workers,
        layout: layout.clone(),
    };
    let mut bytes = Vec::new();
    write_handshake(&mut bytes, &handshake)?;
    Ok(Fixture {
        purpose: "The opening: magic, version, opcode 0x81, then the layout as JSON. \
                  A client that cannot match this layout must fail here.",
        envs: SESSION.envs,
        ..write_fixture(root, "handshake.bin", "handshake", &bytes)?
    })
}

/// Frame zero, an ordinary step, and the step that replaces both episodes.
fn emit_session(root: &Path, layout: &Layout) -> Result<Vec<Fixture>, Box<dyn core::error::Error>> {
    let (mut batch, frame_zero) = ArenaBatch::start(SESSION)?;

    let mut bytes = Vec::new();
    write_reset(&mut bytes, &frame_zero)?;
    let reset = Fixture {
        purpose: "Frame zero, sent unasked after the handshake so a client can step \
                  immediately. No flags, no death count, no reward.",
        ..reset_fixture(root, "reset-frame-zero.bin", &bytes, &frame_zero, layout)?
    };

    // The budget is three frames, so the third step ends both episodes. The
    // actions differ per env so the two observations cannot be confused.
    let mut running = None;
    let mut ended = None;
    for frame in 1..=SESSION.max_frames {
        let stepped = batch.step(&[2, 5])?;
        if frame == 1 {
            running = Some(stepped);
        } else if frame == SESSION.max_frames {
            ended = Some(stepped);
        }
    }
    let running = running.ok_or("the first step produced a batch")?;
    let ended = ended.ok_or("the last step produced a batch")?;

    let mut bytes = Vec::new();
    write_step(&mut bytes, &running)?;
    let ordinary = Fixture {
        purpose: "One ordinary transition per env: frame 1, still running, no terminal \
                  section and no replacement seed.",
        ..step_fixture(root, "step-running.bin", &bytes, &running, layout)?
    };

    let mut bytes = Vec::new();
    write_step(&mut bytes, &ended)?;
    let auto_reset = Fixture {
        purpose: "The auto-reset shape. Both envs hit the frame budget, so `observations` \
                  is already the replacement episode's first frame while the finished \
                  episode's last frame is in the terminal section, and each transition \
                  carries the replacement's seed behind its presence flag.",
        ..step_fixture(root, "step-auto-reset.bin", &bytes, &ended, layout)?
    };

    Ok(vec![reset, ordinary, auto_reset, emit_death(root, layout)?])
}

/// The other way an episode ends: the player is hit rather than timed out.
///
/// `step-auto-reset.bin` has both envs truncating together, which leaves two
/// things untested. A death sets `terminated` without `truncated`, and a
/// learner treats those oppositely -- it bootstraps from a truncated
/// observation and must not from a terminated one. And because the two envs
/// almost never die on the same frame, this is also the mixed batch: one env
/// carries a terminal observation and a replacement seed while the other
/// simply carries on.
fn emit_death(root: &Path, layout: &Layout) -> Result<Fixture, Box<dyn core::error::Error>> {
    // A full arena and an idle player, so a hit arrives on its own. The budget
    // is far away, so nothing here can be a timeout.
    let hunted = BatchConfig {
        envs: 2,
        workers: 1,
        root_seed: 31,
        enemy_count: 100,
        max_frames: 100_000,
        hold_frames: SESSION.hold_frames,
    };
    let (mut batch, _frame_zero) = ArenaBatch::start(hunted)?;

    let mut killed = None;
    for _ in 0..hunted.max_frames {
        let stepped = batch.step(&[0, 0])?;
        if stepped
            .transitions
            .iter()
            .any(|transition| transition.terminated)
        {
            killed = Some(stepped);
            break;
        }
    }
    let killed = killed.ok_or("an idle player is caught eventually")?;

    let mut bytes = Vec::new();
    write_step(&mut bytes, &killed)?;
    Ok(Fixture {
        purpose: "A death: `terminated` without `truncated`, which a learner must not \
                  bootstrap from. The envs do not die together, so this is also the \
                  mixed batch -- one env has a terminal observation and a replacement \
                  seed, the other is still running.",
        ..step_fixture(root, "step-death.bin", &bytes, &killed, layout)?
    })
}

// --- hand-placed scenes -------------------------------------------------

/// Three arenas built by hand, one per env, each aimed at a decoding mistake.
fn emit_scenes(root: &Path, layout: &Layout) -> Result<Fixture, Box<dyn core::error::Error>> {
    let scenes = [asymmetric_scene(), blast_phase_scene(), seam_scene()];
    let mut observations = Vec::with_capacity(scenes.len() * OBSERVATION_VALUES);
    for scene in &scenes {
        let mut buffer = vec![0.0_f32; OBSERVATION_VALUES];
        encode(scene, SESSION.hold_frames, &mut buffer)?;
        observations.extend_from_slice(&buffer);
    }

    let envs = u32::try_from(scenes.len())?;
    let batch = StepBatch {
        transitions: (0..envs)
            .map(|env| EnvTransition {
                frame: 1,
                enemy_deaths: env,
                episode_seed: u64::from(env),
                ..EnvTransition::default()
            })
            .collect(),
        observations,
        terminal: Vec::new(),
    };
    let mut bytes = Vec::new();
    write_step(&mut bytes, &batch)?;

    Ok(Fixture {
        purpose: "Hand-placed arenas, one per env: 0 asymmetric positions and moving \
                  hazards, 1 blast phase either side of its peak, 2 hazards across the \
                  wrap seam. These pin the encoder's axes, signs and wrapping, which a \
                  real episode cannot be relied on to contain on a chosen frame.",
        ..step_fixture(root, "scenes.bin", &bytes, &batch, layout)?
    })
}

/// Nothing mirrored and nothing still: every axis and sign is distinguishable.
///
/// A decoder that transposes x and y, or flips one of them, cannot reproduce
/// this scene, because no two hazards share an offset, a size or a velocity.
fn asymmetric_scene() -> ArenaView {
    ArenaView {
        frame: 11,
        player: Some(PlayerView {
            entity: entity(1),
            // Off-centre, and moving up and to the right in world terms, which
            // is a negative dy once the observation flips to screen axes.
            position: Vec2::new(37.0 * PIXEL, -21.0 * PIXEL),
            velocity: Vec2::new(90.0, 140.0),
            collider: Collider::rectangle(Vec2::splat(12.0)),
        }),
        enemies: vec![
            // Right and slightly up, drifting left: opposite signs on each axis.
            enemy(2, EnemyKind::Normal, 18.0, 6.0, 5.0, -70.0, 25.0),
            // Far left and well down, larger, drifting down.
            enemy(3, EnemyKind::Kamikaze, -41.0, -29.0, 7.0, 15.0, -55.0),
            // Close above, small, nearly still.
            enemy(4, EnemyKind::Normal, 3.0, 23.0, 4.0, -5.0, 2.0),
        ],
        blasts: Vec::new(),
        half_extents: WORLD_HALF_EXTENTS,
    }
}

/// One blast growing and one expiring, plus the kamikaze channel beside them.
///
/// A blast passes through every width twice, so size alone cannot say which
/// half of its life it is in. Phase is `1 - 2 * progress`, so these two must
/// come back with opposite signs.
fn blast_phase_scene() -> ArenaView {
    ArenaView {
        frame: 23,
        player: Some(PlayerView {
            entity: entity(1),
            position: Vec2::new(-14.0 * PIXEL, 9.0 * PIXEL),
            velocity: Vec2::new(-40.0, 0.0),
            collider: Collider::rectangle(Vec2::splat(12.0)),
        }),
        enemies: vec![enemy(2, EnemyKind::Kamikaze, -22.0, -7.0, 6.0, 30.0, -20.0)],
        blasts: vec![
            // Just detonated: phase near +1.
            blast(5, 20.0, 11.0, 9.0, 0.08, 1.0),
            // Nearly gone: phase near -1, and a different size, so the two
            // cannot be told apart by width alone either.
            blast(6, -30.0, 26.0, 5.0, 0.92, 1.0),
        ],
        half_extents: WORLD_HALF_EXTENTS,
    }
}

/// The player in a corner, with hazards over both seams.
///
/// Displacement uses `torus::wrapped_delta`, so a hazard a few pixels past the
/// edge is a few pixels away, not a world away. A decoder that subtracts
/// positions instead sees an empty window here.
fn seam_scene() -> ArenaView {
    let half = WORLD_HALF_EXTENTS;
    ArenaView {
        frame: 47,
        player: Some(PlayerView {
            entity: entity(1),
            // Just inside the top-right corner of the world.
            position: Vec2::new(half.x - 6.0 * PIXEL, half.y - 9.0 * PIXEL),
            velocity: Vec2::new(60.0, 60.0),
            collider: Collider::rectangle(Vec2::splat(12.0)),
        }),
        enemies: vec![
            EnemyView {
                entity: entity(2),
                kind: EnemyKind::Normal,
                // Over the x seam: far left in world coordinates, just right
                // of the player once the world wraps.
                position: Vec2::new(-half.x + 10.0 * PIXEL, half.y - 4.0 * PIXEL),
                velocity: Vec2::new(45.0, -10.0),
                collider: Collider::rectangle(Vec2::splat(5.0 * PIXEL)),
            },
            EnemyView {
                entity: entity(3),
                kind: EnemyKind::Kamikaze,
                // Over the y seam, and over the x seam as well: the corner case.
                position: Vec2::new(-half.x + 3.0 * PIXEL, -half.y + 7.0 * PIXEL),
                velocity: Vec2::new(-20.0, 35.0),
                collider: Collider::rectangle(Vec2::splat(6.0 * PIXEL)),
            },
        ],
        blasts: Vec::new(),
        half_extents: half,
    }
}

const fn entity(id: u32) -> Entity {
    Entity::from_raw_u32(id).expect("a valid entity index")
}

/// An enemy placed in reference pixels, with a world-unit velocity.
fn enemy(
    id: u32,
    kind: EnemyKind,
    x: f32,
    y: f32,
    half: f32,
    velocity_x: f32,
    velocity_y: f32,
) -> EnemyView {
    EnemyView {
        entity: entity(id),
        kind,
        position: Vec2::new(x * PIXEL, y * PIXEL),
        velocity: Vec2::new(velocity_x, velocity_y),
        collider: Collider::rectangle(Vec2::splat(half * PIXEL)),
    }
}

/// A blast placed in reference pixels, at a chosen point in its life.
fn blast(id: u32, x: f32, y: f32, half: f32, age: f32, duration: f32) -> BlastView {
    BlastView {
        entity: entity(id),
        position: Vec2::new(x * PIXEL, y * PIXEL),
        collider: Collider::rectangle(Vec2::splat(half * PIXEL)),
        age,
        duration,
    }
}

// --- writing and describing ---------------------------------------------

fn write_fixture(
    root: &Path,
    name: &'static str,
    kind: &'static str,
    bytes: &[u8],
) -> Result<Fixture, Box<dyn core::error::Error>> {
    let path = root.join(name);
    fs::write(&path, bytes)?;
    println!("wrote {} ({} bytes)", path.display(), bytes.len());
    Ok(Fixture {
        file: name,
        kind,
        purpose: "",
        bytes: bytes.len(),
        envs: 0,
        observation_values: OBSERVATION_VALUES,
        episode_seeds: Vec::new(),
        transitions: Vec::new(),
        terminal_envs: Vec::new(),
        samples: Vec::new(),
        terminal_samples: Vec::new(),
    })
}

fn reset_fixture(
    root: &Path,
    name: &'static str,
    bytes: &[u8],
    batch: &ResetBatch,
    layout: &Layout,
) -> Result<Fixture, Box<dyn core::error::Error>> {
    let envs = u32::try_from(batch.seeds.len())?;
    Ok(Fixture {
        envs,
        episode_seeds: batch.seeds.clone(),
        samples: sample_observations(&batch.observations, layout),
        ..write_fixture(root, name, "reset", bytes)?
    })
}

fn step_fixture(
    root: &Path,
    name: &'static str,
    bytes: &[u8],
    batch: &StepBatch,
    layout: &Layout,
) -> Result<Fixture, Box<dyn core::error::Error>> {
    let envs = u32::try_from(batch.transitions.len())?;
    Ok(Fixture {
        envs,
        episode_seeds: batch
            .transitions
            .iter()
            .map(|transition| transition.episode_seed)
            .collect(),
        transitions: batch
            .transitions
            .iter()
            .enumerate()
            .map(|(index, transition)| describe(index, *transition))
            .collect(),
        terminal_envs: batch.terminal.iter().map(|end| end.env).collect(),
        samples: sample_observations(&batch.observations, layout),
        terminal_samples: sample_terminals(&batch.terminal, layout),
        ..write_fixture(root, name, "step", bytes)?
    })
}

fn describe(index: usize, transition: EnvTransition) -> TransitionExpectation {
    TransitionExpectation {
        env: u32::try_from(index).unwrap_or(u32::MAX),
        frame: transition.frame,
        terminated: transition.terminated,
        truncated: transition.truncated,
        enemy_deaths: transition.enemy_deaths,
        episode_seed: transition.episode_seed,
        reset_seed: transition.reset_seed,
    }
}

/// Spot checks into each env's slice of a flat observation batch.
fn sample_observations(observations: &[f32], layout: &Layout) -> Vec<Sample> {
    observations
        .chunks(OBSERVATION_VALUES)
        .enumerate()
        .flat_map(|(env, observation)| {
            let env = u32::try_from(env).unwrap_or(u32::MAX);
            pick(env, observation, layout)
        })
        .collect()
}

fn sample_terminals(terminals: &[TerminalObservation], layout: &Layout) -> Vec<Sample> {
    terminals
        .iter()
        .flat_map(|end| pick(end.env, &end.observation, layout))
        .collect()
}

/// Which values of one observation the manifest records.
///
/// The player's own velocity and every path value, because both sections are
/// small and are where an axis or a scale error shows up first. Then a bounded,
/// evenly spaced walk of the grid's non-zero values, so the sample says
/// something about where hazards actually landed rather than about the zeroes
/// between them.
fn pick(env: u32, observation: &[f32], layout: &Layout) -> Vec<Sample> {
    let grid = layout.grid_section;
    let occupied: Vec<usize> = (grid.offset..grid.offset + grid.length)
        .filter(|index| observation.get(*index).is_some_and(|value| *value != 0.0))
        .collect();
    let stride = (occupied.len() / GRID_SAMPLES).max(1);
    let paths = layout.path_section;

    // The player's velocity, then the sampled grid, then every path value.
    (0..PLAYER_VALUES)
        .chain(occupied.iter().copied().step_by(stride).take(GRID_SAMPLES))
        .chain(paths.offset..paths.offset + paths.length)
        .filter_map(|index| {
            observation.get(index).map(|value| Sample {
                env,
                index,
                value: f64::from(*value),
                bits: value.to_bits(),
            })
        })
        .collect()
}

fn hex(bytes: &[u8]) -> String {
    bytes
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<Vec<_>>()
        .join(" ")
}

// --- the manifest -------------------------------------------------------

#[derive(Serialize)]
struct Manifest {
    protocol_version: u32,
    magic: String,
    regenerate: &'static str,
    comparison: &'static str,
    observation_values: usize,
    layout: Layout,
    session: Session,
    fixtures: Vec<Fixture>,
}

/// The configuration the real-arena fixtures were produced from.
#[derive(Serialize)]
struct Session {
    envs: u32,
    root_seed: u64,
    enemy_count: usize,
    max_frames: u32,
    hold_frames: u32,
}

#[derive(Serialize)]
struct Fixture {
    file: &'static str,
    kind: &'static str,
    purpose: &'static str,
    bytes: usize,
    envs: u32,
    observation_values: usize,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    episode_seeds: Vec<u64>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    transitions: Vec<TransitionExpectation>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    terminal_envs: Vec<u32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    samples: Vec<Sample>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    terminal_samples: Vec<Sample>,
}

#[derive(Serialize)]
struct TransitionExpectation {
    env: u32,
    frame: u32,
    terminated: bool,
    truncated: bool,
    enemy_deaths: u32,
    episode_seed: u64,
    reset_seed: Option<u64>,
}

/// One value of one observation, by its index in the flat array.
#[derive(Serialize)]
struct Sample {
    env: u32,
    index: usize,
    /// The `f32` widened to `f64`, which is exact and survives JSON.
    value: f64,
    /// The same value's bit pattern, for a reader that would rather compare
    /// integers than trust a decimal.
    bits: u32,
}
