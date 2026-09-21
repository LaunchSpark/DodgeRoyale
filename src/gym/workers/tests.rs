//! Tests for the worker pool.
//!
//! The point of these is that worker count is invisible: the same seed and the
//! same actions must produce the same bytes whether one thread or four did the
//! work.

use bevy::math::Vec2;

use crate::gym::workers::*;
use crate::observation::{OBSERVATION_VALUES, layout};

/// A direction off the compass, so a test exercises the space the policy
/// actually sends rather than the nine headings version 1 allowed.
fn heading(step: usize) -> Vec2 {
    let degrees = f32::from(u16::try_from(step.wrapping_mul(37) % 360).unwrap_or(0));
    let radians = degrees.to_radians();
    Vec2::new(radians.cos(), radians.sin())
}

/// A small batch: few enemies, short episodes, so tests stay quick.
fn config(envs: u32, workers: u32) -> BatchConfig {
    BatchConfig {
        envs,
        workers,
        root_seed: 2_024,
        enemy_count: 8,
        max_frames: 40,
        hold_frames: 24,
    }
}

/// Step a whole batch with a fixed action cycle and collect what came back.
fn run(envs: u32, workers: u32, steps: usize) -> (Vec<u64>, Vec<f32>, Vec<String>) {
    let (mut batch, initial) = ArenaBatch::start(config(envs, workers)).expect("the batch starts");
    let mut seeds = initial.seeds;
    let mut observations = initial.observations;
    let mut records = Vec::new();

    for step in 0..steps {
        let actions: Vec<Vec2> = (0..envs)
            .map(|env| heading(usize::try_from(env).unwrap_or(0).wrapping_add(step)))
            .collect();
        let result = batch.step(&actions).expect("a step runs");
        observations.extend_from_slice(&result.observations);
        for transition in &result.transitions {
            records.push(format!(
                "f{} t{} r{} d{} s{:x} n{:?}",
                transition.frame,
                u8::from(transition.terminated),
                u8::from(transition.truncated),
                transition.enemy_deaths,
                transition.episode_seed,
                transition.reset_seed
            ));
            if let Some(seed) = transition.reset_seed {
                seeds.push(seed);
            }
        }
        for terminal in &result.terminal {
            records.push(format!("terminal env {}", terminal.env));
            observations.extend_from_slice(&terminal.observation);
        }
    }
    (seeds, observations, records)
}

/// The same batch, with a different prediction horizon.
fn held(envs: u32, workers: u32, hold_frames: u32) -> BatchConfig {
    BatchConfig {
        hold_frames,
        ..config(envs, workers)
    }
}

#[test]
fn one_step_advances_exactly_one_frame() {
    // The cadence everything else is built on: a policy decides once per
    // simulated 1/60 second, because that is what one STEP costs. A reader
    // that believed an action was held for several frames would train on one
    // cadence and play on another.
    let (mut batch, _frame_zero) = ArenaBatch::start(config(2, 1)).expect("the batch starts");
    for expected in 1..=5_u32 {
        let stepped = batch.step(&[Vec2::ZERO; 2]).expect("a step runs");
        for transition in &stepped.transitions {
            assert_eq!(
                transition.frame, expected,
                "one step advanced more than one frame"
            );
        }
    }
}

#[test]
fn hold_frames_moves_the_predicted_paths_and_nothing_else() {
    // `hold_frames` is how far ahead the observation's candidate paths are
    // predicted, and it is never an action repeat. Two batches that differ only
    // in it must be in the same world, seeing the same hazards, disagreeing
    // only about where each path would lead.
    let brief = held(1, 1, 6);
    let long = held(1, 1, 48);
    let layout = layout(brief.hold_frames);
    let paths = layout.path_section.offset..layout.path_section.offset + layout.path_section.length;

    let (mut brief, _) = ArenaBatch::start(brief).expect("the brief batch starts");
    let (mut long, _) = ArenaBatch::start(long).expect("the long batch starts");

    let heading = Vec2::new(0.966, 0.259);
    let mut brief_step = brief.step(&[heading]).expect("a step runs");
    let mut long_step = long.step(&[heading]).expect("a step runs");
    for _ in 0..4 {
        brief_step = brief.step(&[heading]).expect("a step runs");
        long_step = long.step(&[heading]).expect("a step runs");
    }

    assert_eq!(
        brief_step.transitions.first().map(|t| t.frame),
        long_step.transitions.first().map(|t| t.frame),
        "the horizon changed how fast the arena advanced"
    );

    let (brief_values, long_values) = (&brief_step.observations, &long_step.observations);
    assert_eq!(brief_values.len(), long_values.len());
    let before_paths = paths.start;
    assert_eq!(
        brief_values.get(..before_paths),
        long_values.get(..before_paths),
        "the player and the hazards must be identical: the horizon is a \
         prediction, and predicting further cannot move the world"
    );
    assert_ne!(
        brief_values.get(paths.clone()),
        long_values.get(paths),
        "a longer horizon must reach further; if the paths match, hold_frames \
         is not reaching the encoder at all"
    );
}

#[test]
fn a_batch_reports_one_observation_for_every_env() {
    let (mut batch, initial) = ArenaBatch::start(config(4, 2)).expect("the batch starts");
    assert_eq!(batch.envs(), 4);
    assert_eq!(initial.seeds.len(), 4);
    assert_eq!(initial.observations.len(), 4 * OBSERVATION_VALUES);

    let result = batch
        .step(&[heading(0), heading(1), heading(2), heading(3)])
        .expect("a step runs");
    assert_eq!(result.transitions.len(), 4);
    assert_eq!(result.observations.len(), 4 * OBSERVATION_VALUES);
    assert!(
        result
            .transitions
            .iter()
            .all(|transition| transition.frame == 1),
        "every env advanced exactly one frame"
    );
}

#[test]
fn worker_count_does_not_change_a_single_byte() {
    let one = run(4, 1, 60);
    let two = run(4, 2, 60);
    let four = run(4, 4, 60);
    assert_eq!(one.0, two.0, "the same seeds");
    assert_eq!(one.1, two.1, "the same observations");
    assert_eq!(one.2, two.2, "the same transitions");
    assert_eq!(one, four, "however the shards are cut");
}

#[test]
fn more_workers_than_envs_is_allowed() {
    let (mut batch, initial) = ArenaBatch::start(config(2, 8)).expect("the batch starts");
    assert_eq!(initial.seeds.len(), 2);
    let result = batch.step(&[Vec2::ZERO; 2]).expect("a step runs");
    assert_eq!(result.transitions.len(), 2);
}

#[test]
fn envs_finish_at_different_times_and_each_resets_on_its_own() {
    // Episodes are 40 frames, so 45 steps takes every env past one ending.
    let (_, _, records) = run(3, 2, 45);
    let resets = records
        .iter()
        .filter(|record| record.contains("nSome"))
        .count();
    assert_eq!(resets, 3, "each env ended once and reset once: {records:?}");
    let terminals = records
        .iter()
        .filter(|record| record.starts_with("terminal env"))
        .count();
    assert_eq!(terminals, 3, "and each carried a final observation");
}

#[test]
fn an_auto_reset_returns_the_new_episodes_frame_and_keeps_the_old_one() {
    let (mut batch, _) = ArenaBatch::start(config(1, 1)).expect("the batch starts");
    let mut last = None;
    for _ in 0..40 {
        last = Some(batch.step(&[Vec2::ZERO]).expect("a step runs"));
    }
    let ending = last.expect("forty steps happened");
    let transition = ending.transitions.first().expect("one env");
    assert!(transition.truncated, "the budget ran out at frame 40");
    assert_eq!(transition.frame, 40);
    assert!(
        transition.reset_seed.is_some(),
        "the env carried on into a new episode"
    );
    assert_eq!(
        ending.terminal.len(),
        1,
        "with the old episode's last frame"
    );

    let next = batch.step(&[Vec2::ZERO]).expect("the new episode steps");
    let transition = next.transitions.first().expect("one env");
    assert_eq!(transition.frame, 1, "the replacement episode starts at one");
    assert!(!transition.truncated);
}

#[test]
fn a_seeded_reset_reproduces_the_batch_exactly() {
    let (mut batch, first) = ArenaBatch::start(config(2, 2)).expect("the batch starts");
    batch.step(&[heading(1), heading(2)]).expect("a step runs");
    let again = batch.reset(Some(2_024)).expect("a seeded reset runs");
    assert_eq!(
        again.seeds, first.seeds,
        "the stream restarts from the root"
    );
    assert_eq!(
        again.observations, first.observations,
        "and so does every arena"
    );
}

#[test]
fn an_unseeded_reset_advances_the_stream_rather_than_repeating_it() {
    let (mut batch, first) = ArenaBatch::start(config(2, 1)).expect("the batch starts");
    let next = batch.reset(None).expect("an unseeded reset runs");
    assert_ne!(next.seeds, first.seeds, "a fresh episode, not the same one");
    let after = batch.reset(None).expect("another reset runs");
    assert_ne!(after.seeds, next.seeds, "and it keeps advancing");
}

#[test]
fn repeated_resets_keep_working() {
    let (mut batch, _) = ArenaBatch::start(config(2, 2)).expect("the batch starts");
    for _ in 0..5 {
        batch.reset(None).expect("a reset runs");
        batch.step(&[Vec2::ZERO; 2]).expect("a step runs");
    }
}

#[test]
fn a_batch_that_cannot_build_its_arenas_says_so_at_startup() {
    let error = ArenaBatch::start(BatchConfig {
        envs: 2,
        workers: 1,
        root_seed: 1,
        // More enemies than one initialisation pass can place.
        enemy_count: 5_000,
        max_frames: 10,
        hold_frames: 24,
    });
    match error {
        Err(BatchError::Arena { reason, .. }) => {
            assert!(
                reason.contains("enemies"),
                "reported as a population: {reason}"
            );
        }
        Err(other) => panic!("expected an arena failure, got {other}"),
        Ok(_) => panic!("five thousand enemies should not fit"),
    }
}

#[test]
fn a_batch_needs_envs_and_workers() {
    assert!(matches!(
        ArenaBatch::start(BatchConfig {
            envs: 0,
            ..config(1, 1)
        }),
        Err(BatchError::InvalidConfig(_))
    ));
    assert!(matches!(
        ArenaBatch::start(BatchConfig {
            workers: 0,
            ..config(1, 1)
        }),
        Err(BatchError::InvalidConfig(_))
    ));
}

#[test]
fn an_action_batch_of_the_wrong_size_is_refused() {
    let (mut batch, _) = ArenaBatch::start(config(3, 2)).expect("the batch starts");
    assert!(matches!(
        batch.step(&[Vec2::ZERO; 2]),
        Err(BatchError::InvalidConfig(_))
    ));
}

#[test]
fn dropping_a_batch_mid_flight_joins_its_workers() {
    let (mut batch, _) = ArenaBatch::start(config(4, 4)).expect("the batch starts");
    batch.step(&[Vec2::ZERO; 4]).expect("a step runs");
    // The interesting part is that this returns rather than hanging: every
    // worker must be told to stop and drained before it is joined.
    drop(batch);
}
