//! Tests for the worker pool.
//!
//! The point of these is that worker count is invisible: the same seed and the
//! same actions must produce the same bytes whether one thread or four did the
//! work.

use crate::gym::workers::*;
use crate::observation::OBSERVATION_VALUES;

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
        let actions: Vec<u8> = (0..envs)
            .map(|env| {
                let raw = usize::try_from(env).unwrap_or(0).wrapping_add(step);
                u8::try_from(raw % 9).unwrap_or(0)
            })
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

#[test]
fn a_batch_reports_one_observation_for_every_env() {
    let (mut batch, initial) = ArenaBatch::start(config(4, 2)).expect("the batch starts");
    assert_eq!(batch.envs(), 4);
    assert_eq!(initial.seeds.len(), 4);
    assert_eq!(initial.observations.len(), 4 * OBSERVATION_VALUES);

    let result = batch.step(&[0, 1, 2, 3]).expect("a step runs");
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
    let result = batch.step(&[0, 0]).expect("a step runs");
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
        last = Some(batch.step(&[0]).expect("a step runs"));
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

    let next = batch.step(&[0]).expect("the new episode steps");
    let transition = next.transitions.first().expect("one env");
    assert_eq!(transition.frame, 1, "the replacement episode starts at one");
    assert!(!transition.truncated);
}

#[test]
fn a_seeded_reset_reproduces_the_batch_exactly() {
    let (mut batch, first) = ArenaBatch::start(config(2, 2)).expect("the batch starts");
    batch.step(&[1, 2]).expect("a step runs");
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
        batch.step(&[0, 0]).expect("a step runs");
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
        batch.step(&[0, 0]),
        Err(BatchError::InvalidConfig(_))
    ));
}

#[test]
fn dropping_a_batch_mid_flight_joins_its_workers() {
    let (mut batch, _) = ArenaBatch::start(config(4, 4)).expect("the batch starts");
    batch.step(&[0, 0, 0, 0]).expect("a step runs");
    // The interesting part is that this returns rather than hanging: every
    // worker must be told to stop and drained before it is joined.
    drop(batch);
}
