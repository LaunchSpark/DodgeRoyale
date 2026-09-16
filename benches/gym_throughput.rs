//! Where a training step's time actually goes.
//!
//! The design assumes no bottleneck and says to measure one: simulation,
//! encoding and pipe transfer separately, at 8 and 64 envs. The decision
//! riding on the numbers is whether the observation needs packing. Channels
//! 0-2 and 4 are occupancy, one bit of information in a whole `f32`, so
//! packing them as bitfields would cut 115,128 bytes to 51,640. That is worth
//! doing only if transfer is what the trainer waits for.
//!
//! Simulation and encoding run on the worker threads, so more workers buys
//! more of them. Transfer runs once, in the coordinator, on one pipe: it does
//! not scale, which is why it is reported per env step beside the other two
//! rather than as a share of a wall clock.
//!
//! No benchmarking crate: this reports means over a fixed number of
//! iterations, which is enough to separate microseconds from milliseconds.
//! Run it optimised, or the numbers describe a debug build:
//!
//! ```sh
//! cargo bench --locked --no-default-features --bench gym_throughput
//! cargo bench --locked --no-default-features --bench gym_throughput -- 4
//! ```

// A benchmark is a dev-only binary. A failed expectation stops the run, which
// is how it reports, and the arithmetic below averages counts the loops have
// just produced.
#![expect(
    clippy::expect_used,
    clippy::arithmetic_side_effects,
    reason = "Dev-only binary: a failed expectation is how it reports, and the maths is averaging"
)]

use std::io::{BufReader, BufWriter, Read, Write};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use dodge_royale::gym::protocol::{EnvTransition, StepBatch, max_payload, read_step, write_step};
use dodge_royale::gym::workers::{ArenaBatch, BatchConfig};
use dodge_royale::observation::{OBSERVATION_VALUES, encode};
use dodge_royale::simulation::{ArenaConfig, DEFAULT_HOLD_FRAMES, HeadlessArena};

/// The two sizes the design names: the training default and the stress case.
const ENV_COUNTS: [u32; 2] = [8, 64];

/// Frames each single-env stage is averaged over.
const FRAMES: u32 = 600;

/// Whole-batch steps, each of which costs every env a frame.
const BATCH_STEPS: u32 = 120;

/// Frames run before the clock starts, so caches and the population settle.
const WARMUP: u32 = 60;

/// Batches pushed through the pipe per measurement.
const TRANSFER_ROUNDS: u32 = 40;

/// Bytes one `f32` observation costs on the wire.
const OBSERVATION_BYTES: usize = OBSERVATION_VALUES * 4;

/// Bytes the same observation would cost with channels 0-2 and 4 packed as
/// bitfields, per the design's fallback. Those four occupancy channels hold
/// one bit each per cell; every other value stays an `f32`.
const PACKED_OBSERVATION_BYTES: usize = {
    let occupancy = 4 * 64 * 64;
    (OBSERVATION_VALUES - occupancy) * 4 + occupancy / 8
};

fn main() {
    let counts = env_counts();
    let config = ArenaConfig::default();
    let workers = available_workers();

    println!("dodge-royale gym throughput");
    println!(
        "  {} enemies, hold-frames {DEFAULT_HOLD_FRAMES}, {OBSERVATION_VALUES} values per \
         observation ({OBSERVATION_BYTES} bytes, {PACKED_OBSERVATION_BYTES} packed)",
        config.enemy_count
    );
    println!(
        "  means over {FRAMES} frames, {BATCH_STEPS} batch steps and {TRANSFER_ROUNDS} pipe rounds"
    );

    // Simulation and encoding are per-env costs that do not depend on how many
    // envs a session runs, so they are measured once rather than per size.
    let single = measure_single_env(config);

    for envs in counts {
        report(envs, workers.min(envs), config, &single);
    }
}

/// Env counts to report on: the arguments, or the design's two sizes.
fn env_counts() -> Vec<u32> {
    let parsed: Vec<u32> = std::env::args()
        .skip(1)
        .filter_map(|argument| argument.parse().ok())
        .filter(|envs| *envs > 0)
        .collect();
    if parsed.is_empty() {
        ENV_COUNTS.to_vec()
    } else {
        parsed
    }
}

/// Worker threads a batch gets, capped the way a real session caps them.
fn available_workers() -> u32 {
    thread::available_parallelism()
        .map(std::num::NonZero::get)
        .ok()
        .and_then(|threads| u32::try_from(threads).ok())
        .unwrap_or(1)
        .max(1)
}

/// What one env costs for one frame, on one thread.
struct SingleEnv {
    /// Advancing the world, with nothing read out of it.
    simulate: Duration,
    /// Reading the world and turning it into the observation.
    encode: Duration,
    /// One env's observation, for building a realistic batch to transfer.
    observation: Vec<f32>,
}

/// Time `HeadlessArena::step`, and then `view` plus `encode`, separately.
///
/// Each frame is timed on its own so that an episode ending, and the arena
/// rebuild that follows it, stays outside the clock: a reset costs far more
/// than a frame and would swamp a mean it is no part of.
fn measure_single_env(config: ArenaConfig) -> SingleEnv {
    let mut arena = HeadlessArena::new(config).expect("the arena fills its population");
    let mut buffer = vec![0.0_f32; OBSERVATION_VALUES];
    let mut simulated = Duration::ZERO;
    let mut encoded = Duration::ZERO;

    for frame in 0..(WARMUP + FRAMES) {
        if arena.done() {
            arena
                .reset(u64::from(frame))
                .expect("a fresh episode fills too");
        }
        // Cycling the action keeps the player moving, so the measured frames
        // are ordinary play rather than a corner the enemies have lost track of.
        arena
            .set_action(u8::try_from(frame % 9).unwrap_or(0))
            .expect("a cycled action is one of the nine");

        let started = Instant::now();
        arena.step().expect("a live episode steps");
        let stepping = started.elapsed();

        let started = Instant::now();
        let view = arena.view();
        encode(&view, DEFAULT_HOLD_FRAMES, &mut buffer).expect("a finite view encodes");
        let encoding = started.elapsed();

        if frame >= WARMUP {
            simulated += stepping;
            encoded += encoding;
        }
    }

    SingleEnv {
        simulate: simulated / FRAMES,
        encode: encoded / FRAMES,
        observation: buffer,
    }
}

/// A step response of the ordinary shape: every env continuing, none finished.
///
/// Terminal observations are deliberately absent. On a 3,600 frame budget they
/// arrive on well under one step in a hundred, so including one would describe
/// a rare batch rather than the batch the trainer spends its time on.
fn sample_batch(envs: u32, observation: &[f32]) -> StepBatch {
    let width = usize::try_from(envs).unwrap_or(0);
    let mut observations = Vec::with_capacity(observation.len().saturating_mul(width));
    for _ in 0..envs {
        observations.extend_from_slice(observation);
    }
    StepBatch {
        transitions: (0..envs)
            .map(|env| EnvTransition {
                frame: 512,
                enemy_deaths: 1,
                episode_seed: u64::from(env),
                ..EnvTransition::default()
            })
            .collect(),
        observations,
        terminal: Vec::new(),
    }
}

/// Push whole batches through a real OS pipe and read them back.
///
/// A real pipe, not a buffer: the trainer sits on the other side of one, and a
/// 7 MB batch does not fit in its kernel buffer, so the writer blocks on the
/// reader exactly as it does in a session. The reader acknowledges once, at
/// the end, so the measurement covers writing, the kernel and parsing.
///
/// Both ends are buffered, because both ends are buffered in a real session:
/// `run_gym` writes through a `BufWriter` and the client reads a locked stdin.
/// That is not a detail. The codec moves four bytes per call, so against a raw
/// pipe every `f32` costs a syscall -- 57,564 of them for two envs -- and the
/// measurement describes the syscalls rather than the transfer.
fn measure_transfer(envs: u32, batch: &StepBatch) -> Duration {
    let (reader, writer) = std::io::pipe().expect("a pipe");
    let mut writer = BufWriter::new(writer);
    let maximum = max_payload(envs, OBSERVATION_VALUES);
    let (finished, acknowledged) = mpsc::channel();

    let drain = thread::spawn(move || {
        let mut reader = BufReader::new(reader);
        for _ in 0..TRANSFER_ROUNDS {
            read_step(&mut reader, maximum).expect("a batch arrives whole");
        }
        let _ = finished.send(());
    });

    let started = Instant::now();
    for _ in 0..TRANSFER_ROUNDS {
        write_step(&mut writer, batch).expect("the batch is written");
    }
    acknowledged.recv().expect("the reader finishes");
    let elapsed = started.elapsed();

    drop(writer);
    let _ = drain.join();
    elapsed / TRANSFER_ROUNDS
}

/// How fast the same number of bytes crosses the same pipe in one call.
///
/// The gap between this and [`measure_transfer`] is what the codec costs
/// rather than the kernel. `write_floats` moves four bytes per call, so a
/// 7 MB batch is nearly two million calls into the buffer. That matters for
/// the design's decision: packing the format sends fewer bytes but bumps the
/// protocol version, while making the codec move whole slices sends the same
/// bytes faster and changes nothing on the wire.
fn measure_pipe_ceiling(bytes: usize) -> Duration {
    let payload = vec![0_u8; bytes];
    let (mut reader, mut writer) = std::io::pipe().expect("a pipe");
    let (finished, acknowledged) = mpsc::channel();

    let drain = thread::spawn(move || {
        let mut sink = vec![0_u8; bytes];
        for _ in 0..TRANSFER_ROUNDS {
            reader
                .read_exact(&mut sink)
                .expect("the payload arrives whole");
        }
        let _ = finished.send(());
    });

    let started = Instant::now();
    for _ in 0..TRANSFER_ROUNDS {
        writer.write_all(&payload).expect("the payload is written");
    }
    acknowledged.recv().expect("the reader finishes");
    let elapsed = started.elapsed();

    drop(writer);
    let _ = drain.join();
    elapsed / TRANSFER_ROUNDS
}

/// Bytes one step response occupies on the wire.
fn wire_bytes(batch: &StepBatch) -> usize {
    let mut sink = Vec::new();
    write_step(&mut sink, batch).expect("the batch serialises");
    sink.len()
}

/// Time one whole batch step, which is what the server actually answers with.
fn measure_batch(envs: u32, workers: u32, config: ArenaConfig) -> Duration {
    let (mut batch, _frame_zero) = ArenaBatch::start(BatchConfig {
        envs,
        workers,
        root_seed: 1,
        enemy_count: config.enemy_count,
        max_frames: config.max_frames,
        hold_frames: DEFAULT_HOLD_FRAMES,
    })
    .expect("every arena fills its population");

    let actions: Vec<u8> = (0..envs)
        .map(|env| u8::try_from(env % 9).unwrap_or(0))
        .collect();
    for _ in 0..WARMUP {
        batch.step(&actions).expect("a batch steps");
    }

    let started = Instant::now();
    for _ in 0..BATCH_STEPS {
        batch.step(&actions).expect("a batch steps");
    }
    started.elapsed() / BATCH_STEPS
}

/// Print one size's table, and the verdict the design asked the numbers for.
fn report(envs: u32, workers: u32, config: ArenaConfig, single: &SingleEnv) {
    let sample = sample_batch(envs, &single.observation);
    let bytes = wire_bytes(&sample);
    let transfer = measure_transfer(envs, &sample);
    let ceiling = measure_pipe_ceiling(bytes);
    let batch = measure_batch(envs, workers, config);
    let width = f64::from(envs);

    println!("\n  {envs} envs, {workers} workers");
    println!("    stage            per env step     per batch step");
    println!("    simulate        {:>9.1} us", micros(single.simulate));
    println!("    encode          {:>9.1} us", micros(single.encode));
    println!(
        "    = batch step    {:>9.1} us     {:>7.2} ms   ({:.0} env steps/s)",
        micros(batch) / width,
        micros(batch) / 1e3,
        width / batch.as_secs_f64()
    );
    println!(
        "    transfer        {:>9.1} us     {:>7.2} ms   ({bytes} bytes, {:.0} MiB/s)",
        micros(transfer) / width,
        micros(transfer) / 1e3,
        mebibytes(bytes) / transfer.as_secs_f64()
    );
    println!(
        "    pipe alone      {:>9.1} us     {:>7.2} ms   ({:.0} MiB/s)",
        micros(ceiling) / width,
        micros(ceiling) / 1e3,
        mebibytes(bytes) / ceiling.as_secs_f64()
    );

    // Simulation and encoding are spread over the workers; transfer is not, so
    // the comparison that decides anything is transfer against the measured
    // batch step, not against the single-threaded work that went into it.
    let moving = transfer.as_secs_f64();
    let stepping = batch.as_secs_f64();
    println!(
        "    transfer is {:.1}x the batch step: {}",
        ratio_of(moving, stepping),
        if moving > stepping / 2.0 {
            "the trainer waits on the pipe as much as on the arenas"
        } else {
            "the arenas dominate; the pipe is not the bottleneck"
        }
    );

    // Two ways to spend less time there, priced before anyone writes either.
    // Packing sends fewer bytes and bumps the protocol version; a bulk-copy
    // codec sends the same bytes at the pipe's own speed and does not.
    let packed = moving * (1.0 - ratio(PACKED_OBSERVATION_BYTES, OBSERVATION_BYTES));
    let bulk = (moving - ceiling.as_secs_f64()).max(0.0);
    println!(
        "    ceilings: a bulk-copy codec saves up to {:.2} ms (same wire format), \
         packing channels 0-2 and 4 up to {:.2} ms (version bump)",
        bulk * 1e3,
        packed * 1e3
    );
}

/// One duration over another, for a plain "N times" comparison.
fn ratio_of(part: f64, whole: f64) -> f64 {
    if whole == 0.0 { 0.0 } else { part / whole }
}

/// A duration in microseconds.
fn micros(cost: Duration) -> f64 {
    cost.as_secs_f64() * 1e6
}

/// Bytes as mebibytes, for a human-sized rate.
#[expect(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    reason = "Byte counts here are megabytes, far inside f64's exactly representable integers"
)]
fn mebibytes(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

/// One byte count over another.
#[expect(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    reason = "Byte counts here are megabytes, far inside f64's exactly representable integers"
)]
fn ratio(part: usize, whole: usize) -> f64 {
    if whole == 0 {
        return 0.0;
    }
    part as f64 / whole as f64
}
