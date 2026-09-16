//! Arenas owned by persistent threads, stepped as one batch.
//!
//! A Bevy `App` is `!Send`, so an arena cannot be moved between threads and
//! cannot be handed to a work-stealing pool. Each worker therefore creates its
//! own arenas, keeps them for the life of the session, and drops them on the
//! thread that made them. Only actions and results cross a channel, and both
//! are owned data.
//!
//! Results are gathered by env index, never by completion order, so the bytes a
//! client sees do not depend on how many workers there are or which one
//! finished first.

use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::thread::{self, JoinHandle};

use crate::observation::{OBSERVATION_VALUES, encode};
use crate::simulation::{ArenaConfig, ArenaError, HeadlessArena};

use super::protocol::{EnvTransition, ResetBatch, StepBatch, TerminalObservation, episode_seed};

/// How the batch is built.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchConfig {
    pub envs: u32,
    pub workers: u32,
    pub root_seed: u64,
    pub enemy_count: usize,
    pub max_frames: u32,
    pub hold_frames: u32,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            envs: 8,
            workers: 2,
            root_seed: 0,
            enemy_count: 100,
            max_frames: 3_600,
            hold_frames: crate::simulation::DEFAULT_HOLD_FRAMES,
        }
    }
}

/// Why a batch could not start, step or reset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BatchError {
    /// A configuration that cannot describe a batch.
    InvalidConfig(&'static str),
    /// An arena refused to start or step, reported with its env.
    Arena { env: u32, reason: String },
    /// A worker died or its channel closed.
    WorkerLost(String),
}

impl core::fmt::Display for BatchError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidConfig(reason) => write!(formatter, "invalid batch config: {reason}"),
            Self::Arena { env, reason } => write!(formatter, "env {env}: {reason}"),
            Self::WorkerLost(reason) => write!(formatter, "worker lost: {reason}"),
        }
    }
}

impl core::error::Error for BatchError {}

/// What one env did during one command.
#[derive(Debug, Clone, PartialEq)]
struct EnvOutcome {
    env: u32,
    transition: EnvTransition,
    /// What the env is on now: after an auto-reset, the new episode's frame.
    observation: Vec<f32>,
    /// The finished episode's last frame, when one ended.
    terminal: Option<Vec<f32>>,
}

/// What a worker is asked to do.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Command {
    /// One action per env in this worker's shard, in shard order.
    Step(Vec<u8>),
    /// Restart every env in the shard. `Some` restarts the seed stream.
    Reset(Option<u64>),
    Shutdown,
}

type Reply = Result<Vec<EnvOutcome>, BatchError>;

/// One env and the episode counter that names its seeds.
struct EnvArena {
    env: u32,
    arena: HeadlessArena,
    root_seed: u64,
    episode: u64,
    hold_frames: u32,
}

impl EnvArena {
    fn start(env: u32, config: BatchConfig) -> Result<Self, BatchError> {
        let seed = episode_seed(config.root_seed, env, 0);
        let arena = HeadlessArena::new(ArenaConfig {
            seed,
            enemy_count: config.enemy_count,
            max_frames: config.max_frames,
            ..ArenaConfig::default()
        })
        .map_err(|error| BatchError::Arena {
            env,
            reason: error.to_string(),
        })?;
        Ok(Self {
            env,
            arena,
            root_seed: config.root_seed,
            episode: 0,
            hold_frames: config.hold_frames,
        })
    }

    fn fault(&self, error: &ArenaError) -> BatchError {
        BatchError::Arena {
            env: self.env,
            reason: error.to_string(),
        }
    }

    fn observe(&mut self) -> Result<Vec<f32>, BatchError> {
        let mut buffer = vec![0.0; OBSERVATION_VALUES];
        let view = self.arena.view();
        encode(&view, self.hold_frames, &mut buffer).map_err(|error| self.fault(&error))?;
        Ok(buffer)
    }

    const fn current_seed(&self) -> u64 {
        episode_seed(self.root_seed, self.env, self.episode)
    }

    /// Step once, auto-resetting if the episode ends.
    fn step(&mut self, action: u8) -> Result<EnvOutcome, BatchError> {
        let episode_seed_before = self.current_seed();
        self.arena
            .set_action(action)
            .map_err(|error| self.fault(&error))?;
        let result = self.arena.step().map_err(|error| self.fault(&error))?;

        let mut transition = EnvTransition {
            frame: result.frame,
            terminated: result.terminated,
            truncated: result.truncated,
            enemy_deaths: result.enemy_deaths,
            episode_seed: episode_seed_before,
            reset_seed: None,
        };
        let mut terminal = None;
        if result.done() {
            // The finished episode's last frame is copied before the arena is
            // rebuilt; a learner needs it to bootstrap a truncation.
            terminal = Some(self.observe()?);
            self.episode = self.episode.saturating_add(1);
            let seed = self.current_seed();
            self.arena.reset(seed).map_err(|error| self.fault(&error))?;
            transition.reset_seed = Some(seed);
        }
        let observation = self.observe()?;
        Ok(EnvOutcome {
            env: self.env,
            transition,
            observation,
            terminal,
        })
    }

    /// Restart. `Some` root restarts the stream; `None` advances it.
    fn reset(&mut self, root: Option<u64>) -> Result<EnvOutcome, BatchError> {
        match root {
            Some(root) => {
                self.root_seed = root;
                self.episode = 0;
            }
            None => self.episode = self.episode.saturating_add(1),
        }
        let seed = self.current_seed();
        self.arena.reset(seed).map_err(|error| self.fault(&error))?;
        let observation = self.observe()?;
        Ok(EnvOutcome {
            env: self.env,
            transition: EnvTransition {
                episode_seed: seed,
                ..EnvTransition::default()
            },
            observation,
            terminal: None,
        })
    }
}

/// One thread, its arenas, and the channels it answers on.
struct Worker {
    envs: Vec<u32>,
    commands: SyncSender<Command>,
    replies: Receiver<Reply>,
    handle: Option<JoinHandle<()>>,
}

/// Every env, stepped together.
pub struct ArenaBatch {
    workers: Vec<Worker>,
    envs: u32,
}

impl ArenaBatch {
    /// Start the workers and their arenas.
    ///
    /// # Errors
    ///
    /// [`BatchError::InvalidConfig`] for a batch that cannot exist, and
    /// [`BatchError::Arena`] when an arena refuses to start.
    pub fn start(config: BatchConfig) -> Result<(Self, ResetBatch), BatchError> {
        if config.envs == 0 {
            return Err(BatchError::InvalidConfig("a batch needs at least one env"));
        }
        if config.workers == 0 {
            return Err(BatchError::InvalidConfig(
                "a batch needs at least one worker",
            ));
        }
        let shards = shard_envs(config.envs, config.workers);
        let mut workers = Vec::with_capacity(shards.len());
        for (index, envs) in shards.into_iter().enumerate() {
            // Bounded at one batch in flight: the coordinator asks, the worker
            // answers, and neither can run ahead of the other.
            let (command_tx, command_rx) = sync_channel::<Command>(1);
            let (reply_tx, reply_rx) = sync_channel::<Reply>(1);
            let shard = envs.clone();
            let handle = thread::Builder::new()
                .name(format!("dodge-arena-{index}"))
                .spawn(move || run_worker(shard, config, &command_rx, &reply_tx))
                .map_err(|error| BatchError::WorkerLost(error.to_string()))?;
            workers.push(Worker {
                envs,
                commands: command_tx,
                replies: reply_rx,
                handle: Some(handle),
            });
        }

        let batch = Self {
            workers,
            envs: config.envs,
        };
        // Every worker reports its arenas' first frame, or the reason it has
        // none. Collecting it here means a failure to start is a failure to
        // start the session, not a surprise on the first step.
        let outcomes = batch.collect_replies()?;
        Ok((batch, reset_batch(&outcomes)))
    }

    /// How many envs this batch steps.
    #[must_use]
    pub const fn envs(&self) -> u32 {
        self.envs
    }

    /// Step every env once.
    ///
    /// # Errors
    ///
    /// As [`BatchError`]; an env that faults ends the batch rather than
    /// returning a partial result.
    pub fn step(&mut self, actions: &[u8]) -> Result<StepBatch, BatchError> {
        if actions.len() != usize::try_from(self.envs).unwrap_or(usize::MAX) {
            return Err(BatchError::InvalidConfig(
                "one action per env, in env order",
            ));
        }
        for worker in &self.workers {
            let shard: Vec<u8> = worker
                .envs
                .iter()
                .map(|env| {
                    actions
                        .get(usize::try_from(*env).unwrap_or(usize::MAX))
                        .copied()
                        .unwrap_or(0)
                })
                .collect();
            worker
                .commands
                .send(Command::Step(shard))
                .map_err(|error| BatchError::WorkerLost(error.to_string()))?;
        }
        let outcomes = self.collect_replies()?;
        Ok(step_batch(&outcomes))
    }

    /// Restart every env.
    ///
    /// # Errors
    ///
    /// As [`BatchError`].
    pub fn reset(&mut self, seed: Option<u64>) -> Result<ResetBatch, BatchError> {
        for worker in &self.workers {
            worker
                .commands
                .send(Command::Reset(seed))
                .map_err(|error| BatchError::WorkerLost(error.to_string()))?;
        }
        let outcomes = self.collect_replies()?;
        Ok(reset_batch(&outcomes))
    }

    /// Gather every worker's shard and order the results by env.
    fn collect_replies(&self) -> Result<Vec<EnvOutcome>, BatchError> {
        let mut outcomes = Vec::new();
        let mut failure = None;
        // Every worker is drained even after one fails, so no worker is left
        // blocked on a send into a channel nobody reads.
        for worker in &self.workers {
            match worker.replies.recv() {
                Ok(Ok(shard)) => outcomes.extend(shard),
                Ok(Err(error)) => failure = failure.or(Some(error)),
                Err(error) => {
                    failure =
                        failure.or_else(|| Some(BatchError::WorkerLost(error.to_string())));
                }
            }
        }
        if let Some(error) = failure {
            return Err(error);
        }
        outcomes.sort_by_key(|outcome| outcome.env);
        Ok(outcomes)
    }
}

impl Drop for ArenaBatch {
    fn drop(&mut self) {
        for worker in &self.workers {
            // A worker waiting to be asked exits on the closed channel even if
            // this send cannot be delivered.
            let _ = worker.commands.send(Command::Shutdown);
        }
        for worker in &mut self.workers {
            // Drain anything in flight first: a worker blocked sending into a
            // full channel would never reach its own exit.
            while worker.replies.try_recv().is_ok() {}
            if let Some(handle) = worker.handle.take() {
                let _ = handle.join();
            }
        }
    }
}

/// Give each worker a stable shard of env indices.
#[expect(
    clippy::arithmetic_side_effects,
    reason = "The divisor is clamped to at least one just above"
)]
fn shard_envs(envs: u32, workers: u32) -> Vec<Vec<u32>> {
    let workers = workers.min(envs).max(1);
    let mut shards: Vec<Vec<u32>> = (0..workers).map(|_| Vec::new()).collect();
    for env in 0..envs {
        let index = usize::try_from(env % workers).unwrap_or(0);
        if let Some(shard) = shards.get_mut(index) {
            shard.push(env);
        }
    }
    shards
}

/// One worker's whole life: build its arenas, then answer until told to stop.
fn run_worker(
    envs: Vec<u32>,
    config: BatchConfig,
    commands: &Receiver<Command>,
    replies: &SyncSender<Reply>,
) {
    let mut arenas = Vec::with_capacity(envs.len());
    for env in envs {
        match EnvArena::start(env, config) {
            Ok(arena) => arenas.push(arena),
            Err(error) => {
                // The coordinator is waiting for this worker's first reply, so
                // a failure to build is reported rather than silently ending.
                let _ = replies.send(Err(error));
                return;
            }
        }
    }

    let first: Reply = arenas
        .iter_mut()
        .map(|arena| {
            arena.observe().map(|observation| EnvOutcome {
                env: arena.env,
                transition: EnvTransition {
                    episode_seed: arena.current_seed(),
                    ..EnvTransition::default()
                },
                observation,
                terminal: None,
            })
        })
        .collect();
    if replies.send(first).is_err() {
        return;
    }

    while let Ok(command) = commands.recv() {
        let reply: Reply = match command {
            Command::Shutdown => return,
            Command::Step(actions) => arenas
                .iter_mut()
                .zip(actions)
                .map(|(arena, action)| arena.step(action))
                .collect(),
            Command::Reset(seed) => arenas.iter_mut().map(|arena| arena.reset(seed)).collect(),
        };
        if replies.send(reply).is_err() {
            return;
        }
    }
}

fn step_batch(outcomes: &[EnvOutcome]) -> StepBatch {
    let mut observations = Vec::with_capacity(outcomes.len().saturating_mul(OBSERVATION_VALUES));
    let mut transitions = Vec::with_capacity(outcomes.len());
    let mut terminal = Vec::new();
    for outcome in outcomes {
        transitions.push(outcome.transition);
        observations.extend_from_slice(&outcome.observation);
        if let Some(final_frame) = outcome.terminal.as_ref() {
            terminal.push(TerminalObservation {
                env: outcome.env,
                observation: final_frame.clone(),
            });
        }
    }
    StepBatch {
        transitions,
        observations,
        terminal,
    }
}

fn reset_batch(outcomes: &[EnvOutcome]) -> ResetBatch {
    let mut observations = Vec::with_capacity(outcomes.len().saturating_mul(OBSERVATION_VALUES));
    let mut seeds = Vec::with_capacity(outcomes.len());
    for outcome in outcomes {
        seeds.push(outcome.transition.episode_seed);
        observations.extend_from_slice(&outcome.observation);
    }
    ResetBatch {
        seeds,
        observations,
    }
}

#[cfg(test)]
mod tests;
