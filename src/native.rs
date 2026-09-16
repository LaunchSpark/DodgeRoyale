use std::{
    env,
    io::{BufWriter, Write},
    time::Duration,
};

use bevy::{
    app::{ScheduleRunnerPlugin, TerminalCtrlCHandlerPlugin},
    log::LogPlugin,
    prelude::*,
};
use clap::{Parser, Subcommand};
use color_eyre::eyre::{Result, WrapErr, eyre};

use dodge_royale::gym::server::serve;
use dodge_royale::gym::workers::BatchConfig;
use dodge_royale::rng::GameSeed;
use dodge_royale::simulation::{
    ArenaConfig, DEFAULT_HOLD_FRAMES, FRAME_SECONDS, HeadlessArena, StepResult,
};

use crate::startup;

#[derive(Parser)]
#[command(version, about)]
struct Args {
    /// Run without a window (also the default when built without graphics).
    #[arg(long)]
    headless: bool,
    /// Exercise the foundations and one headless Bevy frame, then exit.
    #[arg(long)]
    smoke_test: bool,
    /// Replay a run by fixing its random seed. Omit for a fresh random run.
    #[arg(long)]
    seed: Option<u64>,
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Serve a batch of headless arenas to a trainer over stdin and stdout.
    Gym(GymArgs),
}

#[derive(Parser)]
struct GymArgs {
    /// Arenas stepped together.
    #[arg(long, default_value_t = 8)]
    envs: u32,
    /// Threads owning those arenas.
    #[arg(long, default_value_t = 2)]
    threads: u32,
    /// Fixes every episode seed in the session.
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Enemies each arena is kept stocked with.
    #[arg(long, default_value_t = 100)]
    enemies: usize,
    /// Frames an episode may last before it is truncated.
    #[arg(long, default_value_t = 3_600)]
    max_frames: u32,
    /// Frames a predicted path assumes its action is held.
    #[arg(long, default_value_t = DEFAULT_HOLD_FRAMES)]
    hold_frames: u32,
}

pub fn run() -> Result<AppExit> {
    color_eyre::install()?;
    let args = Args::parse();

    // The gym is dispatched before anything else starts: it needs no database,
    // no async runtime and no window, and a trainer's stdout must not carry a
    // startup demo's output.
    if let Some(Command::Gym(gym)) = args.command {
        return run_gym(&gym);
    }

    let database_url = match env::var("DATABASE_URL") {
        Ok(url) => Some(url),
        Err(env::VarError::NotPresent) => None,
        Err(error) => return Err(error.into()),
    };
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .thread_name("dodge-io")
        .enable_all()
        .build()
        .wrap_err("Could not start the Tokio runtime")?;

    let report = runtime.block_on(startup::initialize(database_url.as_deref()))?;
    // Report the seed on every run, so an interesting one can be replayed.
    let seed = args.seed.map_or_else(GameSeed::from_entropy, GameSeed::new);

    if args.headless && !args.smoke_test {
        let exit = run_idle_episode(seed);
        runtime.shutdown_timeout(Duration::from_secs(3));
        return exit;
    }

    let mut app = build_app(args.headless || args.smoke_test);
    app.insert_resource(seed);
    app.add_systems(Startup, move || {
        info!("Foundation demo: {report}");
        info!("DodgeRoyale ready (seed {})", seed.get());
    });
    let exit = if args.smoke_test {
        app.finish();
        app.cleanup();
        app.update();
        info!("Smoke test passed");
        AppExit::Success
    } else {
        app.run()
    };
    runtime.shutdown_timeout(Duration::from_secs(3));
    Ok(exit)
}

/// Serve arenas to a trainer.
///
/// Stdin and stdout are locked and binary; every diagnostic goes to stderr.
fn run_gym(args: &GymArgs) -> Result<AppExit> {
    if args.envs == 0 {
        return Err(eyre!("--envs must be at least one"));
    }
    if args.threads == 0 {
        return Err(eyre!("--threads must be at least one"));
    }
    if args.max_frames == 0 {
        return Err(eyre!("--max-frames must be at least one"));
    }

    let config = BatchConfig {
        envs: args.envs,
        workers: args.threads,
        root_seed: args.seed,
        enemy_count: args.enemies,
        max_frames: args.max_frames,
        hold_frames: args.hold_frames,
    };

    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut reader = stdin.lock();
    let mut writer = BufWriter::new(stdout.lock());

    match serve(config, &mut reader, &mut writer) {
        Ok(()) => {
            writer
                .flush()
                .wrap_err("Could not flush the gym's output")?;
            Ok(AppExit::Success)
        }
        Err(error) => {
            // The ERROR record is already written; this is for the human.
            let _ = writer.flush();
            Err(eyre!("gym session failed: {error}"))
        }
    }
}

/// Play one episode with an idle player, and report how it went.
///
/// The headless entry point exists to prove the simulation runs without a
/// renderer, so it runs a real, finite episode rather than an empty loop.
fn run_idle_episode(seed: GameSeed) -> Result<AppExit> {
    let config = ArenaConfig {
        seed: seed.get(),
        ..ArenaConfig::default()
    };
    let mut arena =
        HeadlessArena::new(config).map_err(|error| eyre!("Could not start the arena: {error}"))?;
    // Printed rather than logged: this path runs no Bevy app, so there is no
    // LogPlugin to carry an info! anywhere.
    eprintln!(
        "Headless arena ready (seed {}, {} enemies, {} frame budget)",
        seed.get(),
        config.enemy_count,
        config.max_frames
    );

    let mut last = StepResult::default();
    while !arena.done() {
        last = arena
            .step()
            .map_err(|error| eyre!("The arena stopped: {error}"))?;
    }

    let seconds = f64::from(last.frame) * f64::from(FRAME_SECONDS);
    let ending = if last.terminated {
        "the player was hit"
    } else {
        "the frame budget ran out"
    };
    eprintln!(
        "Episode over after {} frames ({seconds:.1}s): {ending}",
        last.frame
    );
    Ok(AppExit::Success)
}

fn build_app(headless: bool) -> App {
    let mut app = App::new();
    #[cfg(feature = "graphics")]
    if !headless {
        return dodge_royale::game::build_app();
    }
    #[cfg(not(feature = "graphics"))]
    let _ = headless;

    app.add_plugins((
        MinimalPlugins.set(ScheduleRunnerPlugin::run_loop(Duration::from_secs_f64(
            1.0 / 60.0,
        ))),
        LogPlugin::default(),
        TerminalCtrlCHandlerPlugin,
    ));
    app
}
