use std::{env, time::Duration};

use bevy::{
    app::{ScheduleRunnerPlugin, TerminalCtrlCHandlerPlugin},
    log::LogPlugin,
    prelude::*,
};
use clap::Parser;
use color_eyre::eyre::{Result, WrapErr};

use dodge_royale::rng::GameSeed;

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
}

pub fn run() -> Result<AppExit> {
    color_eyre::install()?;
    let args = Args::parse();
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
