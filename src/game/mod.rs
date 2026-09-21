//! Shared browser and native game presentation.

use bevy::prelude::*;

mod art;
#[cfg(target_arch = "wasm32")]
mod autopilot;
mod camera;
mod config;
mod enemy;
mod ghost;
mod menu;
mod player;
mod player_art;
mod screen;
mod shrapnel;
mod text;
mod world;

/// Build the interactive game without native infrastructure or a database.
pub fn build_app() -> App {
    #[cfg(target_arch = "wasm32")]
    let watch = autopilot::watch_enabled();
    #[cfg(not(target_arch = "wasm32"))]
    let watch = false;
    let mut app = App::new();
    app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            title: "DodgeRoyale".to_owned(),
            resolution: (1280, 720).into(),
            canvas: Some("#game".to_owned()),
            fit_canvas_to_parent: true,
            prevent_default_event_handling: !watch,
            ..default()
        }),
        ..default()
    }))
    .init_resource::<art::ActiveTheme>()
    .insert_resource(ClearColor(art::ActiveTheme::default().background()))
    .add_plugins((
        screen::ScreenPlugin,
        menu::MenuPlugin,
        config::ConfigPlugin,
        world::WorldPlugin,
        player::PlayerPlugin,
        player_art::PlayerArtPlugin,
        camera::FollowCameraPlugin,
        enemy::GameEnemyPlugin,
        shrapnel::ShrapnelPlugin,
        ghost::GhostPlugin,
    ));
    #[cfg(target_arch = "wasm32")]
    app.add_plugins(autopilot::AutopilotPlugin);
    app
}
