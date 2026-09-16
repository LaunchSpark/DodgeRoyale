//! Shared browser and native game presentation.

use bevy::prelude::*;

mod art;
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
    let mut app = App::new();
    app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            title: "DodgeRoyale".to_owned(),
            resolution: (1280, 720).into(),
            canvas: Some("#game".to_owned()),
            fit_canvas_to_parent: true,
            prevent_default_event_handling: true,
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
    app
}
