use bevy::prelude::*;

use crate::art::shadow_translation;
pub(super) use crate::enemy::Velocity2d as Velocity;
pub(super) use crate::simulation::Player;
use crate::simulation::{PlayerIntent, PlayerSet, spawn_player_body};

use super::art::{ActiveTheme, ink};
use super::ghost::Ghosted;
use super::player_art::{PlayerArt, TrailEmitter};
use super::screen::{GameEntity, Screen, WatchMode};

/// How high the player draws above the arena floor.
const PLAYER_Z: f32 = 10.0;

pub(super) struct PlayerPlugin;

impl Plugin for PlayerPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(OnEnter(Screen::Playing), spawn_player)
            // The keyboard is one writer of intent; a policy would be another.
            // Either way the write lands before the simulation integrates it,
            // so a key pressed this frame moves the player this frame.
            .add_systems(
                Update,
                read_keyboard
                    .before(PlayerSet::Move)
                    .run_if(in_state(Screen::Playing)),
            );
    }
}

#[expect(
    clippy::needless_pass_by_value,
    reason = "Bevy injects system parameters by value"
)]
fn spawn_player(mut commands: Commands, theme: Res<ActiveTheme>, art: Res<PlayerArt>) {
    commands.spawn((
        Name::new("Player"),
        GameEntity,
        // The camera lags a seam crossing, so without this the player draws a
        // world away from it for a few frames and vanishes.
        Ghosted,
        spawn_player_body(Vec3::new(0.0, 0.0, PLAYER_Z)),
        TrailEmitter::default(),
        art.sprite(ink()),
        children![(
            Name::new("Player shadow"),
            art.sprite(theme.shadow()),
            Transform::from_translation(shadow_translation().with_z(-1.1)),
        )],
    ));
}

/// Turn the held keys into a direction for the simulation to integrate.
#[expect(
    clippy::needless_pass_by_value,
    reason = "Bevy system parameters are injected by value"
)]
pub(super) fn read_keyboard(
    keys: Res<ButtonInput<KeyCode>>,
    watch: Res<WatchMode>,
    mut players: Query<&mut PlayerIntent>,
) {
    if watch.0 {
        return;
    }
    let direction = Vec2::new(
        input_axis(
            keys.any_pressed([KeyCode::KeyA, KeyCode::ArrowLeft]),
            keys.any_pressed([KeyCode::KeyD, KeyCode::ArrowRight]),
        ),
        input_axis(
            keys.any_pressed([KeyCode::KeyS, KeyCode::ArrowDown]),
            keys.any_pressed([KeyCode::KeyW, KeyCode::ArrowUp]),
        ),
    );
    for mut intent in &mut players {
        intent.0 = direction;
    }
}

const fn input_axis(negative: bool, positive: bool) -> f32 {
    match (negative, positive) {
        (true, false) => -1.0,
        (false, true) => 1.0,
        _ => 0.0,
    }
}
