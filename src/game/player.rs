use bevy::prelude::*;

use crate::art::shadow_translation;
pub(super) use crate::enemy::Velocity2d as Velocity;
use crate::enemy_types::Defeated;
pub(super) use crate::motion::PLAYER_HALF_SIZE;
use crate::motion::advance_motion;
use crate::{collision::Collider, enemy::EnemyTarget};

use super::art::{ActiveTheme, ink};
use super::ghost::Ghosted;
use super::player_art::{PlayerArt, TrailEmitter};
use super::screen::{GameEntity, Screen};

#[derive(Component)]
pub(super) struct Player;

pub(super) struct PlayerPlugin;

impl Plugin for PlayerPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(OnEnter(Screen::Playing), spawn_player)
            .add_systems(Update, move_player.run_if(in_state(Screen::Playing)));
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
        Player,
        EnemyTarget,
        Collider::rectangle(Vec2::splat(PLAYER_HALF_SIZE)),
        Velocity::default(),
        TrailEmitter::default(),
        art.sprite(ink()),
        Transform::from_xyz(0.0, 0.0, 10.0),
        children![(
            Name::new("Player shadow"),
            art.sprite(theme.shadow()),
            Transform::from_translation(shadow_translation().with_z(-1.1)),
        )],
    ));
}

#[expect(
    clippy::needless_pass_by_value,
    reason = "Bevy system parameters are injected by value"
)]
pub(super) fn move_player(
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    player: Single<(&mut Transform, &mut Velocity, Option<&Defeated>), With<Player>>,
) {
    let (mut transform, mut velocity, defeated) = player.into_inner();
    if defeated.is_some() {
        velocity.0 = Vec2::ZERO;
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
    let position = advance_motion(
        transform.translation.truncate(),
        &mut velocity.0,
        direction,
        time.delta_secs(),
    );
    transform.translation.x = position.x;
    transform.translation.y = position.y;
}

const fn input_axis(negative: bool, positive: bool) -> f32 {
    match (negative, positive) {
        (true, false) => -1.0,
        (false, true) => 1.0,
        _ => 0.0,
    }
}
