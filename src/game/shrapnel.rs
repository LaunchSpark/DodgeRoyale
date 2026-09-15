//! Little squares thrown out when an enemy dies.

use bevy::prelude::*;

use crate::{
    art::PIXEL,
    enemy::{Dying, Enemy, EnemySet},
    rng::{Domain, GameSeed},
    shatter::{PIECES, SPEED, burst_direction, is_alive, shrink_at},
};

use super::{
    art::{ActiveTheme, ink, shadowed_rect},
    ghost::Ghosted,
    screen::{GameEntity, Screen},
};

/// Drawn above the arena and the player, like the cart's particles.
const SHRAPNEL_Z: f32 = 12.0;

/// One piece of a shattered enemy.
#[derive(Component)]
struct Shrapnel {
    velocity: Vec2,
    age: f32,
}

/// Angles for the burst. Seeded from entropy so each death differs.
#[derive(Resource, Default)]
struct ShrapnelRng(fastrand::Rng);

pub(super) struct ShrapnelPlugin;

impl Plugin for ShrapnelPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ShrapnelRng>()
            .add_systems(PreStartup, seed_shrapnel)
            .add_systems(
                Update,
                (spawn_shrapnel.in_set(EnemySet::Effects), advance_shrapnel)
                    .run_if(in_state(Screen::Playing)),
            );
    }
}

/// Point the burst angles at this run's shatter stream.
#[expect(
    clippy::needless_pass_by_value,
    reason = "Bevy injects system parameters by value"
)]
fn seed_shrapnel(seed: Res<GameSeed>, mut random: ResMut<ShrapnelRng>) {
    random.0 = seed.rng(Domain::Shatter);
}

/// Scatter a burst wherever an enemy has just been marked for death.
///
/// `Dying` is added in `EnemySet::Deaths` and the entity is removed in
/// `EnemySet::Cleanup`, so `Effects` is the one window where its position is
/// still readable.
#[expect(
    clippy::needless_pass_by_value,
    reason = "Bevy injects system parameters by value"
)]
#[expect(
    clippy::arithmetic_side_effects,
    reason = "Scaling a unit vector by a finite speed cannot overflow"
)]
fn spawn_shrapnel(
    mut commands: Commands,
    theme: Res<ActiveTheme>,
    mut random: ResMut<ShrapnelRng>,
    dying: Query<&Transform, (With<Enemy>, Added<Dying>)>,
) {
    let shadow = theme.shadow();
    for transform in &dying {
        let origin = transform.translation.truncate();
        for _ in 0..PIECES {
            let velocity = burst_direction(random.0.f32()) * SPEED;
            let piece = commands
                .spawn((
                    GameEntity,
                    Ghosted,
                    Shrapnel { velocity, age: 0.0 },
                    Transform::from_translation(origin.extend(SHRAPNEL_Z)),
                    Visibility::default(),
                ))
                .id();
            shadowed_rect(
                &mut commands,
                piece,
                Vec2::ZERO,
                Vec2::splat(PIXEL),
                ink(),
                shadow,
                0.0,
            );
        }
    }
}

/// Carry each piece outwards, shrinking it until it disappears.
#[expect(
    clippy::needless_pass_by_value,
    reason = "Bevy injects system parameters by value"
)]
#[expect(
    clippy::arithmetic_side_effects,
    reason = "Velocity is a finite vector and elapsed time is bounded per frame"
)]
fn advance_shrapnel(
    mut commands: Commands,
    time: Res<Time>,
    mut pieces: Query<(Entity, &mut Shrapnel, &mut Transform)>,
) {
    let delta = time.delta_secs();
    for (entity, mut piece, mut transform) in &mut pieces {
        piece.age += delta;
        let scale = shrink_at(piece.age);
        if is_alive(scale, piece.age) {
            let step = piece.velocity * delta;
            let flown = crate::torus::wrap_position(
                transform.translation.truncate() + step,
                crate::scale::WORLD_HALF_EXTENTS,
            );
            transform.translation.x = flown.x;
            transform.translation.y = flown.y;
            transform.scale = Vec3::new(scale, scale, 1.0);
        } else {
            commands.entity(entity).despawn();
        }
    }
}
