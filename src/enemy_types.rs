//! Hostile reference personalities. Power-ups are intentionally not connected.

use crate::{
    collision::Collider,
    enemy::{Dying, Enemy, EnemyMotion, EnemySet, EnemyState, EnemyTarget, EnemyWorld, Velocity2d},
    torus::distance_in,
};
use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// The reference's hostile personalities (`p = 0` and `p = 1`).
#[derive(Component, Default, Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EnemyKind {
    #[default]
    Normal,
    Kamikaze,
}

/// The reference's `p = -1`: a stationary transient hazard, outside the population.
#[derive(Component, Default)]
#[require(Transform, Collider)]
pub struct KamikazeBlast {
    pub age: f32,
}

#[derive(Component)]
struct KamikazeMotor {
    multiplier: f32,
}

/// Marks a defeated player; movement stops until the gameplay screen is reset.
#[derive(Component)]
pub struct Defeated;

/// A single lethal contact notification per player life.
#[derive(Message)]
pub struct PlayerHit {
    pub target: Entity,
}

/// Reference timings and geometry at 60 Hz and 6.25 world units per pixel.
/// Supply finite, positive durations and dimensions.
#[derive(Resource)]
pub struct KamikazeSettings {
    pub near_distance: f32,
    pub motor_rate: f32,
    pub blast_seconds: f32,
    pub blast_width: f32,
}

impl Default for KamikazeSettings {
    fn default() -> Self {
        Self {
            near_distance: 156.25,
            motor_rate: 0.6,
            blast_seconds: 1.0,
            blast_width: 187.5,
        }
    }
}

pub struct ReferenceEnemyPlugin;

impl Plugin for ReferenceEnemyPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<KamikazeSettings>()
            .add_message::<PlayerHit>()
            .add_systems(
                Update,
                (initialize_kamikazes, update_blasts)
                    .chain()
                    .in_set(EnemySet::Prepare),
            )
            .add_systems(Update, apply_kamikaze_steering.in_set(EnemySet::Modifiers))
            .add_systems(
                Update,
                (resolve_contacts, detonate_dead)
                    .chain()
                    .in_set(EnemySet::Effects),
            );
    }
}

type NewKamikazes<'w, 's> =
    Query<'w, 's, (Entity, &'static EnemyKind), (With<Enemy>, Without<KamikazeMotor>)>;

fn initialize_kamikazes(mut commands: Commands, enemies: NewKamikazes) {
    for (entity, kind) in &enemies {
        if *kind == EnemyKind::Kamikaze {
            commands
                .entity(entity)
                .insert(KamikazeMotor { multiplier: 1.0 });
        }
    }
}

type MovingKamikazes<'w, 's> = Query<
    'w,
    's,
    (
        &'static Transform,
        &'static EnemyState,
        &'static mut EnemyMotion,
        &'static mut KamikazeMotor,
    ),
    With<Enemy>,
>;

#[expect(
    clippy::needless_pass_by_value,
    reason = "Bevy injects system parameters by value"
)]
fn apply_kamikaze_steering(
    time: Res<Time>,
    settings: Res<KamikazeSettings>,
    world: Res<EnemyWorld>,
    targets: Query<&Transform, (With<EnemyTarget>, Without<Enemy>)>,
    mut enemies: MovingKamikazes,
) {
    for (transform, state, mut motion, mut motor) in &mut enemies {
        let near = state
            .target
            .and_then(|entity| targets.get(entity).ok())
            .is_some_and(|target| {
                distance_in(
                    transform.translation.truncate(),
                    target.translation.truncate(),
                    world.half_extents,
                ) <= settings.near_distance
            });
        let change = settings.motor_rate * time.delta_secs().min(0.05);
        motor.multiplier =
            (motor.multiplier + if near { -change } else { change }).clamp(-1.0, 1.0);
        motion.speed_scale = motor.multiplier;
    }
}

type AffectedTargets<'w, 's> = Query<
    'w,
    's,
    (&'static mut Collider, &'static mut Velocity2d),
    (With<EnemyTarget>, Without<Enemy>, Without<Defeated>),
>;

fn hit_player(
    entity: Entity,
    commands: &mut Commands,
    targets: &mut AffectedTargets,
    hits: &mut MessageWriter<PlayerHit>,
) {
    if let Ok((mut collider, mut velocity)) = targets.get_mut(entity)
        && collider.enabled
    {
        collider.enabled = false;
        velocity.0 = Vec2::ZERO;
        commands.entity(entity).insert(Defeated);
        hits.write(PlayerHit { target: entity });
    }
}

fn resolve_contacts(
    mut commands: Commands,
    mut contacts: MessageReader<crate::enemy::EnemyContact>,
    mut targets: AffectedTargets,
    mut hits: MessageWriter<PlayerHit>,
) {
    for contact in contacts.read() {
        hit_player(contact.target, &mut commands, &mut targets, &mut hits);
        commands.entity(contact.enemy).insert(Dying);
    }
}

type DeadKinds<'w, 's> =
    Query<'w, 's, (&'static Transform, &'static EnemyKind), (With<Enemy>, With<Dying>)>;

fn detonate_dead(mut commands: Commands, enemies: DeadKinds) {
    for (transform, kind) in &enemies {
        if *kind == EnemyKind::Kamikaze {
            commands.spawn((
                Name::new("Kamikaze blast"),
                KamikazeBlast::default(),
                Collider::rectangle(Vec2::splat(0.001)),
                *transform,
            ));
        }
    }
}

type BlastTargets<'w, 's> = Query<
    'w,
    's,
    (Entity, &'static Transform, &'static Collider),
    (With<EnemyTarget>, Without<KamikazeBlast>),
>;

#[expect(
    clippy::needless_pass_by_value,
    reason = "Bevy injects system parameters by value"
)]
fn update_blasts(
    mut commands: Commands,
    time: Res<Time>,
    settings: Res<KamikazeSettings>,
    world: Res<EnemyWorld>,
    mut blasts: Query<
        (Entity, &Transform, &mut Collider, &mut KamikazeBlast),
        Without<EnemyTarget>,
    >,
    mut targets: ParamSet<(BlastTargets, AffectedTargets)>,
    mut hits: MessageWriter<PlayerHit>,
) {
    let delta = time.delta_secs().min(0.05);
    let mut victims = BTreeSet::new();
    for (entity, transform, mut collider, mut blast) in &mut blasts {
        blast.age += delta;
        if blast.age >= settings.blast_seconds {
            commands.entity(entity).despawn();
            continue;
        }
        let progress = (blast.age / settings.blast_seconds).clamp(0.0, 1.0);
        let width = settings.blast_width * (1.0 - 2.0_f32.mul_add(progress, -1.0).abs());
        collider.half_extents = Vec2::splat((width * 0.5).max(0.001));
        for (target, target_transform, target_collider) in &targets.p0() {
            if collider.overlaps_wrapped(
                transform.translation.truncate(),
                *target_collider,
                target_transform.translation.truncate(),
                world.half_extents,
            ) {
                victims.insert(target);
            }
        }
    }
    for entity in victims {
        hit_player(entity, &mut commands, &mut targets.p1(), &mut hits);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        enemy::{BoundaryMode, EnemyPlugin, EnemySettings, SteeringMode},
        enemy_population::{
            EnemyPopulation, EnemyPopulationPlugin, EnemySpawnQueue, reference_archetypes,
        },
    };
    use std::time::Duration;

    fn app() -> App {
        let mut app = App::new();
        app.insert_resource(Time::<()>::default())
            .add_plugins((EnemyPlugin, ReferenceEnemyPlugin));
        app
    }

    fn step(app: &mut App, seconds: f32) {
        app.world_mut()
            .resource_mut::<Time>()
            .advance_by(Duration::from_secs_f32(seconds));
        app.update();
    }

    #[test]
    fn normal_contact_defeats_player_once_without_spawning_a_blast() {
        let mut app = app();
        let player = app
            .world_mut()
            .spawn((EnemyTarget, Velocity2d(Vec2::X)))
            .id();
        let enemy = app.world_mut().spawn((Enemy, EnemyKind::Normal)).id();
        let mut cursor = app.world().resource::<Messages<PlayerHit>>().get_cursor();
        step(&mut app, 0.0);
        assert!(app.world().get::<Defeated>(player).is_some());
        assert!(!app.world().get::<Collider>(player).unwrap().enabled);
        assert_eq!(app.world().get::<Velocity2d>(player).unwrap().0, Vec2::ZERO);
        assert!(app.world().get_entity(enemy).is_err());
        assert_eq!(
            cursor
                .read(app.world().resource::<Messages<PlayerHit>>())
                .count(),
            1
        );
        assert_eq!(
            app.world_mut()
                .query::<&KamikazeBlast>()
                .iter(app.world())
                .count(),
            0
        );
        app.world_mut().spawn((Enemy, EnemyKind::Normal));
        step(&mut app, 0.0);
        assert_eq!(
            cursor
                .read(app.world().resource::<Messages<PlayerHit>>())
                .count(),
            0
        );
    }

    #[test]
    fn kamikaze_collision_leaves_one_blast_and_blast_does_not_destroy_enemies() {
        let mut app = app();
        let kamikaze = app.world_mut().spawn((Enemy, EnemyKind::Kamikaze)).id();
        let other = app.world_mut().spawn((Enemy, EnemyKind::Normal)).id();
        step(&mut app, 0.0);
        assert!(app.world().get_entity(kamikaze).is_err());
        assert!(app.world().get_entity(other).is_err());
        assert_eq!(
            app.world_mut()
                .query::<&KamikazeBlast>()
                .iter(app.world())
                .count(),
            1
        );
        let survivor = app.world_mut().spawn((Enemy, EnemyKind::Normal)).id();
        step(&mut app, 0.05);
        assert!(app.world().get_entity(survivor).is_ok());
        assert_eq!(
            app.world_mut()
                .query::<&KamikazeBlast>()
                .iter(app.world())
                .count(),
            1
        );
    }

    #[test]
    fn blast_grows_then_shrinks_and_expires_after_one_second() {
        let mut app = app();
        let blast = app.world_mut().spawn(KamikazeBlast::default()).id();
        for _ in 0..5 {
            step(&mut app, 0.05);
        }
        let quarter = app.world().get::<Collider>(blast).unwrap().half_extents.x;
        for _ in 0..5 {
            step(&mut app, 0.05);
        }
        let peak = app.world().get::<Collider>(blast).unwrap().half_extents.x;
        assert!((peak - 93.75).abs() < 0.001);
        assert!(peak > quarter);
        for _ in 0..5 {
            step(&mut app, 0.05);
        }
        assert!(
            (app.world().get::<Collider>(blast).unwrap().half_extents.x - quarter).abs() < 0.001
        );
        for _ in 0..6 {
            step(&mut app, 0.05);
        }
        assert!(app.world().get_entity(blast).is_err());
    }

    #[test]
    fn blast_interior_is_lethal_even_though_art_is_an_outline() {
        let mut app = app();
        let player = app
            .world_mut()
            .spawn((EnemyTarget, Transform::from_xyz(40.0, 0.0, 0.0)))
            .id();
        app.world_mut().spawn(KamikazeBlast::default());
        for _ in 0..10 {
            step(&mut app, 0.05);
        }
        assert!(app.world().get::<Defeated>(player).is_some());
    }

    #[test]
    fn kamikaze_slows_and_reverses_near_player_then_recovers() {
        let mut app = app();
        let player = app.world_mut().spawn(EnemyTarget).id();
        let kamikaze = app
            .world_mut()
            .spawn((
                Enemy,
                EnemyKind::Kamikaze,
                EnemySettings {
                    speed: 0.0,
                    ..default()
                },
                Transform::from_xyz(100.0, 0.0, 0.0),
            ))
            .id();
        for _ in 0..20 {
            step(&mut app, 0.05);
        }
        let slow = app
            .world()
            .get::<EnemyMotion>(kamikaze)
            .unwrap()
            .speed_scale;
        assert!(slow > 0.0 && slow < 1.0);
        for _ in 0..50 {
            step(&mut app, 0.05);
        }
        assert!(
            app.world()
                .get::<EnemyMotion>(kamikaze)
                .unwrap()
                .speed_scale
                < 0.0
        );
        app.world_mut()
            .get_mut::<Transform>(player)
            .unwrap()
            .translation
            .x = 1_000.0;
        for _ in 0..70 {
            step(&mut app, 0.05);
        }
        assert!(
            (app.world()
                .get::<EnemyMotion>(kamikaze)
                .unwrap()
                .speed_scale
                - 1.0)
                .abs()
                < 0.001
        );
    }

    #[test]
    fn reference_types_wrap_at_arena_edges() {
        let mut app = app();
        let enemy = app
            .world_mut()
            .spawn((
                Enemy,
                EnemyKind::Normal,
                EnemySettings {
                    steering: SteeringMode::Manual,
                    boundary: BoundaryMode::Wrap,
                    ..default()
                },
                Transform::from_xyz(2_999.0, 0.0, 0.0),
                Velocity2d(Vec2::new(500.0, 0.0)),
            ))
            .id();
        step(&mut app, 0.05);
        let moved = app.world().get::<Transform>(enemy).unwrap().translation.x;
        assert!(moved < 0.0, "expected a wrap to the left edge, got {moved}");
        // A seam is not a wall, so the enemy keeps its speed through it.
        assert!(app.world().get::<Velocity2d>(enemy).unwrap().0.x > 0.0);
    }

    #[test]
    fn reference_types_bounce_at_arena_edges() {
        let mut app = app();
        let enemy = app
            .world_mut()
            .spawn((
                Enemy,
                EnemyKind::Normal,
                EnemySettings {
                    steering: SteeringMode::Manual,
                    boundary: BoundaryMode::Bounce,
                    ..default()
                },
                Transform::from_xyz(2_987.0, 0.0, 0.0),
                Velocity2d(Vec2::new(100.0, 0.0)),
            ))
            .id();
        step(&mut app, 0.05);
        assert!(app.world().get::<Velocity2d>(enemy).unwrap().0.x < 0.0);
        assert!(app.world().get::<Transform>(enemy).unwrap().translation.x <= 2_988.0);
    }

    #[test]
    fn catalog_has_only_hostiles_with_reference_size_and_personality_weights() {
        let types = reference_archetypes();
        assert_eq!(types.len(), 8);
        let normal: u32 = types
            .iter()
            .filter(|kind| kind.kind == EnemyKind::Normal)
            .map(|kind| u32::from(kind.weight))
            .sum();
        let kamikaze: u32 = types
            .iter()
            .filter(|kind| kind.kind == EnemyKind::Kamikaze)
            .map(|kind| u32::from(kind.weight))
            .sum();
        assert_eq!((normal, kamikaze), (15_300, 3_500));
        for kind in &types {
            assert_eq!(kind.settings.boundary, BoundaryMode::Wrap);
            // The cart draws an enemy with rect2(x, y, x + s, y + s), which spans
            // s + 1 pixels inclusively, so sizes 3..=6 occupy 4..=7 pixels.
            assert!([12.5, 15.625, 18.75, 21.875].contains(&kind.collider.half_extents.x));
        }
    }

    #[test]
    fn transient_blasts_do_not_consume_population_slots_and_zero_weights_are_disabled() {
        let mut app = app();
        let mut types = reference_archetypes();
        for kind in &mut types {
            if kind.kind == EnemyKind::Normal {
                kind.weight = 0;
            }
        }
        app.insert_resource(EnemyPopulation {
            target_count: 2,
            types,
            ..default()
        })
        .insert_resource(EnemySpawnQueue::with_seed(123))
        .add_plugins(EnemyPopulationPlugin);
        app.world_mut().spawn(EnemyTarget);
        step(&mut app, 0.0);
        let entities: Vec<_> = app
            .world_mut()
            .query_filtered::<Entity, With<Enemy>>()
            .iter(app.world())
            .collect();
        assert_eq!(entities.len(), 2);
        for entity in entities {
            assert_eq!(
                *app.world().get::<EnemyKind>(entity).unwrap(),
                EnemyKind::Kamikaze
            );
            app.world_mut()
                .get_mut::<Transform>(entity)
                .unwrap()
                .translation = Vec3::new(500.0, 0.0, 0.0);
        }
        step(&mut app, 0.0);
        assert_eq!(
            app.world_mut()
                .query::<&KamikazeBlast>()
                .iter(app.world())
                .count(),
            2
        );
        assert_eq!(
            app.world_mut().query::<&Enemy>().iter(app.world()).count(),
            2
        );
    }
}
