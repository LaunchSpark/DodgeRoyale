//! Reusable enemy behavior, independent of rendering and concrete enemy types.

use bevy::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{collision::Collider, tween};

/// The shared enemy base. Add a type-specific marker and override any defaults.
#[derive(Component, Default, Debug)]
#[require(
    EnemySettings,
    EnemyState,
    EnemyMotion,
    Velocity2d,
    Collider,
    Transform
)]
pub struct Enemy;

/// Mark an enemy for death effects, followed by removal in `EnemySet::Cleanup`.
#[derive(Component)]
pub struct Dying;

/// Behavior modifiers; the base settings remain intact for consumers and spawning.
#[derive(Component, Debug)]
pub struct EnemyMotion {
    pub speed_scale: f32,
}

impl Default for EnemyMotion {
    fn default() -> Self {
        Self { speed_scale: 1.0 }
    }
}

/// Actors eligible for automatic pursuit and enemy contact reporting.
#[derive(Component, Default, Debug)]
#[require(Transform, Velocity2d, Collider)]
pub struct EnemyTarget;

/// World-space units per second; also used by the player and camera lookahead.
#[derive(Component, Default, Debug, Clone, Copy)]
pub struct Velocity2d(pub Vec2);

/// Select built-in pursuit or supply velocity in a system before `EnemySet::Move`.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SteeringMode {
    #[default]
    Chase,
    Manual,
}

/// How an actor responds when its collider reaches an arena edge.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BoundaryMode {
    /// Leave one edge and arrive at the opposite one, keeping speed.
    #[default]
    Wrap,
    /// Stop dead at the edge.
    Clamp,
    /// Reflect off the edge.
    Bounce,
}

/// Per-enemy tuning, editable at runtime or deserializable from a type definition.
/// Distances are world units, times are seconds. Invalid settings freeze movement.
#[derive(Component, Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(default)]
pub struct EnemySettings {
    pub steering: SteeringMode,
    pub boundary: BoundaryMode,
    /// Maximum chase speed (manual velocity is also capped to this speed).
    pub speed: f32,
    /// Exponential velocity response per second; must be positive.
    pub response: f32,
    /// Acquire the nearest eligible actor inside this radius.
    pub detection_range: f32,
    /// Retain the current target to this radius; must be >= detection range.
    pub retention_range: f32,
    /// Stop pursuing inside this center-to-center distance.
    pub stop_distance: f32,
    /// Slow down over this additional distance approaching the stop distance.
    pub arrival_radius: f32,
    /// Predict target position by this many seconds of its current velocity.
    pub lookahead_seconds: f32,
}

impl Default for EnemySettings {
    fn default() -> Self {
        Self {
            steering: SteeringMode::Chase,
            boundary: BoundaryMode::Wrap,
            speed: 180.0,
            response: 8.0,
            detection_range: 900.0,
            retention_range: 1_200.0,
            stop_distance: 0.0,
            arrival_radius: 80.0,
            lookahead_seconds: 0.15,
        }
    }
}

impl EnemySettings {
    /// Validate externally supplied tuning before spawning an enemy.
    #[must_use]
    pub fn is_valid(self) -> bool {
        [
            self.speed,
            self.detection_range,
            self.retention_range,
            self.stop_distance,
            self.arrival_radius,
            self.lookahead_seconds,
        ]
        .into_iter()
        .all(|value| value.is_finite() && value >= 0.0)
            && self.response.is_finite()
            && self.response > 0.0
            && self.retention_range >= self.detection_range
    }

    /// Compute a desired velocity; actual acceleration uses our exponential tween.
    /// A velocity that backs away from nearby enemies, for when there is
    /// nothing to chase.
    ///
    /// Sums the vector to every neighbour within [`Self::detection_range`] and
    /// inverts it, so a cluster with no target drifts apart instead of sitting
    /// on top of itself. Distances are measured the short way in a wrapping
    /// world, so a neighbour just across a seam pushes back through it.
    ///
    /// `neighbours` may include this enemy's own position; it contributes
    /// nothing, so callers need not filter themselves out.
    #[must_use]
    #[expect(
        clippy::arithmetic_side_effects,
        reason = "Validated settings and finite actor coordinates"
    )]
    pub fn separation_velocity(
        self,
        position: Vec2,
        neighbours: &[Vec2],
        wrap: Option<Vec2>,
    ) -> Vec2 {
        if !self.is_valid() || !position.is_finite() {
            return Vec2::ZERO;
        }
        let mut crowd = Vec2::ZERO;
        for &other in neighbours {
            if !other.is_finite() {
                continue;
            }
            let delta = crate::torus::nearest_image(position, other, wrap) - position;
            let distance = delta.length();
            // A zero-length delta is this enemy itself, and would not normalize.
            if distance > f32::EPSILON && distance <= self.detection_range {
                crowd += delta;
            }
        }
        -crowd.normalize_or_zero() * self.speed
    }

    #[must_use]
    #[expect(
        clippy::arithmetic_side_effects,
        reason = "Validated settings and finite actor coordinates"
    )]
    pub fn desired_velocity(self, position: Vec2, target: Vec2, target_velocity: Vec2) -> Vec2 {
        if !self.is_valid()
            || !position.is_finite()
            || !target.is_finite()
            || !target_velocity.is_finite()
        {
            return Vec2::ZERO;
        }
        let actual_distance = position.distance(target);
        if actual_distance <= self.stop_distance {
            return Vec2::ZERO;
        }
        let direction =
            (target + target_velocity * self.lookahead_seconds - position).normalize_or_zero();
        let remaining = (actual_distance - self.stop_distance).max(0.0);
        let arrival = if self.arrival_radius > 0.0 {
            (remaining / self.arrival_radius).min(1.0)
        } else {
            1.0
        };
        direction * self.speed * arrival
    }
}

/// Current target, retained until it disappears or leaves the retention range.
/// Consumers can assign an eligible target here to override automatic selection.
#[derive(Component, Default, Debug)]
pub struct EnemyState {
    pub target: Option<Entity>,
    /// Where to drift when there is nothing to chase, away from the crowd.
    pub separation: Vec2,
}

/// Simulation-wide controls. The game pauses the public system sets in menus.
#[derive(Resource, Debug)]
pub struct EnemyWorld {
    /// Centered arena bounds; `None` allows an unbounded world.
    pub half_extents: Option<Vec2>,
    /// Cap time after a backgrounded browser tab; must be finite and positive.
    pub max_delta_seconds: f32,
}

impl Default for EnemyWorld {
    fn default() -> Self {
        Self {
            half_extents: Some(Vec2::new(3_000.0, 2_000.0)),
            max_delta_seconds: 0.05,
        }
    }
}

/// Extension points: acquire/steer, integrate movement, then report contacts.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EnemySet {
    Prepare,
    Steer,
    Modifiers,
    Move,
    Deaths,
    Contacts,
    Effects,
    Cleanup,
    Replenish,
}

/// Emitted once per overlapping enemy/target pair per update, after movement.
/// Consumers decide damage, cooldowns, knockback or other responses.
#[derive(Message, Debug, Clone, Copy)]
pub struct EnemyContact {
    pub enemy: Entity,
    pub target: Entity,
}

/// Install shared enemy behavior without a renderer or a particular enemy type.
pub struct EnemyPlugin;

impl Plugin for EnemyPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<EnemyWorld>()
            .add_message::<EnemyContact>()
            .configure_sets(
                Update,
                (
                    EnemySet::Prepare,
                    EnemySet::Steer,
                    EnemySet::Modifiers,
                    EnemySet::Move,
                    EnemySet::Deaths,
                    EnemySet::Contacts,
                    EnemySet::Effects,
                    EnemySet::Cleanup,
                    EnemySet::Replenish,
                )
                    .chain(),
            )
            .add_systems(
                Update,
                (
                    steer_enemies.in_set(EnemySet::Steer),
                    move_enemies.in_set(EnemySet::Move),
                    kill_colliding_enemies.in_set(EnemySet::Deaths),
                    report_contacts.in_set(EnemySet::Contacts),
                    cleanup_enemies.in_set(EnemySet::Cleanup),
                ),
            );
    }
}

type Targets<'w, 's> = Query<
    'w,
    's,
    (Entity, &'static Transform, &'static Velocity2d),
    (With<EnemyTarget>, Without<Enemy>),
>;
type LivingEnemies = (With<Enemy>, Without<Dying>);

type Movers<'w, 's> = Query<
    'w,
    's,
    (
        &'static mut Transform,
        &'static mut Velocity2d,
        &'static EnemySettings,
        &'static EnemyState,
        &'static Collider,
        &'static EnemyMotion,
    ),
    LivingEnemies,
>;

#[expect(
    clippy::needless_pass_by_value,
    reason = "Bevy injects system parameters by value"
)]
fn steer_enemies(
    world: Res<EnemyWorld>,
    targets: Targets,
    mut enemies: Query<(&Transform, &EnemySettings, &mut EnemyState), LivingEnemies>,
) {
    // Gather every enemy first: separation needs to see the whole crowd, and the
    // borrow must end before the same query is walked mutably below.
    let crowd: Vec<Vec2> = enemies
        .iter()
        .map(|(transform, _, _)| transform.translation.truncate())
        .collect();
    for (transform, settings, mut state) in &mut enemies {
        if settings.steering == SteeringMode::Manual {
            continue;
        }
        let position = transform.translation.truncate();
        if !settings.is_valid() {
            state.target = None;
            state.separation = Vec2::ZERO;
            continue;
        }
        state.separation = settings.separation_velocity(position, &crowd, world.half_extents);
        if state
            .target
            .and_then(|entity| targets.get(entity).ok())
            .is_some_and(|(_, target, _)| {
                crate::torus::distance_in(
                    position,
                    target.translation.truncate(),
                    world.half_extents,
                ) <= settings.retention_range
            })
        {
            continue;
        }
        state.target = targets
            .iter()
            .map(|(entity, target, _)| {
                (
                    entity,
                    crate::torus::distance_in(
                        position,
                        target.translation.truncate(),
                        world.half_extents,
                    ),
                )
            })
            .filter(|(_, distance)| *distance <= settings.detection_range)
            .min_by(|(a, da), (b, db)| da.total_cmp(db).then_with(|| a.cmp(b)))
            .map(|(entity, _)| entity);
    }
}

#[expect(
    clippy::needless_pass_by_value,
    reason = "Bevy injects system parameters by value"
)]
#[expect(
    clippy::arithmetic_side_effects,
    reason = "Validated settings, bounded time and world-space motion"
)]
fn move_enemies(time: Res<Time>, world: Res<EnemyWorld>, targets: Targets, mut enemies: Movers) {
    if !world.max_delta_seconds.is_finite() || world.max_delta_seconds <= 0.0 {
        return;
    }
    let delta = time.delta_secs().clamp(0.0, world.max_delta_seconds);
    for (mut transform, mut velocity, settings, state, collider, motion) in &mut enemies {
        let position = transform.translation.truncate();
        if !settings.is_valid()
            || !collider.is_valid()
            || !position.is_finite()
            || !motion.speed_scale.is_finite()
        {
            velocity.0 = Vec2::ZERO;
            continue;
        }
        if !velocity.0.is_finite() {
            velocity.0 = Vec2::ZERO;
        }
        velocity.0 = velocity.0.clamp_length_max(settings.speed);
        let displacement = if settings.steering == SteeringMode::Manual {
            velocity.0 * delta
        } else {
            let desired = state
                .target
                .and_then(|entity| targets.get(entity).ok())
                .map_or_else(
                    // Nothing to chase, so spread out rather than sit still.
                    || state.separation,
                    |(_, target, motion)| {
                        let seen = crate::torus::nearest_image(
                            position,
                            target.translation.truncate(),
                            world.half_extents,
                        );
                        settings.desired_velocity(position, seen, motion.0)
                    },
                );
            let desired = desired * motion.speed_scale.clamp(-1.0, 1.0);
            let previous = velocity.0;
            velocity.0 = tween::exponential(&previous, &desired, settings.response, delta);
            // Integrate the velocity curve, matching player motion across frame rates.
            desired * delta + (previous - velocity.0) / settings.response
        };
        let mut next = position + displacement;
        if let Some(half_extents) = world.half_extents {
            if !half_extents.is_finite() || !half_extents.cmpge(collider.half_extents).all() {
                velocity.0 = Vec2::ZERO;
                continue;
            }
            if settings.boundary == BoundaryMode::Wrap {
                // There is no wall to stop against, so speed carries through.
                next = crate::torus::wrap_position(next, half_extents);
            } else {
                let bounds = half_extents - collider.half_extents;
                let lower = -bounds - collider.offset;
                let upper = bounds - collider.offset;
                next = next.clamp(lower, upper);
                if (next.x <= lower.x && velocity.0.x < 0.0)
                    || (next.x >= upper.x && velocity.0.x > 0.0)
                {
                    velocity.0.x = if settings.boundary == BoundaryMode::Bounce {
                        -velocity.0.x
                    } else {
                        0.0
                    };
                }
                if (next.y <= lower.y && velocity.0.y < 0.0)
                    || (next.y >= upper.y && velocity.0.y > 0.0)
                {
                    velocity.0.y = if settings.boundary == BoundaryMode::Bounce {
                        -velocity.0.y
                    } else {
                        0.0
                    };
                }
            }
        }
        transform.translation.x = next.x;
        transform.translation.y = next.y;
    }
}

type ContactTargets<'w, 's> = Query<
    'w,
    's,
    (Entity, &'static Transform, &'static Collider),
    (With<EnemyTarget>, Without<Enemy>),
>;

/// Collect every victim before despawning: all members of a simultaneous
/// collision cluster die, and each entity (including its children) is removed once.
#[expect(
    clippy::needless_pass_by_value,
    reason = "Bevy injects system parameters by value"
)]
fn kill_colliding_enemies(
    mut commands: Commands,
    world: Res<EnemyWorld>,
    enemies: Query<(Entity, &Transform, &Collider), LivingEnemies>,
) {
    let mut victims = std::collections::BTreeSet::new();
    for [(a, a_transform, a_collider), (b, b_transform, b_collider)] in enemies.iter_combinations()
    {
        if a_collider.overlaps_wrapped(
            a_transform.translation.truncate(),
            *b_collider,
            b_transform.translation.truncate(),
            world.half_extents,
        ) {
            victims.insert(a);
            victims.insert(b);
        }
    }
    for entity in victims {
        commands.entity(entity).insert(Dying);
    }
}

fn cleanup_enemies(mut commands: Commands, enemies: Query<Entity, (With<Enemy>, With<Dying>)>) {
    for entity in &enemies {
        commands.entity(entity).despawn();
    }
}

#[expect(
    clippy::needless_pass_by_value,
    reason = "Bevy injects system parameters by value"
)]
fn report_contacts(
    world: Res<EnemyWorld>,
    enemies: Query<(Entity, &Transform, &Collider), LivingEnemies>,
    targets: ContactTargets,
    mut contacts: MessageWriter<EnemyContact>,
) {
    for (enemy, transform, collider) in &enemies {
        for (target, target_transform, target_collider) in &targets {
            if collider.overlaps_wrapped(
                transform.translation.truncate(),
                *target_collider,
                target_transform.translation.truncate(),
                world.half_extents,
            ) {
                contacts.write(EnemyContact { enemy, target });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn app() -> App {
        let mut app = App::new();
        app.insert_resource(Time::<()>::default())
            .add_plugins(EnemyPlugin);
        app
    }

    #[test]
    fn simultaneous_collision_cluster_dies_once_including_children() {
        let mut app = app();
        let first = app
            .world_mut()
            .spawn((Enemy, Transform::from_xyz(-20.0, 0.0, 0.0)))
            .id();
        let middle = app.world_mut().spawn(Enemy).id();
        let last = app
            .world_mut()
            .spawn((Enemy, Transform::from_xyz(20.0, 0.0, 0.0)))
            .id();
        let child = app.world_mut().spawn(Transform::default()).id();
        app.world_mut().entity_mut(middle).add_child(child);
        let survivor = app
            .world_mut()
            .spawn((Enemy, Transform::from_xyz(100.0, 0.0, 0.0)))
            .id();
        app.update();
        for victim in [first, middle, last, child] {
            assert!(app.world().get_entity(victim).is_err());
        }
        assert!(app.world().get_entity(survivor).is_ok());
    }

    #[test]
    fn disabled_enemy_colliders_do_not_kill_and_dead_enemies_do_not_report_player_hits() {
        let mut app = app();
        let first = app.world_mut().spawn(Enemy).id();
        let second = app
            .world_mut()
            .spawn((
                Enemy,
                Collider {
                    enabled: false,
                    ..default()
                },
            ))
            .id();
        app.update();
        assert!(app.world().get_entity(first).is_ok());
        assert!(app.world().get_entity(second).is_ok());
        app.world_mut().get_mut::<Collider>(second).unwrap().enabled = true;
        app.world_mut().spawn(EnemyTarget);
        let mut cursor = app
            .world()
            .resource::<Messages<EnemyContact>>()
            .get_cursor();
        app.update();
        assert!(app.world().get_entity(first).is_err());
        assert!(app.world().get_entity(second).is_err());
        assert_eq!(
            cursor
                .read(app.world().resource::<Messages<EnemyContact>>())
                .count(),
            0
        );
    }

    fn step(app: &mut App, seconds: f32) {
        app.world_mut()
            .resource_mut::<Time>()
            .advance_by(Duration::from_secs_f32(seconds));
        app.update();
    }

    fn target(app: &mut App, position: Vec2) -> Entity {
        app.world_mut()
            .spawn((
                EnemyTarget,
                Transform::from_translation(position.extend(0.0)),
            ))
            .id()
    }

    #[test]
    fn base_inserts_dependencies_and_tracks_nearest_target() {
        let mut app = app();
        let far = target(&mut app, Vec2::new(700.0, 0.0));
        let near = target(&mut app, Vec2::new(300.0, 0.0));
        let enemy = app.world_mut().spawn(Enemy).id();
        step(&mut app, 0.05);
        assert_eq!(
            app.world().get::<EnemyState>(enemy).unwrap().target,
            Some(near)
        );
        assert!(app.world().get::<Transform>(enemy).unwrap().translation.x > 0.0);
        assert!(app.world().get::<Collider>(enemy).is_some());
        // A closer newcomer does not cause target switching while the old target is retained.
        app.world_mut()
            .get_mut::<Transform>(far)
            .unwrap()
            .translation
            .x = 30.0;
        step(&mut app, 0.05);
        assert_eq!(
            app.world().get::<EnemyState>(enemy).unwrap().target,
            Some(near)
        );
        // Retain beyond the acquisition radius, even with another target nearby.
        app.world_mut()
            .get_mut::<Transform>(near)
            .unwrap()
            .translation
            .x = 1_000.0;
        step(&mut app, 0.05);
        assert_eq!(
            app.world().get::<EnemyState>(enemy).unwrap().target,
            Some(near)
        );
        app.world_mut().despawn(near);
        step(&mut app, 0.05);
        assert_eq!(
            app.world().get::<EnemyState>(enemy).unwrap().target,
            Some(far)
        );
        app.world_mut()
            .get_mut::<Transform>(far)
            .unwrap()
            .translation
            .x = 2_000.0;
        step(&mut app, 0.05);
        assert_eq!(app.world().get::<EnemyState>(enemy).unwrap().target, None);
    }

    #[test]
    fn lookahead_arrival_and_stop_distance_change_steering() {
        let settings = EnemySettings {
            speed: 200.0,
            stop_distance: 20.0,
            arrival_radius: 100.0,
            lookahead_seconds: 0.5,
            ..default()
        };
        let desired =
            settings.desired_velocity(Vec2::ZERO, Vec2::new(70.0, 0.0), Vec2::new(0.0, 100.0));
        assert!(desired.y > 0.0);
        assert!((desired.length() - 100.0).abs() < 0.001);
        assert_eq!(
            settings.desired_velocity(Vec2::ZERO, Vec2::new(10.0, 0.0), Vec2::X),
            Vec2::ZERO
        );
    }

    #[test]
    fn lost_target_smoothly_brakes_without_despawning_enemy() {
        let mut app = app();
        let prey = target(&mut app, Vec2::new(500.0, 0.0));
        let enemy = app.world_mut().spawn(Enemy).id();
        step(&mut app, 0.05);
        let previous = app.world().get::<Velocity2d>(enemy).unwrap().0.length();
        app.world_mut().despawn(prey);
        step(&mut app, 0.05);
        let now = app.world().get::<Velocity2d>(enemy).unwrap().0.length();
        assert!(now > 0.0 && now < previous);
    }

    #[test]
    fn contacts_are_reported_even_without_a_tracking_target_and_can_be_disabled() {
        let mut app = app();
        let prey = target(&mut app, Vec2::ZERO);
        let enemy = app
            .world_mut()
            .spawn((
                Enemy,
                EnemySettings {
                    steering: SteeringMode::Manual,
                    ..default()
                },
                Transform::from_xyz(-30.0, 0.0, 0.0),
                Velocity2d(Vec2::new(180.0, 0.0)),
            ))
            .id();
        let mut cursor = app
            .world()
            .resource::<Messages<EnemyContact>>()
            .get_cursor();
        step(&mut app, 0.05);
        let contacts: Vec<_> = cursor
            .read(app.world().resource::<Messages<EnemyContact>>())
            .copied()
            .collect();
        assert_eq!(contacts.len(), 1);
        assert_eq!(contacts[0].enemy, enemy);
        assert_eq!(contacts[0].target, prey);
        app.world_mut().get_mut::<Collider>(enemy).unwrap().enabled = false;
        step(&mut app, 0.05);
        assert_eq!(
            cursor
                .read(app.world().resource::<Messages<EnemyContact>>())
                .count(),
            0
        );
    }

    /// Settings with a known speed and reach, for the separation cases.
    fn spreader() -> EnemySettings {
        EnemySettings {
            speed: 100.0,
            detection_range: 500.0,
            ..default()
        }
    }

    #[test]
    fn an_enemy_with_no_neighbours_does_not_spread() {
        assert_eq!(
            spreader().separation_velocity(Vec2::ZERO, &[], None),
            Vec2::ZERO
        );
    }

    #[test]
    fn a_crowded_enemy_moves_directly_away_from_the_crowd() {
        let away = spreader().separation_velocity(Vec2::ZERO, &[Vec2::new(50.0, 0.0)], None);
        assert!(away.x < 0.0, "expected a push to the left, got {away:?}");
        assert!(
            (away.length() - 100.0).abs() < 0.001,
            "speed should be used in full"
        );
    }

    #[test]
    fn neighbours_on_opposite_sides_cancel_each_other() {
        let balanced = spreader().separation_velocity(
            Vec2::ZERO,
            &[Vec2::new(50.0, 0.0), Vec2::new(-50.0, 0.0)],
            None,
        );
        assert!(balanced.length() < 0.001, "a balanced crowd has no way out");
    }

    #[test]
    fn neighbours_beyond_the_detection_range_are_ignored() {
        let far = spreader().separation_velocity(Vec2::ZERO, &[Vec2::new(501.0, 0.0)], None);
        assert_eq!(far, Vec2::ZERO);
    }

    #[test]
    fn an_enemy_does_not_flee_from_itself() {
        // Callers pass every enemy, including the one being steered.
        let alone = spreader().separation_velocity(Vec2::ZERO, &[Vec2::ZERO], None);
        assert!(
            alone.is_finite(),
            "a zero-length delta must not produce NaN"
        );
        assert_eq!(alone, Vec2::ZERO);
    }

    #[test]
    fn spreading_takes_the_short_route_across_a_seam() {
        let world = Vec2::new(300.0, 100.0);
        // The neighbour is fifteen units away through the seam, to the right.
        let away = spreader().separation_velocity(
            Vec2::new(290.0, 0.0),
            &[Vec2::new(-295.0, 0.0)],
            Some(world),
        );
        assert!(
            away.x < 0.0,
            "should back away from the seam, not toward it: {away:?}"
        );
    }

    #[test]
    fn manual_steering_obeys_speed_bounds_offset_and_preserves_z() {
        let mut app = app();
        app.world_mut().resource_mut::<EnemyWorld>().half_extents = Some(Vec2::splat(100.0));
        let enemy = app
            .world_mut()
            .spawn((
                Enemy,
                EnemySettings {
                    steering: SteeringMode::Manual,
                    speed: 200.0,
                    // This case is about clamp geometry, which is no longer the
                    // default now that the world wraps.
                    boundary: BoundaryMode::Clamp,
                    ..default()
                },
                Collider {
                    half_extents: Vec2::splat(10.0),
                    offset: Vec2::new(5.0, 0.0),
                    enabled: true,
                },
                Transform::from_xyz(84.0, 0.0, 7.0),
                Velocity2d(Vec2::new(1_000.0, 0.0)),
            ))
            .id();
        step(&mut app, 0.05);
        assert_eq!(
            app.world().get::<Transform>(enemy).unwrap().translation,
            Vec3::new(85.0, 0.0, 7.0)
        );
        assert_eq!(app.world().get::<Velocity2d>(enemy).unwrap().0, Vec2::ZERO);
    }

    #[test]
    fn stationary_target_motion_matches_across_frame_rates_and_caps_tab_time() {
        fn simulate(frames: u16, seconds: f32) -> Vec3 {
            let mut app = app();
            target(&mut app, Vec2::new(800.0, 0.0));
            let enemy = app.world_mut().spawn(Enemy).id();
            for _ in 0..frames {
                step(&mut app, seconds);
            }
            app.world().get::<Transform>(enemy).unwrap().translation
        }
        assert!(simulate(30, 1.0 / 30.0).abs_diff_eq(simulate(144, 1.0 / 144.0), 0.01));
        assert!(simulate(1, 60.0).abs_diff_eq(simulate(1, 0.05), 0.001));
    }

    #[test]
    fn invalid_configuration_freezes_motion_and_settings_round_trip() {
        let mut app = app();
        target(&mut app, Vec2::new(500.0, 0.0));
        let settings = EnemySettings {
            speed: -5.0,
            ..default()
        };
        assert!(!settings.is_valid());
        let enemy = app
            .world_mut()
            .spawn((Enemy, settings, Velocity2d(Vec2::X)))
            .id();
        step(&mut app, 0.05);
        assert_eq!(
            app.world().get::<Transform>(enemy).unwrap().translation,
            Vec3::ZERO
        );
        assert_eq!(app.world().get::<Velocity2d>(enemy).unwrap().0, Vec2::ZERO);
        let json = serde_json::to_string(&EnemySettings::default()).unwrap();
        assert!(
            serde_json::from_str::<EnemySettings>(&json)
                .unwrap()
                .is_valid()
        );
        let partial: EnemySettings =
            serde_json::from_str(r#"{"speed":240.0,"steering":"Manual"}"#).unwrap();
        assert!(partial.is_valid());
        assert_eq!(partial.steering, SteeringMode::Manual);
    }
}
