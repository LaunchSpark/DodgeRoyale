//! Select enemy types first, then place them safely to replenish the arena.

use crate::enemy_types::EnemyKind;
use crate::{
    collision::Collider,
    enemy::{BoundaryMode, Enemy, EnemySet, EnemySettings, EnemyTarget, EnemyWorld},
};
use bevy::prelude::*;
use std::collections::VecDeque;

/// A type definition copied into the queue before any position is sampled.
#[derive(Clone, Debug)]
pub struct EnemyArchetype {
    pub name: String,
    pub kind: EnemyKind,
    /// Relative spawn frequency; zero disables new selection.
    pub weight: u16,
    pub settings: EnemySettings,
    pub collider: Collider,
}

/// Identifies a spawned type for presentation and type-specific behavior systems.
#[derive(Component, Debug)]
pub struct EnemyType(pub String);

/// Population and placement controls. Type weights are relative frequencies.
#[derive(Resource, Debug)]
pub struct EnemyPopulation {
    pub target_count: usize,
    pub types: Vec<EnemyArchetype>,
    /// Maximum queued entries and placement searches processed per update.
    pub budget_per_frame: usize,
    /// Random candidates tested for each queued type per update.
    pub placement_attempts: u16,
    /// Extra distance beyond the selected type's detection range.
    pub detection_margin: f32,
}

impl Default for EnemyPopulation {
    fn default() -> Self {
        Self {
            target_count: 100,
            budget_per_frame: 64,
            placement_attempts: 64,
            detection_margin: 32.0,
            types: reference_archetypes(),
        }
    }
}

/// Hostile type ratio 76.5:17.5 and size weights 20:50:20:10 from the cart.
/// Power-up personalities are deliberately absent from this catalog.
#[must_use]
#[expect(
    clippy::arithmetic_side_effects,
    reason = "Fixed reference weights and pixel sizes fit their numeric types"
)]
pub fn reference_archetypes() -> Vec<EnemyArchetype> {
    let mut types = Vec::new();
    for (kind, name, frequency, range) in [
        (EnemyKind::Normal, "Normal", 153_u16, 900.0),
        (EnemyKind::Kamikaze, "Kamikaze", 35_u16, 1_100.0),
    ] {
        for (pixels, weight) in [(3_u16, 20_u16), (4, 50), (5, 20), (6, 10)] {
            types.push(EnemyArchetype {
                name: name.to_owned(),
                kind,
                weight: frequency * weight,
                settings: EnemySettings {
                    boundary: BoundaryMode::Wrap,
                    detection_range: range,
                    retention_range: 1_400.0,
                    ..default()
                },
                // The cart draws enemies with rect2(x, y, x + s, y + s), whose
                // span is inclusive, so a size of s occupies s + 1 pixels.
                collider: Collider::rectangle(Vec2::splat((f32::from(pixels) + 1.0) * 6.25 * 0.5)),
            });
        }
    }
    types
}

/// Pending type snapshots persist across failed placement attempts.
#[derive(Resource, Default)]
pub struct EnemySpawnQueue {
    pending: VecDeque<EnemyArchetype>,
    random: fastrand::Rng,
}

impl EnemySpawnQueue {
    /// Use a reproducible stream for tests or replayable spawning.
    #[must_use]
    pub const fn with_seed(seed: u64) -> Self {
        Self {
            pending: VecDeque::new(),
            random: fastrand::Rng::with_seed(seed),
        }
    }

    /// Inspect queued type definitions without changing their selection order.
    pub fn pending(&self) -> impl Iterator<Item = &EnemyArchetype> {
        self.pending.iter()
    }

    /// Discard requests from a finished game while preserving the random stream.
    pub fn clear(&mut self) {
        self.pending.clear();
    }
}

/// Optional population controller; install alongside `EnemyPlugin`.
pub struct EnemyPopulationPlugin;

impl Plugin for EnemyPopulationPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<EnemyPopulation>()
            .init_resource::<EnemySpawnQueue>()
            .add_systems(Update, replenish_enemies.in_set(EnemySet::Replenish));
    }
}

type Actors<'w, 's, F> = Query<'w, 's, (&'static Transform, &'static Collider), F>;

#[expect(
    clippy::needless_pass_by_value,
    reason = "Bevy injects system parameters by value"
)]
fn replenish_enemies(
    mut commands: Commands,
    population: Res<EnemyPopulation>,
    arena: Res<EnemyWorld>,
    mut queue: ResMut<EnemySpawnQueue>,
    enemies: Actors<With<Enemy>>,
    targets: Actors<(With<EnemyTarget>, Without<Enemy>)>,
) {
    let deficit = population.target_count.saturating_sub(enemies.iter().len());
    queue
        .pending
        .truncate(deficit.min(population.budget_per_frame));
    if targets.is_empty() {
        return;
    }
    // Selection is a separate phase. Failed placements retain the chosen type
    // and its exact settings; changing the catalog only affects future choices.
    while !population.types.is_empty()
        && queue.pending.len() < deficit.min(population.budget_per_frame)
    {
        let total = population.types.iter().fold(0_u64, |sum, kind| {
            sum.saturating_add(u64::from(kind.weight))
        });
        if total == 0 {
            break;
        }
        let mut ticket = queue.random.u64(..total);
        for archetype in &population.types {
            if ticket < u64::from(archetype.weight) {
                queue.pending.push_back(archetype.clone());
                break;
            }
            ticket = ticket.saturating_sub(u64::from(archetype.weight));
        }
    }
    if queue.pending.is_empty() {
        return;
    }
    let Some(bounds) = arena.half_extents else {
        return;
    };
    let targets: Vec<_> = targets
        .iter()
        .map(|(transform, collider)| (transform.translation.truncate(), *collider))
        .collect();
    let mut occupied: Vec<_> = enemies
        .iter()
        .map(|(transform, collider)| (transform.translation.truncate(), *collider))
        .collect();
    let requests = queue.pending.len();
    for _ in 0..requests {
        let Some(archetype) = queue.pending.pop_front() else {
            break;
        };
        if let Some(position) = place_enemy(
            &archetype,
            bounds,
            &targets,
            &occupied,
            &population,
            &mut queue.random,
        ) {
            occupied.push((position, archetype.collider));
            commands.spawn((
                Enemy,
                EnemyType(archetype.name.clone()),
                Name::new(archetype.name),
                archetype.kind,
                archetype.settings,
                archetype.collider,
                Transform::from_translation(position.extend(10.0)),
            ));
        } else {
            queue.pending.push_back(archetype);
        }
    }
}

#[expect(
    clippy::arithmetic_side_effects,
    reason = "Validated finite arena and collider dimensions bound sampled positions"
)]
fn place_enemy(
    archetype: &EnemyArchetype,
    bounds: Vec2,
    targets: &[(Vec2, Collider)],
    occupied: &[(Vec2, Collider)],
    population: &EnemyPopulation,
    random: &mut fastrand::Rng,
) -> Option<Vec2> {
    let collider = archetype.collider;
    if !archetype.settings.is_valid()
        || !collider.is_valid()
        || !bounds.is_finite()
        || !bounds.cmpge(collider.half_extents).all()
        || !population.detection_margin.is_finite()
        || population.detection_margin < 0.0
    {
        return None;
    }
    let lower = -bounds + collider.half_extents - collider.offset;
    let upper = bounds - collider.half_extents - collider.offset;
    let exclusion = archetype.settings.detection_range + population.detection_margin;
    for _ in 0..population.placement_attempts {
        let position = lower + (upper - lower) * Vec2::new(random.f32(), random.f32());
        // The world wraps, so a spot just across a seam from a target is close,
        // not far: measuring flat here would spawn enemies on top of the player.
        if targets.iter().all(|(target, target_collider)| {
            crate::torus::distance_in(position, *target, Some(bounds)) > exclusion
                && !collider.overlaps_wrapped(position, *target_collider, *target, Some(bounds))
        }) && occupied.iter().all(|(other, other_collider)| {
            !collider.overlaps_wrapped(position, *other_collider, *other, Some(bounds))
        }) {
            return Some(position);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::enemy::{EnemyPlugin, SteeringMode};

    fn app(count: usize) -> App {
        let mut app = App::new();
        app.insert_resource(Time::<()>::default())
            .insert_resource(EnemyPopulation {
                target_count: count,
                ..default()
            })
            .insert_resource(EnemySpawnQueue::with_seed(123))
            .add_plugins((EnemyPlugin, EnemyPopulationPlugin));
        app.world_mut().spawn(EnemyTarget);
        app
    }

    fn population(app: &mut App) -> Vec<(Entity, Vec2, EnemySettings, Collider)> {
        app.world_mut()
            .query_filtered::<(Entity, &Transform, &EnemySettings, &Collider), With<Enemy>>()
            .iter(app.world())
            .map(|(entity, transform, settings, collider)| {
                (
                    entity,
                    transform.translation.truncate(),
                    *settings,
                    *collider,
                )
            })
            .collect()
    }

    #[test]
    fn fills_exact_population_using_each_selected_types_range_and_reserves_placements() {
        let mut app = app(24);
        app.update();
        let enemies = population(&mut app);
        assert_eq!(enemies.len(), 24);
        assert!(
            enemies
                .iter()
                .any(|(_, _, settings, _)| settings.detection_range > 1_000.0)
        );
        assert!(
            enemies
                .iter()
                .any(|(_, _, settings, _)| settings.detection_range < 1_000.0)
        );
        for (entity, position, settings, collider) in &enemies {
            assert!(position.length() > settings.detection_range + 32.0);
            assert!(
                (position.abs() + collider.half_extents)
                    .cmple(Vec2::new(3_000.0, 2_000.0))
                    .all()
            );
            for (other, other_position, _, other_collider) in &enemies {
                if entity != other {
                    assert!(!collider.overlaps(*position, *other_collider, *other_position));
                }
            }
        }
        app.update();
        assert_eq!(population(&mut app).len(), 24);
        assert_eq!(
            app.world().resource::<EnemySpawnQueue>().pending().count(),
            0
        );
    }

    #[test]
    fn collision_deaths_are_replaced_in_the_same_update() {
        let mut app = app(4);
        app.update();
        let before = population(&mut app);
        let first = before[0].0;
        let second = before[1].0;
        for entity in [first, second] {
            app.world_mut()
                .get_mut::<Transform>(entity)
                .unwrap()
                .translation = Vec3::ZERO;
            app.world_mut()
                .get_mut::<EnemySettings>(entity)
                .unwrap()
                .steering = SteeringMode::Manual;
        }
        app.update();
        assert!(app.world().get_entity(first).is_err());
        assert!(app.world().get_entity(second).is_err());
        assert_eq!(population(&mut app).len(), 4);
        assert!(
            population(&mut app)
                .iter()
                .all(|(_, position, settings, _)| position.length() > settings.detection_range)
        );
    }

    #[test]
    fn failed_placement_keeps_the_selected_type_and_retries_when_space_opens() {
        let mut app = app(1);
        let blocked = EnemyArchetype {
            name: "Queued giant range".to_owned(),
            kind: EnemyKind::Normal,
            weight: 1,
            settings: EnemySettings {
                detection_range: 10_000.0,
                retention_range: 10_000.0,
                ..default()
            },
            collider: Collider::default(),
        };
        app.world_mut().resource_mut::<EnemyPopulation>().types = vec![blocked];
        app.update();
        assert!(population(&mut app).is_empty());
        // A catalog change must not reroll the already selected type.
        app.world_mut().resource_mut::<EnemyPopulation>().types = EnemyPopulation::default().types;
        app.update();
        assert_eq!(
            app.world()
                .resource::<EnemySpawnQueue>()
                .pending()
                .next()
                .unwrap()
                .name,
            "Queued giant range"
        );
        assert!(population(&mut app).is_empty());
        app.world_mut()
            .resource_mut::<EnemyPopulation>()
            .types
            .clear();
        app.world_mut().resource_mut::<EnemyWorld>().half_extents = Some(Vec2::splat(20_000.0));
        app.update();
        let enemies = population(&mut app);
        assert_eq!(enemies.len(), 1);
        assert_eq!(
            app.world().get::<EnemyType>(enemies[0].0).unwrap().0,
            "Queued giant range"
        );
        assert!(enemies[0].1.length() > 10_032.0);
    }

    #[test]
    fn no_player_means_no_spawns_and_budget_limits_work() {
        let mut app = app(8);
        let target = app
            .world_mut()
            .query_filtered::<Entity, With<EnemyTarget>>()
            .single(app.world())
            .unwrap();
        app.world_mut().despawn(target);
        app.update();
        assert!(population(&mut app).is_empty());
        app.world_mut().spawn(EnemyTarget);
        app.world_mut()
            .resource_mut::<EnemyPopulation>()
            .budget_per_frame = 2;
        app.update();
        assert_eq!(population(&mut app).len(), 2);
        for _ in 0..3 {
            app.update();
        }
        assert_eq!(population(&mut app).len(), 8);
    }

    #[test]
    fn placement_respects_all_targets_collider_offsets_and_impossible_geometry() {
        let archetype = EnemyArchetype {
            name: "Offset".to_owned(),
            kind: EnemyKind::Normal,
            weight: 1,
            settings: EnemySettings {
                detection_range: 50.0,
                ..default()
            },
            collider: Collider {
                offset: Vec2::new(80.0, -60.0),
                ..default()
            },
        };
        let bounds = Vec2::splat(300.0);
        let targets = [
            (Vec2::ZERO, Collider::default()),
            (Vec2::new(200.0, 200.0), Collider::default()),
        ];
        let mut random = fastrand::Rng::with_seed(10);
        let position = place_enemy(
            &archetype,
            bounds,
            &targets,
            &[],
            &EnemyPopulation::default(),
            &mut random,
        )
        .unwrap();
        assert!(
            ((position + archetype.collider.offset).abs() + archetype.collider.half_extents)
                .cmple(bounds)
                .all()
        );
        for (target, _) in targets {
            assert!(position.distance(target) > 82.0);
        }
        assert!(
            place_enemy(
                &archetype,
                Vec2::ONE,
                &targets,
                &[],
                &EnemyPopulation::default(),
                &mut random
            )
            .is_none()
        );
    }
}
