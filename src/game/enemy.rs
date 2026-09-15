//! Gameplay lifecycle and filled/outlined visuals for the hostile reference types.

use super::{
    art::{ActiveTheme, ink},
    ghost::Ghosted,
    player::{Player, move_player},
    screen::{GameEntity, Screen, Transition},
};
use crate::{
    art::{PIXEL, shadow_translation},
    collision::Collider,
    enemy::{EnemyPlugin, EnemySet, EnemyWorld},
    enemy_population::{EnemyPopulationPlugin, EnemySpawnQueue, EnemyType},
    enemy_types::{EnemyKind, KamikazeBlast, PlayerHit, ReferenceEnemyPlugin},
    motion::WORLD_HALF_EXTENTS,
};
use bevy::prelude::*;

pub(super) struct GameEnemyPlugin;

impl Plugin for GameEnemyPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((EnemyPlugin, EnemyPopulationPlugin, ReferenceEnemyPlugin))
            .insert_resource(EnemyWorld {
                half_extents: Some(WORLD_HALF_EXTENTS),
                ..default()
            })
            .configure_sets(
                Update,
                (
                    EnemySet::Prepare.after(move_player),
                    EnemySet::Steer,
                    EnemySet::Modifiers,
                    EnemySet::Move,
                    EnemySet::Deaths,
                    EnemySet::Contacts,
                    EnemySet::Effects,
                    EnemySet::Cleanup,
                    EnemySet::Replenish,
                )
                    .run_if(in_state(Screen::Playing)),
            )
            .add_systems(
                Update,
                (decorate_enemies, decorate_blasts, sync_outlines)
                    .chain()
                    .after(EnemySet::Replenish)
                    .run_if(in_state(Screen::Playing)),
            )
            .add_systems(
                Update,
                return_after_defeat
                    .after(EnemySet::Effects)
                    .run_if(in_state(Screen::Playing)),
            )
            .add_systems(OnExit(Screen::Playing), clear_spawn_queue);
    }
}

fn clear_spawn_queue(mut queue: ResMut<EnemySpawnQueue>) {
    queue.clear();
}

fn return_after_defeat(
    mut hits: MessageReader<PlayerHit>,
    players: Query<(), With<Player>>,
    mut transition: ResMut<Transition>,
) {
    for hit in hits.read() {
        if players.contains(hit.target) {
            transition.start(Screen::Menu);
        }
    }
}

#[derive(Clone, Copy)]
enum Edge {
    Top,
    Bottom,
    Left,
    Right,
}

#[derive(Component)]
struct OutlinePiece {
    edge: Edge,
    shadow: bool,
}

fn outline(commands: &mut Commands, entity: Entity, shadow: Color) {
    commands
        .entity(entity)
        .insert((GameEntity, Ghosted, Visibility::default()))
        .with_children(|parent| {
            for edge in [Edge::Top, Edge::Bottom, Edge::Left, Edge::Right] {
                for is_shadow in [true, false] {
                    parent.spawn((
                        OutlinePiece {
                            edge,
                            shadow: is_shadow,
                        },
                        Sprite::from_color(if is_shadow { shadow } else { ink() }, Vec2::ZERO),
                        Transform::default(),
                    ));
                }
            }
        });
}

#[expect(
    clippy::needless_pass_by_value,
    reason = "Bevy injects system parameters by value"
)]
#[expect(
    clippy::arithmetic_side_effects,
    reason = "Spawn placement validates finite collider dimensions before decoration"
)]
fn decorate_enemies(
    mut commands: Commands,
    theme: Res<ActiveTheme>,
    enemies: Query<(Entity, &Collider, &EnemyKind), Added<EnemyType>>,
) {
    for (entity, collider, kind) in &enemies {
        match kind {
            EnemyKind::Normal => {
                let size = collider.half_extents * 2.0;
                commands
                    .entity(entity)
                    .insert((GameEntity, Ghosted, Sprite::from_color(ink(), size)))
                    .with_children(|parent| {
                        parent.spawn((
                            Name::new("Enemy shadow"),
                            Sprite::from_color(theme.shadow(), size),
                            Transform::from_translation(shadow_translation()),
                        ));
                    });
            }
            EnemyKind::Kamikaze => outline(&mut commands, entity, theme.shadow()),
        }
    }
}

#[expect(
    clippy::needless_pass_by_value,
    reason = "Bevy injects system parameters by value"
)]
fn decorate_blasts(
    mut commands: Commands,
    theme: Res<ActiveTheme>,
    blasts: Query<Entity, Added<KamikazeBlast>>,
) {
    for entity in &blasts {
        outline(&mut commands, entity, theme.shadow());
    }
}

#[expect(
    clippy::arithmetic_side_effects,
    reason = "Outline geometry uses finite collider extents and a clamped pixel border"
)]
fn sync_outlines(
    parents: Query<&Collider>,
    mut pieces: Query<(&ChildOf, &OutlinePiece, &mut Sprite, &mut Transform)>,
) {
    for (parent, piece, mut sprite, mut transform) in &mut pieces {
        let Ok(collider) = parents.get(parent.parent()) else {
            continue;
        };
        let size = collider.half_extents * 2.0;
        let width = PIXEL.min(size.min_element());
        let (offset, dimensions) = match piece.edge {
            Edge::Top => (
                Vec2::new(0.0, (size.y - width) * 0.5),
                Vec2::new(size.x, width),
            ),
            Edge::Bottom => (
                Vec2::new(0.0, -(size.y - width) * 0.5),
                Vec2::new(size.x, width),
            ),
            Edge::Left => (
                Vec2::new(-(size.x - width) * 0.5, 0.0),
                Vec2::new(width, size.y),
            ),
            Edge::Right => (
                Vec2::new((size.x - width) * 0.5, 0.0),
                Vec2::new(width, size.y),
            ),
        };
        sprite.custom_size = Some(dimensions);
        transform.translation = offset.extend(0.0)
            + if piece.shadow {
                shadow_translation()
            } else {
                Vec3::ZERO
            };
    }
}
