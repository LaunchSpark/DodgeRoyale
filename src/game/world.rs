//! The arena, painted in the active theme.

use bevy::{camera::visibility::VisibilitySystems, prelude::*, window::PrimaryWindow};

pub(super) use crate::motion::WORLD_HALF_EXTENTS;

use crate::camera_math::viewport_half_size;
use crate::scale::{
    GRID_COLUMNS, GRID_MAJOR_EVERY, GRID_ROWS, GRID_SPACING, MARK_COLUMNS, MARK_ROWS, MARK_SPACING,
};

use super::art::{ActiveTheme, flat_rect, paint_faded};
use super::screen::{GameEntity, Screen};

/// How strongly the minor grid, major grid and floor markings read.
const MINOR: f32 = 0.16;
const MAJOR: f32 = 0.30;
const MARKING: f32 = 0.45;

pub(super) struct WorldPlugin;

/// Which periodic copy of the floor this root represents.
#[derive(Component)]
struct WorldTile {
    x: i16,
    y: i16,
}

impl Plugin for WorldPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(OnEnter(Screen::Playing), spawn_world)
            .add_systems(OnExit(Screen::Playing), despawn_world)
            .add_systems(
                PostUpdate,
                show_visible_tiles
                    .before(VisibilitySystems::VisibilityPropagate)
                    .run_if(in_state(Screen::Playing)),
            );
    }
}

#[expect(
    clippy::needless_pass_by_value,
    reason = "Bevy injects system parameters by value"
)]
#[expect(
    clippy::arithmetic_side_effects,
    reason = "Arena geometry uses finite, bounded world coordinates and sizes"
)]
fn spawn_world(mut commands: Commands, theme: Res<ActiveTheme>) {
    let size = WORLD_HALF_EXTENTS * 2.0;
    // The world is periodic, so it is laid out as a three by three block. A
    // camera near a seam then looks into the neighbouring copy instead of a void,
    // and because the floor tiles exactly the join is invisible.
    for tile_x in -1_i16..=1 {
        for tile_y in -1_i16..=1 {
            let offset = Vec2::new(f32::from(tile_x) * size.x, f32::from(tile_y) * size.y);
            let root = commands
                .spawn((
                    GameEntity,
                    WorldTile {
                        x: tile_x,
                        y: tile_y,
                    },
                    Transform::from_translation(offset.extend(0.0)),
                    if tile_x == 0 && tile_y == 0 {
                        Visibility::Visible
                    } else {
                        Visibility::Hidden
                    },
                ))
                .id();
            flat_rect(
                &mut commands,
                root,
                Vec2::ZERO,
                size,
                theme.background(),
                -10.0,
            );
            spawn_grid(&mut commands, root, theme.0.shadow);
            spawn_landmarks(&mut commands, root, theme.0.shadow);
            spawn_origin(&mut commands, root, theme.0.shadow);
        }
    }
}

/// The centre tile is always present. A neighbouring copy is needed only when
/// the viewport crosses that side of the world's seam.
fn tile_axis_visible(tile: i16, camera: f32, world_half: f32, view_half: f32) -> bool {
    match tile {
        -1 => camera - view_half <= -world_half,
        0 => true,
        1 => camera + view_half >= world_half,
        _ => false,
    }
}

#[expect(
    clippy::needless_pass_by_value,
    reason = "Bevy injects system parameters by value"
)]
fn show_visible_tiles(
    camera: Single<&Transform, With<Camera2d>>,
    window: Single<&Window, With<PrimaryWindow>>,
    mut tiles: Query<(&WorldTile, &mut Visibility)>,
) {
    let eye = camera.translation.truncate();
    let view_half = viewport_half_size(window.width(), window.height());
    for (tile, mut visibility) in &mut tiles {
        let visible = tile_axis_visible(tile.x, eye.x, WORLD_HALF_EXTENTS.x, view_half.x)
            && tile_axis_visible(tile.y, eye.y, WORLD_HALF_EXTENTS.y, view_half.y);
        let desired = if visible {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
        if *visibility != desired {
            *visibility = desired;
        }
    }
}

#[expect(
    clippy::arithmetic_side_effects,
    reason = "Vec2 operators on finite, bounded arena constants"
)]
fn spawn_grid(commands: &mut Commands, root: Entity, shadow: u8) {
    // Lines are laid out from one corner so the pattern tiles: the line that
    // would sit on the far edge is the same line as the one on the near edge.
    let size = WORLD_HALF_EXTENTS * 2.0;
    for index in 0..GRID_COLUMNS {
        let alpha = if index % GRID_MAJOR_EVERY == 0 {
            MAJOR
        } else {
            MINOR
        };
        let x = f32::from(index).mul_add(GRID_SPACING, -WORLD_HALF_EXTENTS.x);
        flat_rect(
            commands,
            root,
            Vec2::new(x, 0.0),
            Vec2::new(1.0, size.y),
            paint_faded(shadow, alpha),
            -9.0,
        );
    }
    for index in 0..GRID_ROWS {
        let alpha = if index % GRID_MAJOR_EVERY == 0 {
            MAJOR
        } else {
            MINOR
        };
        let y = f32::from(index).mul_add(GRID_SPACING, -WORLD_HALF_EXTENTS.y);
        flat_rect(
            commands,
            root,
            Vec2::new(0.0, y),
            Vec2::new(size.x, 1.0),
            paint_faded(shadow, alpha),
            -9.0,
        );
    }
    // Small registration marks help the eye read motion even between landmarks.
    let mark = paint_faded(shadow, MARKING);
    for column in 0..MARK_COLUMNS {
        for row in 0..MARK_ROWS {
            let center = Vec2::new(
                f32::from(column).mul_add(MARK_SPACING, -WORLD_HALF_EXTENTS.x),
                f32::from(row).mul_add(MARK_SPACING, -WORLD_HALF_EXTENTS.y),
            );
            flat_rect(commands, root, center, Vec2::new(14.0, 2.0), mark, -8.0);
            flat_rect(commands, root, center, Vec2::new(2.0, 14.0), mark, -8.0);
        }
    }
}

#[expect(
    clippy::arithmetic_side_effects,
    reason = "Deterministic floor markings stay within the fixed arena bounds"
)]
fn spawn_landmarks(commands: &mut Commands, root: Entity, shadow: u8) {
    // These are painted floor markings, not collision obstacles.
    let mark = paint_faded(shadow, MARKING);
    let faint = paint_faded(shadow, MAJOR);
    for (center, size) in [
        (Vec2::new(-960.0, 600.0), Vec2::new(320.0, 160.0)),
        (Vec2::new(960.0, -600.0), Vec2::new(160.0, 320.0)),
        (Vec2::new(1_680.0, 1_200.0), Vec2::new(320.0, 320.0)),
        (Vec2::new(-1_680.0, -1_200.0), Vec2::new(400.0, 200.0)),
        (Vec2::new(-2_400.0, 960.0), Vec2::new(160.0, 320.0)),
        (Vec2::new(2_400.0, -960.0), Vec2::new(320.0, 160.0)),
    ] {
        flat_frame(commands, root, center, size, 3.0, mark);
        flat_frame(commands, root, center, size - Vec2::splat(24.0), 1.0, faint);
        for offset in [-24.0, 0.0, 24.0] {
            flat_rect(
                commands,
                root,
                center + Vec2::new(offset, 0.0),
                Vec2::new(8.0, 48.0),
                faint,
                -6.0,
            );
        }
    }

    // Short, widely spaced lane dashes lead away from the spawn point.
    for direction in [Vec2::X, Vec2::NEG_X, Vec2::Y, Vec2::NEG_Y] {
        for distance in [240.0, 320.0, 400.0, 480.0] {
            flat_rect(
                commands,
                root,
                direction * distance,
                if direction.x.abs() > 0.0 {
                    Vec2::new(28.0, 3.0)
                } else {
                    Vec2::new(3.0, 28.0)
                },
                mark,
                -7.0,
            );
        }
    }
}

fn spawn_origin(commands: &mut Commands, root: Entity, shadow: u8) {
    let mark = paint_faded(shadow, MARKING);
    flat_frame(commands, root, Vec2::ZERO, Vec2::splat(144.0), 1.0, mark);
    for x in [-1.0, 1.0] {
        for y in [-1.0, 1.0] {
            flat_rect(
                commands,
                root,
                Vec2::new(x * 88.0, y * 104.0),
                Vec2::new(32.0, 3.0),
                mark,
                -5.0,
            );
            flat_rect(
                commands,
                root,
                Vec2::new(x * 104.0, y * 88.0),
                Vec2::new(3.0, 32.0),
                mark,
                -5.0,
            );
        }
    }
    flat_rect(commands, root, Vec2::ZERO, Vec2::new(32.0, 2.0), mark, -5.0);
    flat_rect(commands, root, Vec2::ZERO, Vec2::new(2.0, 32.0), mark, -5.0);
}

/// A four-sided outline drawn flat on the floor.
#[expect(
    clippy::arithmetic_side_effects,
    reason = "Frame geometry is only called with finite positive arena dimensions"
)]
fn flat_frame(
    commands: &mut Commands,
    root: Entity,
    center: Vec2,
    size: Vec2,
    width: f32,
    color: Color,
) {
    for sign in [-1.0, 1.0] {
        flat_rect(
            commands,
            root,
            center + Vec2::new(sign * size.x * 0.5, 0.0),
            Vec2::new(width, size.y),
            color,
            -6.0,
        );
        flat_rect(
            commands,
            root,
            center + Vec2::new(0.0, sign * size.y * 0.5),
            Vec2::new(size.x, width),
            color,
            -6.0,
        );
    }
}

/// Remove the arena when gameplay ends.
fn despawn_world(mut commands: Commands, entities: Query<Entity, With<GameEntity>>) {
    for entity in &entities {
        commands.entity(entity).despawn();
    }
}

#[cfg(test)]
mod tests {
    use super::tile_axis_visible;

    #[test]
    fn adjacent_floor_copies_appear_only_at_a_visible_seam() {
        let half_world = 3_000.0;
        let half_view = 800.0;
        for tile in [-1, 1] {
            assert!(!tile_axis_visible(tile, 0.0, half_world, half_view));
        }
        assert!(tile_axis_visible(0, 0.0, half_world, half_view));
        assert!(tile_axis_visible(-1, -2_200.0, half_world, half_view));
        assert!(tile_axis_visible(1, 2_200.0, half_world, half_view));
        assert!(!tile_axis_visible(1, -2_200.0, half_world, half_view));
        assert!(!tile_axis_visible(-1, 2_200.0, half_world, half_view));
    }
}
