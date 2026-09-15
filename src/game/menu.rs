//! The title screen and its selectable items.

use bevy::prelude::*;

use super::art::{ActiveTheme, ink, shadowed_rect};
use super::screen::{MenuEntity, Screen, Transition};
use super::text::draw_text;

/// The items on the title screen, in order.
const ITEMS: [&str; 2] = ["START GAME", "CONFIG"];

const TITLE_SCALE: f32 = 26.0;
const ITEM_SCALE: f32 = 14.0;
const ITEM_SPACING: f32 = 96.0;
const FIRST_ITEM_Y: f32 = -40.0;
const MARKER_X: f32 = -420.0;

/// Which title-screen item is highlighted.
#[derive(Resource, Default)]
pub(super) struct MenuCursor(pub usize);

pub(super) struct MenuPlugin;

impl Plugin for MenuPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<MenuCursor>()
            .add_systems(OnEnter(Screen::Menu), enter_menu)
            .add_systems(OnExit(Screen::Menu), despawn_menu)
            .add_systems(Update, navigate.run_if(in_state(Screen::Menu)));
    }
}

#[expect(
    clippy::needless_pass_by_value,
    reason = "Bevy injects system parameters by value"
)]
fn enter_menu(mut commands: Commands, theme: Res<ActiveTheme>, cursor: Res<MenuCursor>) {
    draw_menu(&mut commands, &theme, cursor.0);
}

/// Draw the whole screen. Cheap enough to redraw whenever the cursor moves.
fn draw_menu(commands: &mut Commands, theme: &ActiveTheme, selected: usize) {
    let shadow = theme.shadow();
    let title = draw_text(
        commands,
        "DODGE ROYALE",
        Vec2::new(0.0, 150.0),
        TITLE_SCALE,
        ink(),
        shadow,
        60.0,
    );
    commands.entity(title).insert(MenuEntity);

    for (index, label) in ITEMS.iter().enumerate() {
        let y = precise(index).mul_add(-ITEM_SPACING, FIRST_ITEM_Y);
        let row = draw_text(
            commands,
            label,
            Vec2::new(0.0, y),
            ITEM_SCALE,
            ink(),
            shadow,
            60.0,
        );
        commands.entity(row).insert(MenuEntity);
        if index == selected {
            let marker = draw_marker(
                commands,
                Vec2::new(MARKER_X, y),
                ITEM_SCALE * 2.0,
                ink(),
                shadow,
            );
            commands.entity(marker).insert(MenuEntity);
        }
    }
}

fn despawn_menu(mut commands: Commands, entities: Query<Entity, With<MenuEntity>>) {
    for entity in &entities {
        commands.entity(entity).despawn();
    }
}

/// Move the highlight, and act on the highlighted item.
#[expect(
    clippy::needless_pass_by_value,
    reason = "Bevy injects system parameters by value"
)]
fn navigate(
    mut commands: Commands,
    keys: Res<ButtonInput<KeyCode>>,
    theme: Res<ActiveTheme>,
    mut cursor: ResMut<MenuCursor>,
    mut transition: ResMut<Transition>,
    entities: Query<Entity, With<MenuEntity>>,
) {
    if transition.is_running() {
        return;
    }
    let down = keys.any_just_pressed([KeyCode::ArrowDown, KeyCode::KeyS]);
    let up = keys.any_just_pressed([KeyCode::ArrowUp, KeyCode::KeyW]);
    if down || up {
        cursor.0 = step(cursor.0, ITEMS.len(), down);
        for entity in &entities {
            commands.entity(entity).despawn();
        }
        draw_menu(&mut commands, &theme, cursor.0);
        return;
    }
    if keys.any_just_pressed([KeyCode::Enter, KeyCode::Space]) {
        match cursor.0 {
            0 => transition.start(Screen::Playing),
            _ => transition.start(Screen::Config),
        }
    }
}

/// Draw the highlight square, returning a root that owns it.
pub(super) fn draw_marker(
    commands: &mut Commands,
    center: Vec2,
    size: f32,
    color: Color,
    shadow: Color,
) -> Entity {
    let root = commands
        .spawn((Transform::default(), Visibility::default()))
        .id();
    shadowed_rect(
        commands,
        root,
        center,
        Vec2::splat(size),
        color,
        shadow,
        60.0,
    );
    root
}

/// Move one place through `len` items, wrapping at both ends.
pub(super) const fn step(current: usize, len: usize, forward: bool) -> usize {
    if len == 0 {
        return 0;
    }
    if forward {
        let next = current.saturating_add(1);
        if next >= len { 0 } else { next }
    } else if current == 0 {
        len.saturating_sub(1)
    } else {
        current.saturating_sub(1)
    }
}

/// Convert a small row index to a float without a lossy cast.
pub(super) fn precise(value: usize) -> f32 {
    u16::try_from(value).map_or(0.0, f32::from)
}
