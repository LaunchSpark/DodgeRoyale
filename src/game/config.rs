//! The config screen.
//!
//! The theme and enemy rows take effect. Difficulty and powerups are stored but
//! read by nothing, so each of those is labelled on screen as not implemented.

use bevy::prelude::*;

use crate::enemy_population::EnemyPopulation;
use crate::settings::Settings;

use super::art::{ActiveTheme, ink, paint};
use super::menu::{draw_marker, precise, step};
use super::screen::{MenuEntity, Screen, Transition};
use super::text::draw_text;

/// The rows, in order. The final row leaves the screen.
const ROWS: [&str; 5] = ["THEME", "DIFFICULTY", "ENEMIES", "POWERUPS", "BACK"];
/// Index of the theme row.
const THEME_ROW: usize = 0;
/// Index of the typeable enemy-count row.
const ENEMIES_ROW: usize = 2;
/// Index of the row that returns to the title screen.
const BACK_ROW: usize = 4;

const TITLE_SCALE: f32 = 20.0;
const ROW_SCALE: f32 = 12.0;
const TAG_SCALE: f32 = 5.0;
const ROW_SPACING: f32 = 88.0;
const FIRST_ROW_Y: f32 = 130.0;
const LABEL_X: f32 = -300.0;
const VALUE_X: f32 = 260.0;
const MARKER_X: f32 = -500.0;

/// The live settings. Only [`Settings::theme`] is consumed anywhere.
#[derive(Resource, Default)]
pub(super) struct ActiveSettings(pub Settings);

/// Which config row is highlighted, and whether digits are being typed into it.
#[derive(Resource, Default)]
struct ConfigCursor {
    row: usize,
    typing: bool,
}

pub(super) struct ConfigPlugin;

impl Plugin for ConfigPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ActiveSettings>()
            .init_resource::<ConfigCursor>()
            .add_systems(OnEnter(Screen::Config), enter_config)
            .add_systems(OnEnter(Screen::Playing), apply_enemy_count)
            .add_systems(OnExit(Screen::Config), despawn_config)
            .add_systems(Update, navigate.run_if(in_state(Screen::Config)));
    }
}

#[expect(
    clippy::needless_pass_by_value,
    reason = "Bevy injects system parameters by value"
)]
fn enter_config(
    mut commands: Commands,
    theme: Res<ActiveTheme>,
    settings: Res<ActiveSettings>,
    cursor: Res<ConfigCursor>,
) {
    draw_config(&mut commands, &theme, &settings.0, cursor.row);
}

/// The display name of a theme, by index.
fn theme_name(index: usize) -> &'static str {
    crate::art::THEMES
        .get(index)
        .map_or("blue", |found| found.name)
}

/// The value shown for a row, if it has one.
fn value_of(settings: &Settings, row: usize) -> Option<String> {
    match row {
        THEME_ROW => Some(theme_name(settings.theme).to_owned()),
        1 => Some(settings.difficulty.label().to_owned()),
        ENEMIES_ROW => Some(settings.starting_enemies.to_string()),
        3 => Some(if settings.powerups { "ON" } else { "OFF" }.to_owned()),
        _ => None,
    }
}

/// Carry the chosen count into the population controller when play begins.
#[expect(
    clippy::needless_pass_by_value,
    reason = "Bevy injects system parameters by value"
)]
fn apply_enemy_count(settings: Res<ActiveSettings>, mut population: ResMut<EnemyPopulation>) {
    population.target_count = settings.0.starting_enemies;
}

fn draw_config(commands: &mut Commands, theme: &ActiveTheme, settings: &Settings, selected: usize) {
    let shadow = theme.shadow();
    let faded = ink().with_alpha(0.5);
    let title = draw_text(
        commands,
        "CONFIG",
        Vec2::new(0.0, 260.0),
        TITLE_SCALE,
        ink(),
        shadow,
        60.0,
    );
    commands.entity(title).insert(MenuEntity);

    for (index, label) in ROWS.iter().enumerate() {
        let y = precise(index).mul_add(-ROW_SPACING, FIRST_ROW_Y);
        let name = draw_text(
            commands,
            label,
            Vec2::new(LABEL_X, y),
            ROW_SCALE,
            ink(),
            shadow,
            60.0,
        );
        commands.entity(name).insert(MenuEntity);

        if let Some(value) = value_of(settings, index) {
            let shown = draw_text(
                commands,
                &value,
                Vec2::new(VALUE_X, y),
                ROW_SCALE,
                ink(),
                shadow,
                60.0,
            );
            commands.entity(shown).insert(MenuEntity);
        }

        // Difficulty and powerups are stored but unread; say so on screen.
        if index != THEME_ROW && index != ENEMIES_ROW && index != BACK_ROW {
            let tag = draw_text(
                commands,
                "NOT IMPLEMENTED",
                Vec2::new(LABEL_X, y - 34.0),
                TAG_SCALE,
                faded,
                paint(theme.0.shadow).with_alpha(0.5),
                59.0,
            );
            commands.entity(tag).insert(MenuEntity);
        }

        if index == selected {
            let marker = draw_marker(
                commands,
                Vec2::new(MARKER_X, y),
                ROW_SCALE * 2.0,
                ink(),
                shadow,
            );
            commands.entity(marker).insert(MenuEntity);
        }
    }
}

fn despawn_config(mut commands: Commands, entities: Query<Entity, With<MenuEntity>>) {
    for entity in &entities {
        commands.entity(entity).despawn();
    }
}

/// Move between rows, change a row's value, or leave.
#[expect(
    clippy::needless_pass_by_value,
    reason = "Bevy injects system parameters by value"
)]
#[expect(
    clippy::too_many_arguments,
    reason = "Bevy injects each system parameter separately; grouping hides them"
)]
fn navigate(
    mut commands: Commands,
    keys: Res<ButtonInput<KeyCode>>,
    mut theme: ResMut<ActiveTheme>,
    mut clear: ResMut<ClearColor>,
    mut settings: ResMut<ActiveSettings>,
    mut cursor: ResMut<ConfigCursor>,
    mut transition: ResMut<Transition>,
    entities: Query<Entity, With<MenuEntity>>,
) {
    if transition.is_running() {
        return;
    }
    let down = keys.any_just_pressed([KeyCode::ArrowDown, KeyCode::KeyS]);
    let up = keys.any_just_pressed([KeyCode::ArrowUp, KeyCode::KeyW]);
    let right = keys.any_just_pressed([KeyCode::ArrowRight, KeyCode::KeyD]);
    let left = keys.any_just_pressed([KeyCode::ArrowLeft, KeyCode::KeyA]);

    if keys.any_just_pressed([KeyCode::Enter, KeyCode::Space]) && cursor.row == BACK_ROW {
        transition.start(Screen::Menu);
        return;
    }
    if keys.just_pressed(KeyCode::Escape) {
        transition.start(Screen::Menu);
        return;
    }

    // Digits type straight into the enemy count. The first keystroke after
    // selecting the row replaces the value; later ones extend it.
    if cursor.row == ENEMIES_ROW && !(down || up || left || right) {
        let mut edited = false;
        for key in keys.get_just_pressed() {
            if let Some(digit) = digit_of(*key) {
                if !cursor.typing {
                    settings.0.set_enemies(0);
                    cursor.typing = true;
                }
                settings.0.push_enemy_digit(digit);
                edited = true;
            }
        }
        if keys.just_pressed(KeyCode::Backspace) {
            settings.0.pop_enemy_digit();
            cursor.typing = true;
            edited = true;
        }
        if edited {
            redraw(&mut commands, &entities, &theme, &settings.0, cursor.row);
            return;
        }
    }

    if down || up {
        cursor.row = step(cursor.row, ROWS.len(), down);
        cursor.typing = false;
    } else if right || left {
        change_row(&mut settings.0, cursor.row, right);
        cursor.typing = false;
        if cursor.row == THEME_ROW
            && let Some(found) = crate::art::THEMES.get(settings.0.theme)
        {
            theme.0 = *found;
            clear.0 = theme.background();
        }
    } else {
        return;
    }

    redraw(&mut commands, &entities, &theme, &settings.0, cursor.row);
}

/// Replace the drawn screen with a freshly rendered one.
fn redraw(
    commands: &mut Commands,
    entities: &Query<Entity, With<MenuEntity>>,
    theme: &ActiveTheme,
    settings: &Settings,
    selected: usize,
) {
    for entity in entities {
        commands.entity(entity).despawn();
    }
    draw_config(commands, theme, settings, selected);
}

/// The digit a key stands for, if it is one.
const fn digit_of(key: KeyCode) -> Option<u32> {
    Some(match key {
        KeyCode::Digit0 | KeyCode::Numpad0 => 0,
        KeyCode::Digit1 | KeyCode::Numpad1 => 1,
        KeyCode::Digit2 | KeyCode::Numpad2 => 2,
        KeyCode::Digit3 | KeyCode::Numpad3 => 3,
        KeyCode::Digit4 | KeyCode::Numpad4 => 4,
        KeyCode::Digit5 | KeyCode::Numpad5 => 5,
        KeyCode::Digit6 | KeyCode::Numpad6 => 6,
        KeyCode::Digit7 | KeyCode::Numpad7 => 7,
        KeyCode::Digit8 | KeyCode::Numpad8 => 8,
        KeyCode::Digit9 | KeyCode::Numpad9 => 9,
        _ => return None,
    })
}

/// Apply a left or right press to one row.
const fn change_row(settings: &mut Settings, row: usize, forward: bool) {
    match row {
        THEME_ROW => settings.cycle_theme(forward),
        1 => {
            settings.difficulty = if forward {
                settings.difficulty.next()
            } else {
                settings.difficulty.previous()
            };
        }
        ENEMIES_ROW => settings.bump_enemies(forward),
        3 => settings.powerups = !settings.powerups,
        _ => {}
    }
}
