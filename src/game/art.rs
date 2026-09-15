//! Bridges the shared palette into Bevy rendering types.

use bevy::prelude::*;

use crate::art::{self, Theme};

/// The theme every system paints with.
#[derive(Resource, Clone, Copy)]
pub(super) struct ActiveTheme(pub Theme);

impl Default for ActiveTheme {
    fn default() -> Self {
        Self(art::DEFAULT)
    }
}

impl ActiveTheme {
    /// The colour behind everything.
    pub(super) fn background(self) -> Color {
        paint(self.0.background)
    }

    /// The colour every drop shadow is drawn in.
    pub(super) fn shadow(self) -> Color {
        paint(self.0.shadow)
    }
}

/// Convert a palette index into a renderable colour.
pub(super) fn paint(index: u8) -> Color {
    let rgb = art::color(index);
    Color::srgb_u8(rgb.red, rgb.green, rgb.blue)
}

/// A palette colour softened so floor detail sits behind the action.
pub(super) fn paint_faded(index: u8, alpha: f32) -> Color {
    paint(index).with_alpha(alpha)
}

/// The foreground colour, identical in every theme.
pub(super) fn ink() -> Color {
    paint(art::INK)
}

/// Spawn a rectangle drawn the way the reference draws everything: a shadow one
/// pixel below, then the body on top.
pub(super) fn shadowed_rect(
    commands: &mut Commands,
    parent: Entity,
    center: Vec2,
    size: Vec2,
    color: Color,
    shadow: Color,
    z: f32,
) -> Entity {
    let id = commands
        .spawn((
            Sprite::from_color(color, size),
            Transform::from_translation(center.extend(z)),
            children![(
                Sprite::from_color(shadow, size),
                Transform::from_translation(art::shadow_translation()),
            )],
        ))
        .id();
    commands.entity(parent).add_child(id);
    id
}

/// Spawn a flat rectangle with no shadow, for floor markings.
pub(super) fn flat_rect(
    commands: &mut Commands,
    parent: Entity,
    center: Vec2,
    size: Vec2,
    color: Color,
    z: f32,
) {
    let id = commands
        .spawn((
            Sprite::from_color(color, size),
            Transform::from_translation(center.extend(z)),
        ))
        .id();
    commands.entity(parent).add_child(id);
}
