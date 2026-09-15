//! Draws strings with the original bitmap font, shadow included.

use bevy::prelude::*;

use crate::glyphs::{GLYPH_HEIGHT, GLYPH_WIDTH, glyph, text_width};

use super::art::shadowed_rect;

/// Blank columns between adjacent glyphs, matching [`text_width`].
const GLYPH_GAP: usize = 1;

/// Draw `text` centred on `center`, one font pixel measuring `scale` units.
///
/// Every pixel is parented to a single returned entity, so a caller can remove
/// the whole string by despawning it.
#[expect(
    clippy::arithmetic_side_effects,
    reason = "Layout uses bounded glyph counts and finite scales"
)]
pub(super) fn draw_text(
    commands: &mut Commands,
    text: &str,
    center: Vec2,
    scale: f32,
    color: Color,
    shadow: Color,
    z: f32,
) -> Entity {
    let width = precise(text_width(text)) * scale;
    let height = precise(GLYPH_HEIGHT) * scale;
    // Offset by half a pixel so each square sits on the grid, not on its corner.
    let origin = center - Vec2::new(width, height) * 0.5 + Vec2::splat(scale * 0.5);
    let root = commands
        .spawn((Transform::default(), Visibility::default()))
        .id();

    let mut pen = 0.0_f32;
    for character in text.chars() {
        if let Some(rows) = glyph(character) {
            for (index, row) in rows.iter().enumerate() {
                for column in 0..GLYPH_WIDTH {
                    // Bit 2 is the leftmost pixel of the row.
                    if row >> (GLYPH_WIDTH - 1 - column) & 1 != 1 {
                        continue;
                    }
                    // Rows run top to bottom; world Y runs bottom to top.
                    let flipped = GLYPH_HEIGHT - 1 - index;
                    let offset = Vec2::new(
                        precise(column).mul_add(scale, pen),
                        precise(flipped) * scale,
                    );
                    shadowed_rect(
                        commands,
                        root,
                        origin + offset,
                        Vec2::splat(scale),
                        color,
                        shadow,
                        z,
                    );
                }
            }
        }
        pen += precise(GLYPH_WIDTH + GLYPH_GAP) * scale;
    }
    root
}

/// Convert a small layout count to a float without a lossy cast.
fn precise(value: usize) -> f32 {
    u16::try_from(value).map_or(0.0, f32::from)
}
