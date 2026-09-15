//! The fixed mapping between the reference cart's pixels and world units.
//!
//! Simulation and presentation both need this: shrapnel speed and enemy sizes
//! are expressed in cart pixels, and so are shadows and glyphs. It lives apart
//! from the palette so headless builds need no rendering code.

use bevy::math::Vec2;

/// Half the width and height of the arena, in world units.
pub const WORLD_HALF_EXTENTS: Vec2 = Vec2::new(3_000.0, 2_000.0);

/// Spacing of the arena's minor grid lines.
///
/// Chosen so it divides the world on both axes: the world wraps, so a spacing
/// that does not tile exactly would show a gap of the wrong size at the seam.
pub const GRID_SPACING: f32 = 125.0;

/// Every this many minor lines is drawn as a major one.
///
/// Four keeps the major lines tiling too, at 500 units: twelve across, eight down.
pub const GRID_MAJOR_EVERY: i16 = 4;

/// Spacing of the small registration marks on the floor.
pub const MARK_SPACING: f32 = 500.0;

/// Minor grid lines across the world.
pub const GRID_COLUMNS: i16 = 48;
/// Minor grid lines down the world.
pub const GRID_ROWS: i16 = 32;
/// Registration marks across the world.
pub const MARK_COLUMNS: i16 = 12;
/// Registration marks down the world.
pub const MARK_ROWS: i16 = 8;

/// Height of the reference cart's screen, in its own pixels.
pub const REFERENCE_SCREEN: f32 = 128.0;

/// World units the camera shows across, at its widest.
pub const VIEW_WIDTH: f32 = 1_600.0;

/// World units the camera shows down the screen.
pub const VIEW_HEIGHT: f32 = 800.0;

/// One reference pixel expressed in world units.
pub const PIXEL: f32 = VIEW_HEIGHT / REFERENCE_SCREEN;

#[cfg(test)]
mod tests {
    use super::*;

    /// Every repeating floor feature must fit the world a whole number of times,
    /// or the seam shows a gap of the wrong size when the world wraps.
    #[test]
    fn floor_markings_tile_the_world_exactly() {
        let size = WORLD_HALF_EXTENTS * 2.0;
        for spacing in [GRID_SPACING, MARK_SPACING] {
            for extent in [size.x, size.y] {
                let count = extent / spacing;
                assert!(
                    (count - count.round()).abs() < 0.001,
                    "{extent} does not divide by {spacing}"
                );
            }
        }
    }

    #[test]
    fn the_line_counts_match_their_spacing() {
        let size = WORLD_HALF_EXTENTS * 2.0;
        for (count, spacing, extent) in [
            (GRID_COLUMNS, GRID_SPACING, size.x),
            (GRID_ROWS, GRID_SPACING, size.y),
            (MARK_COLUMNS, MARK_SPACING, size.x),
            (MARK_ROWS, MARK_SPACING, size.y),
        ] {
            let span = f32::from(count) * spacing;
            assert!(
                (span - extent).abs() < 0.001,
                "{count} at {spacing} does not span {extent}"
            );
        }
    }

    #[test]
    fn major_lines_tile_the_world_exactly() {
        let size = WORLD_HALF_EXTENTS * 2.0;
        let major = GRID_SPACING * f32::from(GRID_MAJOR_EVERY);
        for extent in [size.x, size.y] {
            let count = extent / major;
            assert!(
                (count - count.round()).abs() < 0.001,
                "{extent} does not divide by major spacing {major}"
            );
        }
    }

    #[test]
    fn one_reference_pixel_scales_to_the_current_viewport() {
        // The cart is 128 px tall; the game views 800 world units.
        assert!((PIXEL - 6.25).abs() < f32::EPSILON);
    }

    #[test]
    fn the_viewport_is_wider_than_it_is_tall() {
        let width = core::hint::black_box(VIEW_WIDTH);
        let height = core::hint::black_box(VIEW_HEIGHT);
        assert!(width > height);
        assert!(core::hint::black_box(REFERENCE_SCREEN) > 0.0);
    }
}
