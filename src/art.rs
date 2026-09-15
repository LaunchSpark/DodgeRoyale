//! The reference cart's palette, themes and shadow convention.
//!
//! The cart draws every object twice: once one pixel down in the current
//! theme's shadow colour, then again on top in white. Only the background and
//! shadow change between themes; the foreground is always [`INK`].

use bevy::math::{Vec2, Vec3};

pub use crate::scale::PIXEL;

/// Every drop shadow falls straight down by exactly one reference pixel.
pub const SHADOW_OFFSET: Vec2 = Vec2::new(0.0, -PIXEL);

/// The palette index the foreground is drawn in, in every theme.
pub const INK: u8 = 7;

/// An opaque colour taken from the reference palette.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Rgb {
    pub red: u8,
    pub green: u8,
    pub blue: u8,
}

impl Rgb {
    /// Build a colour from its three 8-bit channels.
    pub const fn new(red: u8, green: u8, blue: u8) -> Self {
        Self { red, green, blue }
    }
}

/// The reference palette, indexed exactly as the cart indexes it.
pub const PICO8: [Rgb; 16] = [
    Rgb::new(0, 0, 0),
    Rgb::new(29, 43, 83),
    Rgb::new(126, 37, 83),
    Rgb::new(0, 135, 81),
    Rgb::new(171, 82, 54),
    Rgb::new(95, 87, 79),
    Rgb::new(194, 195, 199),
    Rgb::new(255, 241, 232),
    Rgb::new(255, 0, 77),
    Rgb::new(255, 163, 0),
    Rgb::new(255, 236, 39),
    Rgb::new(0, 228, 54),
    Rgb::new(41, 173, 255),
    Rgb::new(131, 118, 156),
    Rgb::new(255, 119, 168),
    Rgb::new(255, 204, 170),
];

/// Look up a palette colour, falling back to [`INK`] rather than panicking.
pub fn color(index: u8) -> Rgb {
    PICO8
        .get(usize::from(index))
        .copied()
        .unwrap_or(Rgb::new(255, 241, 232))
}

/// A selectable colour scheme: only the background and shadow ever change.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Theme {
    pub name: &'static str,
    pub background: u8,
    pub shadow: u8,
}

impl Theme {
    /// Pair a background palette index with the shadow index drawn beneath it.
    const fn new(name: &'static str, background: u8, shadow: u8) -> Self {
        Self {
            name,
            background,
            shadow,
        }
    }
}

/// The thirteen themes offered by the reference cart, in its own order.
///
/// Only [`DEFAULT`] is painted today; the rest wait on a settings screen.
pub const THEMES: [Theme; 13] = [
    Theme::new("blue", 12, 1),
    Theme::new("dark blue", 1, 0),
    Theme::new("green", 3, 1),
    Theme::new("indigo", 13, 1),
    Theme::new("purple", 2, 1),
    Theme::new("orange", 9, 8),
    Theme::new("pink", 14, 2),
    Theme::new("grey", 6, 5),
    Theme::new("dark grey", 5, 0),
    Theme::new("black", 0, 0),
    Theme::new("neon red", 0, 8),
    Theme::new("neon blue", 0, 12),
    Theme::new("neon green", 0, 11),
];

/// The theme the cart starts in.
pub const DEFAULT: Theme = Theme::new("blue", 12, 1);

/// How far behind its body a shadow sits, so the body always wins the overlap.
const SHADOW_DEPTH: f32 = 0.1;

/// Where a shadow sits relative to the body that casts it.
pub const fn shadow_translation() -> Vec3 {
    Vec3::new(SHADOW_OFFSET.x, SHADOW_OFFSET.y, -SHADOW_DEPTH)
}

/// Find a theme by its display name.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "Extracted reference data; a theme selector has not been built yet"
    )
)]
pub fn theme(name: &str) -> Option<&'static Theme> {
    THEMES.iter().find(|candidate| candidate.name == name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn palette_holds_the_sixteen_reference_colors() {
        assert_eq!(PICO8.len(), 16);
        assert_eq!(color(0), Rgb::new(0, 0, 0));
        assert_eq!(color(7), Rgb::new(255, 241, 232));
        assert_eq!(color(12), Rgb::new(41, 173, 255));
    }

    #[test]
    fn out_of_range_indices_fall_back_to_ink_instead_of_panicking() {
        assert_eq!(color(16), color(INK));
        assert_eq!(color(u8::MAX), color(INK));
    }

    #[test]
    fn every_theme_pairs_a_background_with_a_shadow() {
        assert_eq!(THEMES.len(), 13);
        for entry in &THEMES {
            assert!(entry.background < 16, "{} background", entry.name);
            assert!(entry.shadow < 16, "{} shadow", entry.name);
            assert!(!entry.name.is_empty());
        }
    }

    #[test]
    fn theme_table_matches_the_reference_cart() {
        let found = theme("blue").expect("blue theme exists");
        assert_eq!((found.background, found.shadow), (12, 1));
        let neon = theme("neon green").expect("neon green theme exists");
        assert_eq!((neon.background, neon.shadow), (0, 11));
        assert!(theme("chartreuse").is_none());
    }

    #[test]
    fn the_palette_holds_exactly_as_many_themes_as_settings_expects() {
        assert_eq!(THEMES.len(), crate::settings::THEME_COUNT);
    }

    #[test]
    fn every_theme_name_is_renderable_by_the_font() {
        for theme in &THEMES {
            for character in theme.name.chars() {
                assert!(
                    crate::glyphs::glyph(character).is_some(),
                    "{:?} contains an unrenderable {character:?}",
                    theme.name
                );
            }
        }
    }

    #[test]
    fn default_theme_is_the_reference_default() {
        assert_eq!(DEFAULT.name, "blue");
        assert_eq!((DEFAULT.background, DEFAULT.shadow), (12, 1));
    }

    #[test]
    fn shadows_sit_behind_the_body_that_casts_them() {
        let offset = shadow_translation();
        assert!(offset.x.abs() < f32::EPSILON);
        assert!((offset.y + PIXEL).abs() < f32::EPSILON);
        assert!(offset.z < 0.0, "shadow must not draw over its body");
    }

    #[test]
    fn shadows_fall_straight_down_by_one_reference_pixel() {
        assert_eq!(SHADOW_OFFSET, Vec2::new(0.0, -PIXEL));
    }
}
