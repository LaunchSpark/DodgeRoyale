//! The burst of little squares an enemy leaves behind when it dies.
//!
//! The reference cart scatters eleven particles at random angles, each moving a
//! pixel per frame. These are plain squares that shrink as they travel, sharing
//! the shrink rate the cart uses for its trails.

use bevy::math::Vec2;
use core::f32::consts::TAU;

use crate::scale::PIXEL;

/// How many pieces a shattering enemy throws out.
pub const PIECES: usize = 11;

/// Travel speed in world units per second: one reference pixel per 60 Hz frame.
pub const SPEED: f32 = 60.0 * PIXEL;

/// The longest a piece may live.
pub const LIFE_SECONDS: f32 = 1.0;

/// The cart scales a shrinking particle by this much every 60 Hz frame.
const SHRINK_PER_FRAME: f32 = 0.9;

/// Below this fraction of its original size a piece is invisible, so it retires.
const VANISHED: f32 = 0.001;

/// A unit vector pointing a given fraction of a full turn around the circle.
#[must_use]
pub fn burst_direction(turns: f32) -> Vec2 {
    let radians = turns * TAU;
    Vec2::new(radians.cos(), radians.sin())
}

/// The size multiplier for a piece of the given age.
#[must_use]
pub fn shrink_at(age_seconds: f32) -> f32 {
    SHRINK_PER_FRAME.powf(age_seconds * 60.0)
}

/// Whether a piece is still worth drawing.
#[must_use]
pub fn is_alive(scale: f32, age_seconds: f32) -> bool {
    age_seconds <= LIFE_SECONDS && scale > VANISHED
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::math::Vec2;

    #[test]
    fn burst_matches_the_reference_count_and_speed() {
        // The cart spawns eleven pieces travelling one pixel per 60 Hz frame.
        assert_eq!(PIECES, 11);
        let per_second = 60.0 * crate::scale::PIXEL;
        assert!((SPEED - per_second).abs() < f32::EPSILON);
        assert!((LIFE_SECONDS - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn directions_are_unit_vectors_around_the_whole_circle() {
        for step in 0_u8..16 {
            let direction = burst_direction(f32::from(step) / 16.0);
            assert!(
                (direction.length() - 1.0).abs() < 0.001,
                "step {step} produced {direction:?}"
            );
        }
    }

    #[test]
    fn quarter_turns_point_at_the_four_cardinals() {
        for (turns, expected) in [
            (0.0, Vec2::X),
            (0.25, Vec2::Y),
            (0.5, Vec2::NEG_X),
            (0.75, Vec2::NEG_Y),
        ] {
            assert!(burst_direction(turns).abs_diff_eq(expected, 0.001));
        }
    }

    #[test]
    fn pieces_shrink_at_the_reference_rate() {
        // The cart multiplies a particle's radius by 0.9 every frame.
        assert!((shrink_at(0.0) - 1.0).abs() < f32::EPSILON);
        assert!((shrink_at(1.0 / 60.0) - 0.9).abs() < 0.001);
        assert!((shrink_at(2.0 / 60.0) - 0.81).abs() < 0.001);
    }

    #[test]
    fn a_piece_vanishes_well_before_its_lifetime_expires() {
        assert!(shrink_at(LIFE_SECONDS) < 0.01);
        assert!(!is_alive(1.0, LIFE_SECONDS + 0.001));
        assert!(is_alive(1.0, 0.0));
    }

    #[test]
    fn a_piece_that_has_shrunk_to_nothing_is_no_longer_alive() {
        assert!(!is_alive(0.0001, 0.1), "sub-pixel pieces should be retired");
    }
}
