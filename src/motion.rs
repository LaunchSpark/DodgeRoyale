//! Renderer-independent player movement, shared by browser and native builds.

use bevy::math::{StableInterpolate, Vec2};

use crate::torus::wrap_position;

pub use crate::scale::WORLD_HALF_EXTENTS;
/// Half the player's collision hitbox width and height; visual art is independent.
pub const PLAYER_HALF_SIZE: f32 = 12.0;
/// Top speed in world units per second.
///
/// The reference cart runs a fixed 60 Hz loop of `v *= 0.8; v += 0.5`, so its
/// velocity settles at `0.5 / (1 - 0.8)` = 2.5 pixels per frame, or 150 per
/// second. One reference pixel is 6.25 world units.
const PLAYER_SPEED: f32 = 937.5;
/// How sharply velocity approaches its target, per second.
///
/// The cart keeps 0.8 of its velocity each 60 Hz frame; the equivalent
/// continuous decay is `-ln(0.8) * 60`, which reproduces that ramp exactly.
const MOVEMENT_RESPONSE: f32 = 13.388_61;
pub const MAX_FRAME_SECONDS: f32 = 0.05;

/// Top speed in world units per second.
///
/// Published so observation code can bound how far a predicted path reaches
/// without restating the number.
#[must_use]
pub const fn top_speed() -> f32 {
    PLAYER_SPEED
}

/// How sharply velocity approaches its target, per second.
///
/// Published for the same reason as [`top_speed`]: a second implementation of
/// these rules has to agree on the number, and one that restated it would be
/// free to drift from this one without anything noticing.
#[must_use]
pub const fn movement_response() -> f32 {
    MOVEMENT_RESPONSE
}

/// Advance a player's position and velocity using smooth acceleration and braking.
///
/// Input is normalized and elapsed time is capped at 50 ms. The returned
/// position is wrapped into the arena, which has no walls: travelling past an
/// edge continues from the opposite one at the same speed. Call with finite
/// positions, velocities and time.
#[expect(
    clippy::arithmetic_side_effects,
    reason = "Normalized input, capped time and fixed speed keep motion arithmetic bounded"
)]
pub fn advance_motion(position: Vec2, velocity: &mut Vec2, direction: Vec2, seconds: f32) -> Vec2 {
    // Returning to a backgrounded browser tab must not teleport the player.
    let delta = seconds.clamp(0.0, MAX_FRAME_SECONDS);
    let target = direction.normalize_or_zero() * PLAYER_SPEED;
    let previous_velocity = *velocity;
    velocity.smooth_nudge(&target, MOVEMENT_RESPONSE, delta);
    // Integrate the exponential velocity curve so acceleration and travel agree
    // across refresh rates, instead of applying end-of-frame velocity throughout.
    let displacement = target * delta + (previous_velocity - *velocity) / MOVEMENT_RESPONSE;
    // The world is a torus: leaving one edge arrives at the opposite one. There
    // is no wall to stop against, so velocity carries through a seam untouched.
    wrap_position(position + displacement, WORLD_HALF_EXTENTS)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagonal_input_has_the_same_speed_as_cardinal_input() {
        let mut cardinal = Vec2::ZERO;
        let mut diagonal = Vec2::ZERO;
        advance_motion(Vec2::ZERO, &mut cardinal, Vec2::X, 0.05);
        advance_motion(Vec2::ZERO, &mut diagonal, Vec2::ONE, 0.05);
        assert!((cardinal.length() - diagonal.length()).abs() < 0.001);
    }

    #[test]
    fn acceleration_and_travel_are_independent_of_frame_partitioning() {
        let mut fast_position = Vec2::ZERO;
        let mut fast_velocity = Vec2::ZERO;
        for _ in 0..120 {
            fast_position = advance_motion(fast_position, &mut fast_velocity, Vec2::X, 1.0 / 120.0);
        }
        let mut slow_position = Vec2::ZERO;
        let mut slow_velocity = Vec2::ZERO;
        for _ in 0..30 {
            slow_position = advance_motion(slow_position, &mut slow_velocity, Vec2::X, 1.0 / 30.0);
        }
        assert!(fast_position.distance(slow_position) < 0.01);
        assert!(fast_velocity.distance(slow_velocity) < 0.01);
    }

    /// The cart runs a fixed 60 Hz loop of `v *= 0.8; v += 0.5; pos += v`, so
    /// velocity settles at `0.5 / (1 - 0.8)` = 2.5 reference pixels per frame,
    /// which is 150 per second. One reference pixel is 6.25 world units.
    const REFERENCE_TERMINAL: f32 = 150.0 * 6.25;

    #[test]
    fn terminal_speed_matches_the_reference_cart() {
        let mut velocity = Vec2::ZERO;
        for _ in 0..600 {
            advance_motion(Vec2::ZERO, &mut velocity, Vec2::X, 1.0 / 60.0);
        }
        assert!(
            (velocity.x - REFERENCE_TERMINAL).abs() < 1.0,
            "settled at {}, expected {REFERENCE_TERMINAL}",
            velocity.x
        );
    }

    #[test]
    fn acceleration_ramp_matches_the_reference_cart() {
        // The cart's first three frames reach 0.5, 0.9 and 1.22 px per frame.
        let mut velocity = Vec2::ZERO;
        for expected in [0.5_f32, 0.9, 1.22] {
            advance_motion(Vec2::ZERO, &mut velocity, Vec2::X, 1.0 / 60.0);
            let want = expected * 60.0 * 6.25;
            assert!(
                (velocity.x - want).abs() < 1.0,
                "got {}, expected {want}",
                velocity.x
            );
        }
    }

    #[test]
    fn releasing_input_smoothly_brakes() {
        let mut velocity = Vec2::new(PLAYER_SPEED, 0.0);
        let next = advance_motion(Vec2::ZERO, &mut velocity, Vec2::ZERO, 0.05);
        assert!(next.x > 0.0);
        assert!(velocity.x > 0.0 && velocity.x < PLAYER_SPEED);
        for _ in 0..20 {
            advance_motion(Vec2::ZERO, &mut velocity, Vec2::ZERO, 0.05);
        }
        assert!(velocity.length() < 0.001);
    }

    #[test]
    fn leaving_an_edge_arrives_at_the_opposite_one() {
        for (start, direction, expected_sign) in [
            (Vec2::new(2_999.0, 0.0), Vec2::X, Vec2::NEG_X),
            (Vec2::new(-2_999.0, 0.0), Vec2::NEG_X, Vec2::X),
            (Vec2::new(0.0, 1_999.0), Vec2::Y, Vec2::NEG_Y),
            (Vec2::new(0.0, -1_999.0), Vec2::NEG_Y, Vec2::Y),
        ] {
            let mut velocity = direction * PLAYER_SPEED;
            let next = advance_motion(start, &mut velocity, direction, 0.05);
            // The position reappears on the far side, along the travelled axis.
            assert!(
                next.dot(expected_sign) > 0.0,
                "moving {direction:?} from {start:?} gave {next:?}"
            );
            // Crossing a seam is not a wall: speed is untouched.
            assert!(
                velocity.dot(direction) > 0.0,
                "wrapping cancelled velocity: {velocity:?}"
            );
        }
    }

    #[test]
    fn crossing_a_seam_covers_the_same_ground_as_open_space() {
        // Two players a pixel apart either side of the seam travel together.
        let mut near_seam = Vec2::new(2_990.0, 0.0);
        let mut open = Vec2::new(0.0, 0.0);
        let mut seam_velocity = Vec2::ZERO;
        let mut open_velocity = Vec2::ZERO;
        for _ in 0..30 {
            near_seam = advance_motion(near_seam, &mut seam_velocity, Vec2::X, 1.0 / 60.0);
            open = advance_motion(open, &mut open_velocity, Vec2::X, 1.0 / 60.0);
        }
        let travelled =
            crate::torus::wrapped_distance(Vec2::new(2_990.0, 0.0), near_seam, WORLD_HALF_EXTENTS);
        let reference = crate::torus::wrapped_distance(Vec2::ZERO, open, WORLD_HALF_EXTENTS);
        assert!(
            (travelled - reference).abs() < 0.001,
            "seam crossing travelled {travelled}, open space {reference}"
        );
        assert!(seam_velocity.abs_diff_eq(open_velocity, 0.001));
    }

    #[test]
    fn background_tab_time_is_capped() {
        let mut velocity = Vec2::ZERO;
        let mut regular_velocity = Vec2::ZERO;
        let resumed = advance_motion(Vec2::ZERO, &mut velocity, Vec2::X, 60.0);
        let regular = advance_motion(Vec2::ZERO, &mut regular_velocity, Vec2::X, 0.05);
        assert!(resumed.distance(regular) < 0.001);
        assert!(velocity.distance(regular_velocity) < 0.001);
    }
}
