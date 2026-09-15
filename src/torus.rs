//! Wrapping world coordinates.
//!
//! The world is a torus: leaving one edge arrives at the opposite one. Every
//! position is kept inside `[-half, half)` on both axes, and every difference is
//! wrapped the same way, which makes it the shortest of the two routes between
//! two points: the direct one, or the one that runs out to the border and back.
//!
//! Each axis picks its route independently, so a path may wrap horizontally
//! while staying direct vertically.

use bevy::math::Vec2;

/// Bring a position back inside the world, wrapping at every edge.
///
/// A world with a zero, negative or non-finite size has no seams to wrap at, so
/// the position is returned untouched rather than producing infinities.
#[must_use]
#[expect(
    clippy::arithmetic_side_effects,
    reason = "Guarded against non-finite and non-positive world sizes"
)]
pub fn wrap_position(position: Vec2, half_extents: Vec2) -> Vec2 {
    let size = half_extents * 2.0;
    if !position.is_finite() || !size.is_finite() || size.cmple(Vec2::ZERO).any() {
        return position;
    }
    (position + half_extents).rem_euclid(size) - half_extents
}

/// The shortest vector from `from` to `to`, crossing a seam when that is nearer.
///
/// Never longer than half the world on either axis.
#[must_use]
#[expect(
    clippy::arithmetic_side_effects,
    reason = "The difference is finite whenever both positions are"
)]
pub fn wrapped_delta(from: Vec2, to: Vec2, half_extents: Vec2) -> Vec2 {
    wrap_position(to - from, half_extents)
}

/// The shortest distance between two points on the torus.
#[must_use]
pub fn wrapped_distance(a: Vec2, b: Vec2, half_extents: Vec2) -> f32 {
    wrapped_delta(a, b, half_extents).length()
}

/// Distance between two points, taking a seam when the world wraps.
///
/// `wrap` carries the world's half extents on a torus, and `None` on a flat
/// plane, so callers holding an optional world size need no branch of their own.
#[must_use]
pub fn distance_in(a: Vec2, b: Vec2, wrap: Option<Vec2>) -> f32 {
    wrap.map_or_else(
        || a.distance(b),
        |half_extents| wrapped_distance(a, b, half_extents),
    )
}

/// Where `target` appears to be when seen from `from`.
///
/// On a torus every point has many images; this is the one reached by the short
/// route. Steering toward it makes pursuit agree with contact, instead of
/// chasing the long way around a world that has no far side.
#[must_use]
#[expect(
    clippy::arithmetic_side_effects,
    reason = "Offsetting by a wrapped delta stays finite"
)]
pub fn nearest_image(from: Vec2, target: Vec2, wrap: Option<Vec2>) -> Vec2 {
    wrap.map_or(target, |half_extents| {
        from + wrapped_delta(from, target, half_extents)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::math::Vec2;

    /// A deliberately lopsided world, so axis mistakes cannot hide.
    const HALF: Vec2 = Vec2::new(300.0, 100.0);

    #[test]
    fn a_point_inside_the_world_is_left_alone() {
        for point in [Vec2::ZERO, Vec2::new(299.0, -99.0), Vec2::new(-12.5, 40.0)] {
            assert!(wrap_position(point, HALF).abs_diff_eq(point, 0.001));
        }
    }

    #[test]
    fn leaving_one_edge_arrives_at_the_opposite_one() {
        // Ten units past the right edge is ten units in from the left.
        assert!(
            wrap_position(Vec2::new(310.0, 0.0), HALF).abs_diff_eq(Vec2::new(-290.0, 0.0), 0.001)
        );
        assert!(
            wrap_position(Vec2::new(0.0, -110.0), HALF).abs_diff_eq(Vec2::new(0.0, 90.0), 0.001)
        );
    }

    #[test]
    fn wrapping_is_idempotent_and_survives_many_laps() {
        let far = Vec2::new(300.0f32.mul_add(7.0, 25.0), 100.0f32.mul_add(-5.0, -10.0));
        let once = wrap_position(far, HALF);
        assert!(wrap_position(once, HALF).abs_diff_eq(once, 0.001));
        assert!(once.x.abs() <= HALF.x && once.y.abs() <= HALF.y);
    }

    #[test]
    fn a_step_past_the_edge_lands_where_the_eye_expects() {
        // Three units above a player near the top edge is three units of travel,
        // even though the coordinate itself jumps to the far side.
        let player = Vec2::new(0.0, 99.0);
        let stepped = wrap_position(player + Vec2::new(0.0, 3.0), HALF);
        assert!(stepped.abs_diff_eq(Vec2::new(0.0, -98.0), 0.001));
        // The travelled distance is still three, not the long way round.
        assert!((wrapped_distance(player, stepped, HALF) - 3.0).abs() < 0.001);
    }

    #[test]
    fn the_shortest_route_crosses_the_seam_when_that_is_closer() {
        let left = Vec2::new(-290.0, 0.0);
        let right = Vec2::new(290.0, 0.0);
        // Twenty units across the seam, not 580 the long way.
        assert!((wrapped_distance(left, right, HALF) - 20.0).abs() < 0.001);
        assert!(wrapped_delta(left, right, HALF).abs_diff_eq(Vec2::new(-20.0, 0.0), 0.001));
    }

    #[test]
    fn each_axis_chooses_its_own_route() {
        // Wraps horizontally, direct vertically.
        let from = Vec2::new(-290.0, -10.0);
        let to = Vec2::new(290.0, 40.0);
        assert!(wrapped_delta(from, to, HALF).abs_diff_eq(Vec2::new(-20.0, 50.0), 0.001));
    }

    #[test]
    fn no_delta_is_ever_longer_than_half_the_world() {
        for x in [-899.0, -301.0, -1.0, 0.0, 17.0, 299.0, 901.0] {
            for y in [-250.0, -99.0, 0.0, 61.0, 340.0] {
                let delta = wrapped_delta(Vec2::ZERO, Vec2::new(x, y), HALF);
                assert!(delta.x.abs() <= HALF.x + 0.001, "x={x} gave {delta:?}");
                assert!(delta.y.abs() <= HALF.y + 0.001, "y={y} gave {delta:?}");
            }
        }
    }

    #[test]
    fn distance_does_not_depend_on_which_end_you_start_from() {
        let a = Vec2::new(-280.0, 80.0);
        let b = Vec2::new(250.0, -90.0);
        let there = wrapped_distance(a, b, HALF);
        let back = wrapped_distance(b, a, HALF);
        assert!((there - back).abs() < 0.001);
    }

    #[test]
    fn an_unwrapped_world_measures_in_a_straight_line() {
        let a = Vec2::new(-290.0, 0.0);
        let b = Vec2::new(290.0, 0.0);
        assert!((distance_in(a, b, None) - 580.0).abs() < 0.001);
        assert!((distance_in(a, b, Some(HALF)) - 20.0).abs() < 0.001);
    }

    #[test]
    fn the_nearest_image_is_the_one_reached_by_the_short_route() {
        let from = Vec2::new(-290.0, 0.0);
        let target = Vec2::new(290.0, 0.0);
        // Seen from the left edge, the target sits just off the left side.
        let image = nearest_image(from, target, Some(HALF));
        assert!(image.abs_diff_eq(Vec2::new(-310.0, 0.0), 0.001));
        // Its distance matches the wrapped one, so steering agrees with contact.
        assert!((from.distance(image) - distance_in(from, target, Some(HALF))).abs() < 0.001);
        // Without wrapping the target is used as given.
        assert!(nearest_image(from, target, None).abs_diff_eq(target, 0.001));
    }

    #[test]
    fn a_degenerate_world_returns_the_position_untouched() {
        // Zero or negative extents would otherwise divide by zero.
        for half in [Vec2::ZERO, Vec2::new(-5.0, 10.0), Vec2::new(f32::NAN, 1.0)] {
            let point = Vec2::new(7.0, -3.0);
            assert!(wrap_position(point, half).abs_diff_eq(point, 0.001));
        }
    }
}
