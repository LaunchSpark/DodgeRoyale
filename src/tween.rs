//! Reusable exponential tweening for continuously changing targets.

use bevy::math::StableInterpolate;

/// Approach a target using `1 - exp(-decay_rate * delta_secs)` as the blend factor.
///
/// Works with scalars, vectors, and other stable interpolatable values. Higher
/// decay rates follow more tightly. A fixed target gives the same result for
/// equal elapsed time regardless of how frames divide that time. Changing the
/// target continues from the current value without restarting an animation.
///
/// Supply finite values and timing. Nonpositive rates or time leave the current
/// value unchanged. This approaches the target asymptotically, without a duration.
#[must_use]
#[allow(
    clippy::arithmetic_side_effects,
    reason = "Positive finite float timing produces a blend factor between zero and one"
)]
pub fn exponential<T: StableInterpolate>(
    current: &T,
    target: &T,
    decay_rate: f32,
    delta_secs: f32,
) -> T {
    if decay_rate <= 0.0 || delta_secs <= 0.0 {
        return current.clone();
    }
    // exp_m1 preserves precision for small frame times: -(exp(-rate * dt) - 1).
    let blend = -(-decay_rate * delta_secs).exp_m1();
    current.interpolate_stable(target, blend)
}

#[cfg(test)]
mod tests {
    use super::exponential;
    use bevy::math::{Vec2, Vec3};

    #[test]
    #[expect(
        clippy::float_cmp,
        reason = "Paused tweening must return the original value exactly"
    )]
    fn paused_time_or_rate_does_not_move() {
        for (rate, delta) in [(12.0, 0.0), (0.0, 1.0), (12.0, -1.0), (-1.0, 1.0)] {
            assert_eq!(exponential(&3.0_f32, &9.0, rate, delta), 3.0);
        }
    }

    #[test]
    fn a_long_frame_reaches_the_target_without_overshooting() {
        let current = Vec3::new(10.0, -10.0, 7.0);
        let target = Vec3::new(-100.0, 100.0, 7.0);
        let result = exponential(&current, &target, 12.0, 10.0);
        assert_eq!(result, target);
    }

    #[test]
    fn reversing_targets_stays_bounded_and_matches_across_frame_rates() {
        let simulate = |frames: u16, delta: f32| {
            let mut position = Vec2::ZERO;
            for target in [
                Vec2::new(400.0, -200.0),
                Vec2::new(-400.0, 200.0),
                Vec2::ZERO,
            ] {
                for _ in 0..frames {
                    let next = exponential(&position, &target, 12.0, delta);
                    assert!(next.cmpge(position.min(target)).all());
                    assert!(next.cmple(position.max(target)).all());
                    position = next;
                }
            }
            position
        };
        assert!(simulate(15, 1.0 / 30.0).abs_diff_eq(simulate(72, 1.0 / 144.0), 0.001));
    }
}
