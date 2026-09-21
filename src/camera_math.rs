//! Pure camera geometry, testable without rendering.

pub use crate::scale::{VIEW_HEIGHT, VIEW_WIDTH};

use bevy::prelude::*;

pub const CAMERA_DECAY: f32 = 12.0;

/// How far the camera lets its subject drift from centre before following.
///
/// Four percent of the visible height: a little over one player wide, and far
/// smaller than the lookahead at any real travelling speed, so a player
/// actually going somewhere is still followed at once. What it removes is the
/// camera answering movement that is not going anywhere -- a tap, a reversal,
/// the wobble of a player holding position -- which reads as the world shaking
/// rather than as the camera tracking.
pub const CAMERA_DEADZONE: f32 = VIEW_HEIGHT * 0.04;

/// How far the camera should move, given where its subject is relative to it.
///
/// Inside the deadzone the answer is zero and the camera holds still. Outside,
/// the camera moves only far enough to put the subject back on the edge of the
/// zone, never to centre it, so the result is continuous as the subject crosses
/// the edge. Moving to centre instead would jump by the zone's whole width the
/// instant it was crossed: a lurch at the boundary, and chatter around it.
#[allow(
    clippy::arithmetic_side_effects,
    reason = "Bounded: the clamp never returns a vector longer than the offset"
)]
pub fn follow_offset(offset: Vec2, deadzone: f32) -> Vec2 {
    offset - offset.clamp_length_max(deadzone.max(0.0))
}

#[allow(
    clippy::arithmetic_side_effects,
    reason = "Viewport dimensions are positive after guarding a minimized canvas"
)]
pub fn viewport_half_size(width: f32, height: f32) -> Vec2 {
    let aspect = width.max(1.0) / height.max(1.0);
    let visible_height = VIEW_HEIGHT.min(VIEW_WIDTH / aspect);
    Vec2::new(visible_height * aspect, visible_height) * 0.5
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn viewport_handles_resize_and_extreme_aspect_ratios() {
        assert!(viewport_half_size(1600.0, 800.0).abs_diff_eq(Vec2::new(800.0, 400.0), 0.001));
        assert!(
            viewport_half_size(4000.0, 100.0)
                .cmple(Vec2::new(800.0, 400.0))
                .all()
        );
        assert!(viewport_half_size(0.0, 0.0).is_finite());
    }

    #[test]
    fn a_subject_inside_the_deadzone_does_not_move_the_camera() {
        for offset in [
            Vec2::ZERO,
            Vec2::new(CAMERA_DEADZONE - 0.01, 0.0),
            Vec2::new(0.0, -CAMERA_DEADZONE + 0.01),
            Vec2::splat(CAMERA_DEADZONE * 0.7),
        ] {
            assert_eq!(
                follow_offset(offset, CAMERA_DEADZONE),
                Vec2::ZERO,
                "{offset}"
            );
        }
    }

    #[test]
    fn outside_the_deadzone_the_subject_lands_on_its_edge() {
        for offset in [
            Vec2::new(CAMERA_DEADZONE * 4.0, 0.0),
            Vec2::new(-200.0, 300.0),
            Vec2::splat(-1_000.0),
        ] {
            let moved = follow_offset(offset, CAMERA_DEADZONE);
            let remaining = offset - moved;
            assert!(
                (remaining.length() - CAMERA_DEADZONE).abs() < 0.001,
                "{offset} left {remaining} from centre"
            );
            // Straight at the subject: following must not introduce a sideways
            // drift the player never asked for.
            assert!(moved.normalize().abs_diff_eq(offset.normalize(), 0.001));
        }
    }

    /// The reason the camera moves to the zone's edge and not to its centre.
    #[test]
    fn crossing_the_edge_is_continuous() {
        let inside = follow_offset(Vec2::new(CAMERA_DEADZONE - 0.001, 0.0), CAMERA_DEADZONE);
        let outside = follow_offset(Vec2::new(CAMERA_DEADZONE + 0.001, 0.0), CAMERA_DEADZONE);
        assert!(
            (outside - inside).length() < 0.01,
            "the target jumped by {} crossing the edge",
            (outside - inside).length()
        );
    }

    #[test]
    fn a_zero_deadzone_follows_exactly_as_before() {
        let offset = Vec2::new(37.0, -11.0);
        assert_eq!(follow_offset(offset, 0.0), offset);
        assert_eq!(
            follow_offset(offset, -5.0),
            offset,
            "a negative zone is no zone"
        );
    }

    /// A player crossing the world must not be held back by the zone: at top
    /// speed the lookahead alone is several times its radius.
    #[test]
    fn travelling_at_speed_is_followed_at_nearly_full_rate() {
        let lookahead = crate::motion::top_speed() * 0.15;
        assert!(lookahead > CAMERA_DEADZONE * 3.0, "lookahead {lookahead}");
        let offset = Vec2::new(lookahead, 0.0);
        let followed = follow_offset(offset, CAMERA_DEADZONE).length();
        assert!(followed > offset.length() * 0.7, "held back to {followed}");
    }

    /// The complaint this exists to answer: a player fidgeting in place must
    /// leave the world still. A per-step check cannot say that -- a residual
    /// too small to see in one frame could still walk the camera across the
    /// screen over a few seconds -- so this runs ten seconds and then a hundred
    /// and requires the same bound from both.
    ///
    /// Not exact equality. `lerp` is `self * (1 - t) + other * t`, which loses a
    /// unit in the last place even when the two are the same point, so a camera
    /// asked to stay exactly where it is still rounds. What matters is that the
    /// error is a rounding artifact and not a drift: a thousandth of a world
    /// unit is under a thousandth of a rendered pixel, and it does not grow.
    #[test]
    fn a_subject_fidgeting_inside_the_zone_does_not_move_the_camera() {
        let start = Vec2::new(-137.0, 42.0);
        let settle = |frames: u16| {
            let mut camera = start;
            for frame in 0..frames {
                let angle = f32::from(frame) * 0.7;
                let wobble = Vec2::new(angle.cos(), angle.sin()) * (CAMERA_DEADZONE * 0.9);
                let target = camera + follow_offset(start + wobble - camera, CAMERA_DEADZONE);
                camera = crate::tween::exponential(&camera, &target, CAMERA_DECAY, 1.0 / 60.0);
            }
            (camera - start).length()
        };
        let ten_seconds = settle(600);
        let a_hundred = settle(6_000);
        assert!(ten_seconds < 0.001, "drifted {ten_seconds} in ten seconds");
        assert!(
            a_hundred < 0.001,
            "drifted {a_hundred} in a hundred seconds, from {ten_seconds} in ten"
        );
    }

    /// Repeated steps settle with the subject resting on the edge of the zone
    /// rather than drifting back to centre, which is what keeps it a zone.
    #[test]
    fn the_camera_settles_on_the_edge_and_stays_there() {
        let subject = Vec2::new(500.0, 0.0);
        let mut camera = Vec2::ZERO;
        for _ in 0..600 {
            let target = camera + follow_offset(subject - camera, CAMERA_DEADZONE);
            camera = crate::tween::exponential(&camera, &target, CAMERA_DECAY, 1.0 / 60.0);
        }
        let remaining = (subject - camera).length();
        assert!(
            (remaining - CAMERA_DEADZONE).abs() < 1.0,
            "settled {remaining} from the subject"
        );
    }

    #[test]
    fn exponential_follow_is_stable_across_frame_rates() {
        let target = Vec2::new(400.0, -240.0);
        let mut slow = Vec2::ZERO;
        let mut fast = Vec2::ZERO;
        for _ in 0..30 {
            slow = crate::tween::exponential(&slow, &target, CAMERA_DECAY, 1.0 / 30.0);
        }
        for _ in 0..144 {
            fast = crate::tween::exponential(&fast, &target, CAMERA_DECAY, 1.0 / 144.0);
        }
        assert!(slow.abs_diff_eq(fast, 0.001));
        assert!(slow.cmple(target.max(Vec2::ZERO)).all());
        assert!(slow.cmpge(target.min(Vec2::ZERO)).all());
    }
}
