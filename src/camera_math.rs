//! Pure camera geometry, testable without rendering.

pub use crate::scale::{VIEW_HEIGHT, VIEW_WIDTH};

use bevy::prelude::*;

pub const CAMERA_DECAY: f32 = 12.0;

#[allow(
    clippy::arithmetic_side_effects,
    reason = "Viewport dimensions are positive after guarding a minimized canvas"
)]
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "The wrapping camera does not clamp; seam ghosting needs this next"
    )
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
