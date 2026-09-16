//! Lightweight world-space, axis-aligned hitboxes shared by actors.

use bevy::prelude::*;

/// A sensor hitbox in world units. Rotation and transform scale do not change it.
///
/// Keep colliding actors unparented. This detects overlap; it does not resolve
/// penetration or simulate rigid bodies. Touching edges count as contact.
#[derive(Component, Debug, Clone, Copy, PartialEq)]
pub struct Collider {
    /// Positive half width and half height, independent of sprite size.
    pub half_extents: Vec2,
    /// World-space offset from the actor's transform translation.
    pub offset: Vec2,
    /// Disable contact reporting without removing the component.
    pub enabled: bool,
}

impl Default for Collider {
    fn default() -> Self {
        Self::rectangle(Vec2::splat(12.0))
    }
}

impl Collider {
    /// Construct a centered, enabled rectangular hitbox.
    #[must_use]
    pub const fn rectangle(half_extents: Vec2) -> Self {
        Self {
            half_extents,
            offset: Vec2::ZERO,
            enabled: true,
        }
    }

    /// Invalid geometry is excluded from movement and collision calculations.
    #[must_use]
    pub fn is_valid(self) -> bool {
        self.half_extents.is_finite()
            && self.half_extents.cmpgt(Vec2::ZERO).all()
            && self.offset.is_finite()
    }

    /// Check a pair of translated hitboxes, including their offsets.
    ///
    /// Measures across a flat plane. Use [`Self::overlaps_wrapped`] in a world
    /// that wraps, where two boxes either side of a seam are in fact adjacent.
    #[must_use]
    pub fn overlaps(self, position: Vec2, other: Self, other_position: Vec2) -> bool {
        self.overlaps_wrapped(position, other, other_position, None)
    }

    /// Check a pair of translated hitboxes in a world that may wrap.
    ///
    /// `wrap` carries the world's half extents when it is a torus, so the pair
    /// is measured by the shorter of the direct route and the one across a seam.
    #[must_use]
    #[expect(
        clippy::arithmetic_side_effects,
        reason = "Finite world-space hitbox arithmetic"
    )]
    pub fn overlaps_wrapped(
        self,
        position: Vec2,
        other: Self,
        other_position: Vec2,
        wrap: Option<Vec2>,
    ) -> bool {
        if !self.enabled
            || !other.enabled
            || !self.is_valid()
            || !other.is_valid()
            || !position.is_finite()
            || !other_position.is_finite()
        {
            return false;
        }
        let from = position + self.offset;
        let to = other_position + other.offset;
        let separation = wrap.map_or_else(
            || to - from,
            |half_extents| crate::torus::wrapped_delta(from, to, half_extents),
        );
        separation
            .abs()
            .cmple(self.half_extents + other.half_extents)
            .all()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hitboxes_touch_across_a_seam_in_a_wrapping_world() {
        let world = Vec2::new(300.0, 100.0);
        let a = Collider::rectangle(Vec2::splat(10.0));
        let b = Collider::rectangle(Vec2::splat(10.0));
        // Five units apart, but on opposite sides of the seam.
        let left = Vec2::new(-298.0, 0.0);
        let right = Vec2::new(297.0, 0.0);
        assert!(
            a.overlaps_wrapped(left, b, right, Some(world)),
            "a seam crossing must not hide a contact"
        );
        // The same pair on a flat plane is far apart.
        assert!(!a.overlaps(left, b, right));
        assert!(!a.overlaps_wrapped(left, b, right, None));
    }

    #[test]
    fn a_wrapping_world_does_not_invent_contacts_at_a_distance() {
        let world = Vec2::new(300.0, 100.0);
        let a = Collider::rectangle(Vec2::splat(10.0));
        let b = Collider::rectangle(Vec2::splat(10.0));
        // Mid-world, far from both each other and any seam.
        assert!(!a.overlaps_wrapped(
            Vec2::new(-100.0, 0.0),
            b,
            Vec2::new(100.0, 0.0),
            Some(world)
        ));
        // Touching pairs still touch.
        assert!(a.overlaps_wrapped(Vec2::ZERO, b, Vec2::new(19.0, 0.0), Some(world)));
    }

    #[test]
    fn contact_respects_edges_offsets_and_both_axes() {
        let a = Collider::rectangle(Vec2::new(10.0, 5.0));
        let b = Collider {
            offset: Vec2::new(-5.0, 0.0),
            ..a
        };
        assert!(a.overlaps(Vec2::ZERO, b, Vec2::new(25.0, 10.0)));
        assert!(!a.overlaps(Vec2::ZERO, b, Vec2::new(25.01, 10.0)));
        assert!(!a.overlaps(Vec2::ZERO, b, Vec2::new(25.0, 10.01)));
        assert!(b.overlaps(Vec2::new(25.0, 10.0), a, Vec2::ZERO));
    }

    #[test]
    fn disabled_and_invalid_colliders_never_contact() {
        let good = Collider::default();
        for bad in [
            Collider {
                enabled: false,
                ..good
            },
            Collider::rectangle(Vec2::ZERO),
            Collider::rectangle(Vec2::splat(-1.0)),
            Collider::rectangle(Vec2::splat(f32::NAN)),
        ] {
            assert!(!good.overlaps(Vec2::ZERO, bad, Vec2::ZERO));
            assert!(!bad.overlaps(Vec2::ZERO, good, Vec2::ZERO));
        }
    }
}
