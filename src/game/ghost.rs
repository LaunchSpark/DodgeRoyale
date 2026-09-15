//! Draws a wrapping world so its seams are invisible.
//!
//! Positions are canonical inside `[-half, half)`, so an enemy twenty units away
//! across a seam still *stores* a coordinate a whole world away and would be
//! drawn off screen. Each frame every marked entity is moved to the image of
//! itself nearest the camera, which is the one the player can actually see.
//!
//! This is safe to do to simulation state: wrapping is invariant to which image
//! a position is expressed as, and the next movement step re-canonicalises it.

use bevy::prelude::*;
use bevy::transform::TransformSystems;

use crate::scale::WORLD_HALF_EXTENTS;
use crate::torus::nearest_image;

/// Marks an entity that should be drawn at whichever image is nearest the camera.
#[derive(Component)]
pub(super) struct Ghosted;

pub(super) struct GhostPlugin;

impl Plugin for GhostPlugin {
    fn build(&self, app: &mut App) {
        // After movement, before transforms propagate to the renderer.
        app.add_systems(
            PostUpdate,
            place_at_nearest_image.before(TransformSystems::Propagate),
        );
    }
}

#[expect(
    clippy::needless_pass_by_value,
    reason = "Bevy injects system parameters by value"
)]
fn place_at_nearest_image(
    camera: Single<&Transform, (With<Camera2d>, Without<Ghosted>)>,
    mut ghosted: Query<&mut Transform, (With<Ghosted>, Without<Camera2d>)>,
) {
    let eye = camera.translation.truncate();
    for mut transform in &mut ghosted {
        let image = nearest_image(
            eye,
            transform.translation.truncate(),
            Some(WORLD_HALF_EXTENTS),
        );
        transform.translation.x = image.x;
        transform.translation.y = image.y;
    }
}
