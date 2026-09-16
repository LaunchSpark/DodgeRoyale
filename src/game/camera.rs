//! An independent, exponentially smoothed camera with viewport-aware boundaries.

use bevy::{camera::ScalingMode, prelude::*};

use super::player::{Player, Velocity};
use crate::simulation::PlayerSet;

use super::screen::Screen;

use crate::camera_math::{CAMERA_DECAY, VIEW_HEIGHT, VIEW_WIDTH};
use crate::motion::MAX_FRAME_SECONDS;
use crate::scale::WORLD_HALF_EXTENTS;
use crate::torus::{wrap_position, wrapped_delta};
use crate::tween;

const LOOKAHEAD_SECONDS: f32 = 0.15;

pub(super) struct FollowCameraPlugin;

impl Plugin for FollowCameraPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_camera)
            .add_systems(OnEnter(Screen::Menu), center_camera)
            .add_systems(OnEnter(Screen::Config), center_camera)
            .add_systems(
                Update,
                follow_player
                    .after(PlayerSet::Move)
                    .run_if(in_state(Screen::Playing)),
            );
    }
}

#[derive(Component)]
struct FollowCamera;

fn spawn_camera(mut commands: Commands) {
    commands.spawn((
        Camera2d,
        FollowCamera,
        Projection::Orthographic(OrthographicProjection {
            scaling_mode: ScalingMode::AutoMax {
                max_width: VIEW_WIDTH,
                max_height: VIEW_HEIGHT,
            },
            ..OrthographicProjection::default_2d()
        }),
    ));
}

#[expect(
    clippy::needless_pass_by_value,
    reason = "Bevy injects system parameters by value"
)]
#[allow(
    clippy::arithmetic_side_effects,
    reason = "Player velocity and lookahead time are bounded gameplay values"
)]
fn follow_player(
    player: Single<(&Transform, &Velocity), With<Player>>,
    mut camera: Single<&mut Transform, (With<FollowCamera>, Without<Player>)>,
    time: Res<Time>,
    input: Res<ButtonInput<KeyCode>>,
) {
    let (player_transform, velocity) = *player;
    let desired = wrap_position(
        player_transform.translation.truncate() + velocity.0 * LOOKAHEAD_SECONDS,
        WORLD_HALF_EXTENTS,
    );
    let here = camera.translation.truncate();
    // Chase the nearest image of the player. Without this a seam crossing looks
    // like the player fled to the far side, and the camera pans the long way.
    let target = here + wrapped_delta(here, desired, WORLD_HALF_EXTENTS);
    let next = if input.just_pressed(KeyCode::KeyR) {
        target
    } else {
        // Cap the frame time to match `advance_motion`: a backgrounded browser
        // tab or a frame spike must not snap the camera to its target the way a
        // huge delta otherwise would (blend factor → 1.0). The player already
        // applies this same cap, so the camera lag stays perceptually consistent
        // across refresh rates and tab-switch resumptions on both native and web.
        let dt = time.delta_secs().min(MAX_FRAME_SECONDS);
        tween::exponential(&here, &target, CAMERA_DECAY, dt)
    };
    camera.translation = wrap_position(next, WORLD_HALF_EXTENTS).extend(camera.translation.z);
}

/// Return the camera to the world origin, where the menu screens are drawn.
///
/// Gameplay leaves the camera wherever the player died. Without this, a menu
/// drawn at the origin would sit off-screen and the player would see an empty
/// background. The wipe covers the screen at this moment, so the jump is unseen.
fn center_camera(mut camera: Single<&mut Transform, With<FollowCamera>>) {
    camera.translation.x = 0.0;
    camera.translation.y = 0.0;
}
