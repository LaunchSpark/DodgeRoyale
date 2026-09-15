//! Reference-style circular player and the shrinking trail from `code.lua`.

use bevy::{
    asset::RenderAssetUsages,
    image::ImageSampler,
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat},
};

use super::{
    art::{ActiveTheme, ink},
    ghost::Ghosted,
    player::{Player, move_player},
    screen::{GameEntity, Screen},
};
use crate::art::{PIXEL, SHADOW_OFFSET};
use crate::scale::WORLD_HALF_EXTENTS;
use crate::torus::{nearest_image, wrap_position};

const DIAMETER: f32 = 9.0 * PIXEL;
const SAMPLE_SECONDS: f32 = 1.0 / 60.0;
const TRAIL_SECONDS: f32 = 10.0 / 60.0;

#[derive(Resource)]
pub(super) struct PlayerArt(Handle<Image>);

impl FromWorld for PlayerArt {
    #[expect(
        clippy::arithmetic_side_effects,
        reason = "Raster coordinates range from -4 to 4, so squared distances fit in i16"
    )]
    fn from_world(world: &mut World) -> Self {
        // A tiny filled circle, rasterized in reference pixels. Tint the same mask
        // for the white body and theme shadow, with nearest-neighbor sampling.
        let pixels: Vec<u8> = (-4_i16..=4)
            .flat_map(|y| {
                (-4_i16..=4).flat_map(move |x| {
                    let alpha = if x * x + y * y <= 20 { 255 } else { 0 };
                    [255, 255, 255, alpha]
                })
            })
            .collect();
        let mut image = Image::new(
            Extent3d {
                width: 9,
                height: 9,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            pixels,
            TextureFormat::Rgba8UnormSrgb,
            RenderAssetUsages::default(),
        );
        image.sampler = ImageSampler::nearest();
        Self(world.resource_mut::<Assets<Image>>().add(image))
    }
}

impl PlayerArt {
    pub(super) fn sprite(&self, color: Color) -> Sprite {
        Sprite {
            image: self.0.clone(),
            color,
            custom_size: Some(Vec2::splat(DIAMETER)),
            ..default()
        }
    }
}

#[derive(Component, Default)]
pub(super) struct TrailEmitter {
    previous: Vec2,
    elapsed: f32,
}

#[derive(Component)]
struct TrailParticle {
    age: f32,
}

pub(super) struct PlayerArtPlugin;

impl Plugin for PlayerArtPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<PlayerArt>().add_systems(
            Update,
            (age_trail, emit_trail)
                .chain()
                .after(move_player)
                .run_if(in_state(Screen::Playing)),
        );
    }
}

#[expect(
    clippy::needless_pass_by_value,
    reason = "Bevy injects system parameters by value"
)]
fn age_trail(
    mut commands: Commands,
    time: Res<Time>,
    mut particles: Query<(Entity, &mut TrailParticle, &mut Sprite)>,
) {
    // Cap matches `emit_trail` and `advance_motion`: a WASM frame spike must not
    // instantly age particles to death before they have been seen even once.
    let delta = time.delta_secs().min(0.05);
    for (entity, mut particle, mut sprite) in &mut particles {
        particle.age += delta;
        if particle.age >= TRAIL_SECONDS {
            commands.entity(entity).despawn();
        } else {
            sprite.custom_size = Some(Vec2::splat(DIAMETER * 0.9_f32.powf(particle.age * 60.0)));
        }
    }
}

#[expect(
    clippy::needless_pass_by_value,
    reason = "Bevy injects system parameters by value"
)]
#[expect(
    clippy::arithmetic_side_effects,
    reason = "Finite positions and capped frame time bound particle emission and interpolation"
)]
fn emit_trail(
    mut commands: Commands,
    time: Res<Time>,
    art: Res<PlayerArt>,
    theme: Res<ActiveTheme>,
    player: Single<(&Transform, &mut TrailEmitter), With<Player>>,
) {
    let (transform, mut emitter) = player.into_inner();
    let position = transform.translation.truncate();
    // Seen from where the player is now, the previous position may lie beyond a
    // seam. Interpolating to its raw coordinate would string the trail straight
    // across the middle of the world instead of through the seam.
    let previous = nearest_image(position, emitter.previous, Some(WORLD_HALF_EXTENTS));
    // Match the player's background-tab cap, and emit at 60 Hz at interpolated
    // positions so low/high refresh rates produce the same trail density.
    let delta = time.delta_secs().min(0.05);
    let mut sample_time = SAMPLE_SECONDS - emitter.elapsed;
    // A capped 50 ms frame needs at most four samples, including roundoff.
    for _ in 0..4 {
        if sample_time > delta {
            break;
        }
        if position.distance_squared(previous) > 0.0001 {
            let center = wrap_position(
                previous.lerp(position, sample_time / delta),
                WORLD_HALF_EXTENTS,
            );
            let age = delta - sample_time;
            let size = Vec2::splat(DIAMETER * 0.9_f32.powf(age * 60.0));
            for (offset, color, depth) in [
                (SHADOW_OFFSET, theme.shadow(), 8.9),
                (Vec2::ZERO, ink(), 9.1),
            ] {
                let mut sprite = art.sprite(color);
                sprite.custom_size = Some(size);
                commands.spawn((
                    GameEntity,
                    Ghosted,
                    TrailParticle { age },
                    sprite,
                    Transform::from_translation((center + offset).extend(depth)),
                ));
            }
        }
        sample_time += SAMPLE_SECONDS;
    }
    emitter.elapsed = (emitter.elapsed + delta) % SAMPLE_SECONDS;
    emitter.previous = position;
}
