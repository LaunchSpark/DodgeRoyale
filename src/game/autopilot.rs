//! Browser watch mode: the real Bevy game asks the local policy server for an
//! action using the same observation encoder the headless trainer uses, and
//! draws the danger field that answer came out of.
//!
//! The field is the point of watching. An action alone says what the policy
//! did; the field says what it believed about every cell of the 256-pixel
//! window it was reading, which is the only way to tell a bad decision from a
//! bad picture of the world.
//!
//! It is drawn over the window it was computed for, not over wherever the
//! player has since moved to. The observation is player-centred at the instant
//! it is encoded, so pairing the reply with that frame's position keeps the hot
//! cells sitting on the hazards that caused them. At loopback latency the two
//! are a frame apart and the field looks centred on the player, which is what
//! it is; when inference falls behind, the field visibly lags rather than
//! silently disagreeing with the sprites underneath it.

use std::collections::VecDeque;

use bevy::asset::RenderAssetUsages;
use bevy::image::Image;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use js_sys::{Float32Array, Function, Reflect};
use wasm_bindgen::{JsCast, JsValue};

use crate::observation::{self, GRID, OBSERVATION_VALUES, WINDOW_PIXELS};
use crate::scale::{PIXEL, WORLD_HALF_EXTENTS};
use crate::simulation::{
    DEFAULT_HOLD_FRAMES, Player, PlayerIntent, PlayerSet, intent_from_command, view_world,
};
use crate::torus::wrapped_delta;

use super::player::read_keyboard;
use super::screen::{Screen, WatchMode};

/// Above the floor markings (-5) and below anything that moves, so the field
/// reads as the ground the action happens on rather than a pane in front of it.
const FIELD_Z: f32 = -4.0;

/// Frames of player positions kept for pairing a reply with its observation.
/// Two seconds: far longer than loopback inference takes, and short enough that
/// a stalled policy server drops its stale answer instead of drawing it.
const ANCHOR_FRAMES: usize = 120;

/// How fast the drawn colour range follows the field's own.
///
/// The range is tracked rather than rescaled per frame. A per-frame rescale
/// makes the whole picture flash whenever the hottest cell changes, which is
/// every time an enemy moves.
const RANGE_SMOOTHING: f32 = 0.1;

/// Safe to lethal. The first two stops are the world's own blues, so an empty
/// window sinks into the floor; the last two are the hazard colours, so danger
/// arrives as the colour of the thing causing it.
const RAMP: [(f32, [f32; 3]); 4] = [
    (0.00, [0.09, 0.22, 0.29]),
    (0.45, [0.17, 0.54, 0.74]),
    (0.75, [0.94, 0.67, 0.24]),
    (1.00, [0.91, 0.24, 0.24]),
];

/// Opacity at the safest and most dangerous cell. The floor stays legible under
/// calm water, and a lethal cell is unmistakable without hiding what is
/// standing in it.
const ALPHA_CALM: f32 = 0.12;
const ALPHA_LETHAL: f32 = 0.74;

#[derive(Resource)]
struct PolicyBridge {
    layout: String,
    frame: u32,
}

/// The quad the field is painted on.
#[derive(Component)]
struct DangerField;

#[derive(Resource)]
struct FieldOverlay {
    image: Handle<Image>,
    /// The smoothed colour range: the field's own min and max, followed.
    range: Option<(f32, f32)>,
    /// The frame whose field is currently on screen, so an unchanged reply is
    /// not re-uploaded sixty times a second.
    drawn: Option<u32>,
    /// Where the player was on each recent frame, for anchoring a reply.
    anchors: VecDeque<(u32, Vec2)>,
}

pub(super) struct AutopilotPlugin;

impl Plugin for AutopilotPlugin {
    fn build(&self, app: &mut App) {
        let enabled = watch_enabled();
        let layout =
            serde_json::to_string(&observation::layout(DEFAULT_HOLD_FRAMES)).unwrap_or_default();
        app.insert_resource(WatchMode(enabled))
            .insert_resource(PolicyBridge { layout, frame: 0 })
            .add_systems(Startup, spawn_field)
            .add_systems(OnEnter(Screen::Playing), (notify_episode, forget_field))
            .add_systems(OnExit(Screen::Playing), hide_field)
            .add_systems(
                Update,
                (
                    drive_agent.after(read_keyboard).before(PlayerSet::Move),
                    draw_field.after(PlayerSet::Move),
                )
                    .chain()
                    .run_if(in_state(Screen::Playing)),
            );
        if enabled {
            // Replace the Menu initial event before the first frame. The viewer
            // never creates menu entities, even briefly during startup.
            app.insert_state(Screen::Playing);
        }
    }
}

pub(super) fn watch_enabled() -> bool {
    browser_call("dodgeWatchEnabled")
        .and_then(|function| function.call0(&JsValue::NULL).ok())
        .and_then(|value| value.as_bool())
        .unwrap_or(false)
}

fn browser_call(name: &str) -> Option<Function> {
    Reflect::get(&js_sys::global(), &JsValue::from_str(name))
        .ok()?
        .dyn_into::<Function>()
        .ok()
}

#[expect(
    clippy::needless_pass_by_value,
    reason = "Bevy injects resources by value"
)]
fn notify_episode(mode: Res<WatchMode>) {
    if mode.0
        && let Some(function) = browser_call("dodgeEpisode")
    {
        let _ = function.call0(&JsValue::NULL);
    }
}

/// The field's texture and the quad it is drawn on, spawned once and hidden
/// until the first field arrives.
#[expect(
    clippy::needless_pass_by_value,
    reason = "Bevy injects resources by value"
)]
fn spawn_field(mut commands: Commands, mut images: ResMut<Assets<Image>>, mode: Res<WatchMode>) {
    if !mode.0 {
        return;
    }
    let edge = u32::try_from(GRID).unwrap_or(1);
    let image = images.add(Image::new_fill(
        Extent3d {
            width: edge,
            height: edge,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, 0, 0, 0],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD | RenderAssetUsages::MAIN_WORLD,
    ));
    commands.insert_resource(FieldOverlay {
        image: image.clone(),
        range: None,
        drawn: None,
        anchors: VecDeque::new(),
    });
    // The window is 256 reference pixels across by definition, so the quad is
    // exactly the region the observation describes. Sampling is the engine's
    // default -- bilinear, clamped at the edge -- which is the same reading the
    // controller's own `grid_sample` makes of this field.
    commands.spawn((
        DangerField,
        Sprite {
            image,
            custom_size: Some(Vec2::splat(WINDOW_PIXELS * PIXEL)),
            ..default()
        },
        Transform::from_xyz(0.0, 0.0, FIELD_Z),
        Visibility::Hidden,
    ));
}

fn forget_field(overlay: Option<ResMut<FieldOverlay>>) {
    if let Some(mut overlay) = overlay {
        // A new episode's first field belongs to a world the old range says
        // nothing about, and the old anchors are positions in a dead run.
        overlay.range = None;
        overlay.drawn = None;
        overlay.anchors.clear();
    }
}

fn hide_field(mut field: Query<&mut Visibility, With<DangerField>>) {
    for mut visibility in &mut field {
        *visibility = Visibility::Hidden;
    }
}

/// The direction the policy last answered with, or nowhere.
fn latest_direction() -> Vec2 {
    browser_call("dodgeAction")
        .and_then(|function| function.call0(&JsValue::NULL).ok())
        .and_then(|value| value.dyn_into::<Float32Array>().ok())
        .filter(|values| values.length() == 2)
        .map(|values| Vec2::new(values.get_index(0), values.get_index(1)))
        .filter(|direction| direction.is_finite())
        .unwrap_or(Vec2::ZERO)
}

fn drive_agent(world: &mut World) {
    if !world.resource::<WatchMode>().0 {
        return;
    }
    let (layout, frame) = {
        let mut bridge = world.resource_mut::<PolicyBridge>();
        bridge.frame = bridge.frame.saturating_add(1);
        (bridge.layout.clone(), bridge.frame)
    };

    // Every frame, because the trainer decides every frame: a STEP advances the
    // arena exactly one 60 Hz frame on exactly one direction. `hold_frames` is
    // the span the observation's candidate paths are predicted over and never
    // an action repeat, so holding a direction here would be a cadence the
    // policy was never trained on.
    let direction = intent_from_command(latest_direction());
    let mut players = world.query_filtered::<&mut PlayerIntent, With<Player>>();
    for mut intent in players.iter_mut(world) {
        intent.0 = direction;
    }

    let view = view_world(world, frame);
    // Recorded before the observation is sent, because the reply that comes
    // back is about the window centred here, whatever the player does next.
    if let Some(position) = view.player.as_ref().map(|player| player.position)
        && let Some(mut overlay) = world.get_resource_mut::<FieldOverlay>()
    {
        overlay.anchors.push_back((frame, position));
        while overlay.anchors.len() > ANCHOR_FRAMES {
            overlay.anchors.pop_front();
        }
    }

    let mut observation = vec![0.0; OBSERVATION_VALUES];
    if observation::encode(&view, DEFAULT_HOLD_FRAMES, &mut observation).is_ok()
        && let Some(function) = browser_call("dodgeObserve")
    {
        let values = Float32Array::from(observation.as_slice());
        let _ = function.call3(
            &JsValue::NULL,
            &values,
            &JsValue::from_str(&layout),
            &JsValue::from_f64(f64::from(frame)),
        );
    }
}

/// The most recent reply's field, and the frame whose observation produced it.
fn latest_field() -> Option<(u32, Vec<f32>)> {
    let published = browser_call("dodgeField")?
        .call0(&JsValue::NULL)
        .ok()
        .filter(|value| !value.is_null() && !value.is_undefined())?;
    let frame = Reflect::get(&published, &JsValue::from_str("frame"))
        .ok()?
        .as_f64()?;
    let values = Reflect::get(&published, &JsValue::from_str("values"))
        .ok()?
        .dyn_into::<Float32Array>()
        .ok()?;
    if usize::try_from(values.length()).ok()? != GRID.saturating_mul(GRID) {
        return None;
    }
    // The frame number crossed JS as a double. Nothing here needs to prove it
    // is one of ours: a value this game never sent finds no anchor below, and
    // an unanchored field is not drawn.
    Some((frame_number(frame)?, values.to_vec()))
}

/// A JavaScript number as a frame number, or nothing if it cannot be one.
#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "Truncation only widens what try_from then rejects"
)]
fn frame_number(value: f64) -> Option<u32> {
    u32::try_from(value.trunc() as i64).ok()
}

#[expect(
    clippy::needless_pass_by_value,
    reason = "Bevy injects resources by value"
)]
#[expect(
    clippy::type_complexity,
    reason = "One Bevy query, spelled out rather than hidden behind an alias"
)]
fn draw_field(
    mode: Res<WatchMode>,
    overlay: Option<ResMut<FieldOverlay>>,
    mut images: ResMut<Assets<Image>>,
    players: Query<&Transform, With<Player>>,
    mut field: Query<(&mut Transform, &mut Visibility), (With<DangerField>, Without<Player>)>,
) {
    if !mode.0 {
        return;
    }
    let (Some(mut overlay), Ok(player), Ok((mut transform, mut visibility))) =
        (overlay, players.single(), field.single_mut())
    else {
        return;
    };

    let Some((frame, values)) = latest_field() else {
        *visibility = Visibility::Hidden;
        return;
    };
    let Some((_, anchor)) = overlay
        .anchors
        .iter()
        .rev()
        .find(|(recorded, _)| *recorded == frame)
        .copied()
    else {
        // Older than the ring, so the position it describes is no longer
        // known. Better no picture than one placed where nothing was read.
        *visibility = Visibility::Hidden;
        return;
    };

    // The nearest wrapped image of the anchor, for the same reason the camera
    // chases the nearest image of the player: across a seam the far coordinate
    // is a whole world away, and the field would fly off the screen.
    let here = player.translation.truncate();
    let centre = nearest_image(here, anchor);
    transform.translation = centre.extend(FIELD_Z);
    *visibility = Visibility::Visible;

    if overlay.drawn == Some(frame) {
        return;
    }
    overlay.drawn = Some(frame);
    let range = follow_range(overlay.range, &values);
    overlay.range = Some(range);
    if let Some(mut image) = images.get_mut(&overlay.image) {
        image.data = Some(shade_field(&values, range));
        // Report what actually reached the texture. Without it nothing outside
        // this function can tell a field that was drawn from one that arrived,
        // failed to find its anchor, and was silently dropped.
        if let Some(function) = browser_call("dodgeFieldDrawn") {
            let _ = function.call1(&JsValue::NULL, &JsValue::from_f64(f64::from(frame)));
        }
    }
}

/// The anchor as the copy of it closest to where the player is now.
#[expect(
    clippy::arithmetic_side_effects,
    reason = "Bounded world positions: a wrapped offset added to a position"
)]
fn nearest_image(here: Vec2, anchor: Vec2) -> Vec2 {
    here + wrapped_delta(here, anchor, WORLD_HALF_EXTENTS)
}

/// The colour range, eased toward this field's own lowest and highest cell.
fn follow_range(previous: Option<(f32, f32)>, values: &[f32]) -> (f32, f32) {
    let low = values.iter().copied().fold(f32::INFINITY, f32::min);
    let high = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !low.is_finite() || !high.is_finite() {
        return previous.unwrap_or((0.0, 1.0));
    }
    match previous {
        None => (low, high),
        Some((was_low, was_high)) => (
            RANGE_SMOOTHING.mul_add(low - was_low, was_low),
            RANGE_SMOOTHING.mul_add(high - was_high, was_high),
        ),
    }
}

/// The field as RGBA rows, top row first, matching the observation window.
fn shade_field(values: &[f32], (low, high): (f32, f32)) -> Vec<u8> {
    let span = high - low;
    let mut data = Vec::with_capacity(values.len().saturating_mul(4));
    for value in values {
        let danger = if span > f32::EPSILON {
            (value - low) / span
        } else {
            0.0
        };
        data.extend_from_slice(&shade(danger.clamp(0.0, 1.0)));
    }
    data
}

/// One normalised danger value as an RGBA pixel.
fn shade(danger: f32) -> [u8; 4] {
    let mut colour = RAMP.last().map_or([0.0, 0.0, 0.0], |(_, rgb)| *rgb);
    for pair in RAMP.windows(2) {
        if let [(low, from), (high, to)] = pair
            && danger <= *high
        {
            let span = high - low;
            let towards = if span > f32::EPSILON {
                ((danger - low) / span).clamp(0.0, 1.0)
            } else {
                0.0
            };
            colour = [
                (to[0] - from[0]).mul_add(towards, from[0]),
                (to[1] - from[1]).mul_add(towards, from[1]),
                (to[2] - from[2]).mul_add(towards, from[2]),
            ];
            break;
        }
    }
    let alpha = (ALPHA_LETHAL - ALPHA_CALM).mul_add(danger, ALPHA_CALM);
    [
        byte(colour[0]),
        byte(colour[1]),
        byte(colour[2]),
        byte(alpha),
    ]
}

#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "Clamped into [0, 1] and scaled, so the cast cannot lose a digit or sign"
)]
fn byte(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}
