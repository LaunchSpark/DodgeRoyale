//! Turn one frame of the arena into the numbers a policy reads.
//!
//! The window is local and deliberately so: 256 reference pixels across,
//! centred on the player, wrapping with the world. Threats outside it are
//! unobserved, which is a design choice rather than an oversight -- a grid over
//! the whole 960x640 arena would put 30x20 pixels in a cell, and an ordinary
//! enemy is four to seven pixels wide, so the detail that decides a dodge would
//! be averaged away.
//!
//! One encoder serves both training and, later, play: the policy that learns
//! from these numbers must see the same numbers when it drives the real game.

#![expect(
    clippy::arithmetic_side_effects,
    reason = "Pixel geometry over finite, validated values; indices are clamped into the grid"
)]

use bevy::math::Vec2;
use serde::{Deserialize, Serialize};

use crate::scale::PIXEL;
use crate::simulation::{
    Action, ArenaError, ArenaView, BlastView, EnemyView, PlayerView, SAMPLE_COUNT, SAMPLE_FRAMES,
    predict_path,
};
use crate::torus::wrapped_delta;

/// Cells across the observed window, on each axis.
pub const GRID: usize = 64;

/// [`GRID`] as a float, for pixel arithmetic. A test keeps the two in step.
const GRID_EDGE: f32 = 64.0;

/// Reference pixels a cell covers, on each axis.
pub const CELL_PIXELS: f32 = 4.0;

/// Reference pixels the window covers, on each axis.
pub const WINDOW_PIXELS: f32 = GRID_EDGE * CELL_PIXELS;

/// Half the window, which is where the player always is.
pub const WINDOW_HALF: f32 = WINDOW_PIXELS / 2.0;

/// Cells in one channel.
pub const CELLS: usize = GRID * GRID;

/// Divisor putting velocities in roughly [-1, 1].
///
/// The fastest thing in the arena is the player, at 2.5 reference pixels a
/// frame; enemies travel well under one.
pub const VELOCITY_SCALE: f32 = 4.0;

/// Values before the channels: the player's own velocity.
pub const PLAYER_VALUES: usize = 2;

/// Values after the channels: nine paths of six samples in x and y.
pub const PATH_VALUES: usize = 9 * SAMPLE_COUNT * 2;

/// Every value in one observation.
pub const OBSERVATION_VALUES: usize = PLAYER_VALUES + CHANNELS.len() * CELLS + PATH_VALUES;

/// What each grid channel holds, in order.
///
/// The names travel in the handshake, so Python reads the layout rather than
/// assuming it: a channel that moves without the name moving with it would
/// repoint every trained filter at a different thing.
pub const CHANNELS: [Channel; 7] = [
    Channel::NormalEnemy,
    Channel::Kamikaze,
    Channel::Blast,
    Channel::BlastPhase,
    Channel::Player,
    Channel::VelocityX,
    Channel::VelocityY,
];

/// One grid of numbers, one value per cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Channel {
    /// An ordinary enemy occupies the cell.
    NormalEnemy,
    /// A kamikaze occupies the cell.
    Kamikaze,
    /// A kamikaze blast covers the cell.
    Blast,
    /// Where a blast is in its life: +1 just detonated and growing, 0 at its
    /// widest, -1 about to vanish. Size alone cannot say which, because a
    /// blast passes through every width twice.
    BlastPhase,
    /// The player's own hitbox.
    Player,
    /// Horizontal velocity of whatever owns the cell.
    VelocityX,
    /// Vertical velocity of the same, positive downward.
    VelocityY,
}

impl Channel {
    /// The name used in the handshake.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::NormalEnemy => "normal-enemy",
            Self::Kamikaze => "kamikaze",
            Self::Blast => "blast",
            Self::BlastPhase => "blast-phase",
            Self::Player => "player",
            Self::VelocityX => "velocity-x",
            Self::VelocityY => "velocity-y",
        }
    }
}

/// Which hazard owns a cell when several cover it.
///
/// A lethal thing must never hide behind a harmless one, and two hazards of the
/// same kind are separated by size and then identity, so a contested cell
/// resolves the same way on every run of a seed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum HazardRank {
    Normal,
    Kamikaze,
    Blast,
}

/// One hazard, converted into window pixels.
#[derive(Debug, Clone, Copy)]
struct Painted {
    rank: HazardRank,
    /// Sort key, lowest first, so ties break identically every run.
    identity: u64,
    area: f32,
    centre: Vec2,
    half: Vec2,
    velocity: Vec2,
    /// `None` unless the hazard is a blast.
    phase: Option<f32>,
}

/// A section of the flat observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Section {
    pub offset: usize,
    pub length: usize,
}

/// Everything a reader needs to interpret an observation.
///
/// Sent once, in the handshake. A policy checkpoint stores it too, so a model
/// trained against one layout cannot silently be fed another.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Layout {
    /// Bumped whenever the meaning of any value changes.
    pub version: u32,
    pub dtype: String,
    pub grid: usize,
    pub cell_pixels: u32,
    pub window_pixels: u32,
    /// World units per reference pixel, for anyone converting back.
    pub world_units_per_pixel: f32,
    pub velocity_scale: f32,
    /// Positive Y points down, as it does on a screen.
    pub y_axis: String,
    pub channels: Vec<String>,
    pub actions: Vec<String>,
    pub horizons: Vec<u32>,
    pub hold_frames: u32,
    /// Path samples are divided by this, so they match the window's half width.
    pub path_scale: f32,
    pub player_section: Section,
    pub grid_section: Section,
    pub path_section: Section,
    pub observation_values: usize,
}

/// The layout this build produces.
#[must_use]
pub fn layout(hold_frames: u32) -> Layout {
    let grid_values = CHANNELS.len().saturating_mul(CELLS);
    Layout {
        version: 1,
        dtype: "f32".to_owned(),
        grid: GRID,
        cell_pixels: 4,
        window_pixels: 256,
        world_units_per_pixel: PIXEL,
        velocity_scale: VELOCITY_SCALE,
        y_axis: "down".to_owned(),
        channels: CHANNELS
            .iter()
            .map(|channel| channel.name().to_owned())
            .collect(),
        actions: Action::ALL
            .iter()
            .map(|action| action.name().to_owned())
            .collect(),
        horizons: SAMPLE_FRAMES.to_vec(),
        hold_frames,
        path_scale: WINDOW_HALF,
        player_section: Section {
            offset: 0,
            length: PLAYER_VALUES,
        },
        grid_section: Section {
            offset: PLAYER_VALUES,
            length: grid_values,
        },
        path_section: Section {
            offset: PLAYER_VALUES.saturating_add(grid_values),
            length: PATH_VALUES,
        },
        observation_values: OBSERVATION_VALUES,
    }
}

/// The furthest a predicted path can travel, in reference pixels.
///
/// Momentum at top speed plus a full hold plus the coast after it. Compared
/// against the window's half width, this says whether the far samples can
/// leave the observed window.
#[must_use]
pub fn max_path_pixels(hold_frames: u32) -> f32 {
    let start = Vec2::new(crate::motion::top_speed(), 0.0);
    predict_path(Vec2::ZERO, start, Vec2::X, hold_frames).map_or(0.0, |path| {
        path.iter()
            .map(|point| point.length() / PIXEL)
            .fold(0.0_f32, f32::max)
    })
}

/// Fill `buffer` with one observation of `view`.
///
/// # Errors
///
/// [`ArenaError::NonFinite`] if the buffer is the wrong length or the view
/// carries a value that is not finite.
pub fn encode(view: &ArenaView, hold_frames: u32, buffer: &mut [f32]) -> Result<(), ArenaError> {
    if buffer.len() != OBSERVATION_VALUES {
        return Err(ArenaError::NonFinite(
            "an observation buffer must be exactly the layout's length",
        ));
    }
    buffer.fill(0.0);
    let Some(player) = view.player else {
        // No player is no observation: an arena in this state has nothing to
        // centre a window on.
        return Ok(());
    };
    if !player.position.is_finite() || !player.velocity.is_finite() {
        return Err(ArenaError::NonFinite("the player's state must be finite"));
    }

    write_player(&player, buffer);
    write_grid(view, &player, buffer);
    write_paths(&player, hold_frames, buffer)
}

/// The player's own velocity, which the critic and the senses both read.
fn write_player(player: &PlayerView, buffer: &mut [f32]) {
    let velocity = to_pixels_per_frame(player.velocity);
    if let Some(slot) = buffer.first_mut() {
        *slot = clamp_unit(velocity.x / VELOCITY_SCALE);
    }
    if let Some(slot) = buffer.get_mut(1) {
        *slot = clamp_unit(velocity.y / VELOCITY_SCALE);
    }
}

/// Paint every hazard, then the player, into the grid channels.
fn write_grid(view: &ArenaView, player: &PlayerView, buffer: &mut [f32]) {
    let mut hazards: Vec<Painted> =
        Vec::with_capacity(view.enemies.len().saturating_add(view.blasts.len()));
    for enemy in &view.enemies {
        if let Some(painted) = paint_enemy(enemy, player, view.half_extents) {
            hazards.push(painted);
        }
    }
    for blast in &view.blasts {
        if let Some(painted) = paint_blast(blast, player, view.half_extents) {
            hazards.push(painted);
        }
    }
    // Lowest priority first, so a later write wins the cell. Ordering by rank,
    // then area, then identity makes the winner independent of query order.
    hazards.sort_unstable_by(|left, right| {
        left.rank
            .cmp(&right.rank)
            .then(left.area.total_cmp(&right.area))
            .then(left.identity.cmp(&right.identity))
    });

    for hazard in &hazards {
        for (column, row) in covered_cells(hazard.centre, hazard.half) {
            clear_hazard_channels(buffer, column, row);
            let channel = match hazard.rank {
                HazardRank::Normal => 0,
                HazardRank::Kamikaze => 1,
                HazardRank::Blast => 2,
            };
            write_cell(buffer, channel, column, row, 1.0);
            if let Some(phase) = hazard.phase {
                write_cell(buffer, 3, column, row, phase);
            }
            let velocity = to_pixels_per_frame(hazard.velocity);
            write_cell(
                buffer,
                5,
                column,
                row,
                clamp_unit(velocity.x / VELOCITY_SCALE),
            );
            write_cell(
                buffer,
                6,
                column,
                row,
                clamp_unit(velocity.y / VELOCITY_SCALE),
            );
        }
    }

    // The player is its own channel, so it never contests a hazard's cell.
    // Half extents are a size, so the Y flip must not leave one negative --
    // a negative half extent describes an empty rectangle and paints nothing.
    let half = to_window_pixels(player.collider.half_extents).abs();
    let centre = to_window_pixels(player.collider.offset);
    for (column, row) in covered_cells(Vec2::new(WINDOW_HALF, WINDOW_HALF) + centre, half) {
        write_cell(buffer, 4, column, row, 1.0);
    }
}

/// Where each action takes the player, as offsets from the player.
fn write_paths(
    player: &PlayerView,
    hold_frames: u32,
    buffer: &mut [f32],
) -> Result<(), ArenaError> {
    let base = PLAYER_VALUES.saturating_add(CHANNELS.len().saturating_mul(CELLS));
    for (index, action) in Action::ALL.iter().enumerate() {
        let path = predict_path(
            player.position,
            player.velocity,
            action.direction(),
            hold_frames,
        )?;
        for (sample, point) in path.iter().enumerate() {
            // Relative to the player and never clamped: a sample outside the
            // window is a real answer, and the reader samples the field with
            // border padding rather than pretending the path stopped.
            let offset = to_window_pixels(wrapped_delta(player.position, *point, half_world()));
            let at = base
                .saturating_add(index.saturating_mul(SAMPLE_COUNT).saturating_mul(2))
                .saturating_add(sample.saturating_mul(2));
            if let Some(slot) = buffer.get_mut(at) {
                *slot = offset.x / WINDOW_HALF;
            }
            if let Some(slot) = buffer.get_mut(at.saturating_add(1)) {
                *slot = offset.y / WINDOW_HALF;
            }
        }
    }
    Ok(())
}

const fn half_world() -> Vec2 {
    crate::scale::WORLD_HALF_EXTENTS
}

/// World units to reference pixels, with the screen's downward Y.
fn to_window_pixels(offset: Vec2) -> Vec2 {
    Vec2::new(offset.x / PIXEL, -offset.y / PIXEL)
}

/// World units per second to reference pixels per frame, Y flipped.
fn to_pixels_per_frame(velocity: Vec2) -> Vec2 {
    let per_frame = velocity / (PIXEL * 60.0);
    Vec2::new(per_frame.x, -per_frame.y)
}

const fn clamp_unit(value: f32) -> f32 {
    value.clamp(-1.0, 1.0)
}

fn paint_enemy(enemy: &EnemyView, player: &PlayerView, half_extents: Vec2) -> Option<Painted> {
    if !enemy.collider.enabled {
        return None;
    }
    let rank = match enemy.kind {
        crate::enemy_types::EnemyKind::Normal => HazardRank::Normal,
        crate::enemy_types::EnemyKind::Kamikaze => HazardRank::Kamikaze,
    };
    painted(
        rank,
        enemy.entity.to_bits(),
        enemy.position + enemy.collider.offset,
        enemy.collider.half_extents,
        enemy.velocity,
        None,
        player,
        half_extents,
    )
}

fn paint_blast(blast: &BlastView, player: &PlayerView, half_extents: Vec2) -> Option<Painted> {
    if !blast.collider.enabled || blast.duration <= 0.0 {
        return None;
    }
    let progress = (blast.age / blast.duration).clamp(0.0, 1.0);
    let phase = 2.0_f32.mul_add(-progress, 1.0).clamp(-1.0, 1.0);
    painted(
        HazardRank::Blast,
        blast.entity.to_bits(),
        blast.position + blast.collider.offset,
        blast.collider.half_extents,
        // A blast does not travel; only its size changes.
        Vec2::ZERO,
        Some(phase),
        player,
        half_extents,
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "One hazard's conversion, named at each call site"
)]
fn painted(
    rank: HazardRank,
    identity: u64,
    position: Vec2,
    half_extents: Vec2,
    velocity: Vec2,
    phase: Option<f32>,
    player: &PlayerView,
    half_extents_world: Vec2,
) -> Option<Painted> {
    if !position.is_finite() || !half_extents.is_finite() || !velocity.is_finite() {
        return None;
    }
    // The nearest wrapped copy: a threat just across a seam is close, however
    // far apart the two coordinates look.
    let delta = wrapped_delta(player.position, position, half_extents_world);
    let centre = Vec2::new(WINDOW_HALF, WINDOW_HALF) + to_window_pixels(delta);
    let half = to_window_pixels(half_extents).abs();
    Some(Painted {
        rank,
        identity,
        area: half.x * half.y,
        centre,
        half,
        velocity,
        phase,
    })
}

/// Every cell a rectangle covers with positive area, clipped to the window.
///
/// A hazard entirely outside the window paints nothing; it is not dragged onto
/// an edge cell, because the window is local by design and a threat beyond it
/// is simply unobserved.
fn covered_cells(centre: Vec2, half: Vec2) -> Vec<(usize, usize)> {
    let min = centre - half;
    let max = centre + half;
    if max.x <= 0.0 || max.y <= 0.0 || min.x >= WINDOW_PIXELS || min.y >= WINDOW_PIXELS {
        return Vec::new();
    }
    let first_column = cell_index(min.x);
    let last_column = cell_index_end(max.x);
    let first_row = cell_index(min.y);
    let last_row = cell_index_end(max.y);
    let mut cells = Vec::new();
    for row in first_row..=last_row {
        for column in first_column..=last_column {
            cells.push((column, row));
        }
    }
    cells
}

/// The cell a coordinate falls in, clipped into the window.
///
/// The one place this module turns a float into an index. The value is clamped
/// into `[0, GRID)` before the conversion, so the cast cannot lose a digit or
/// wrap: everything outside the window has already been rejected by
/// [`covered_cells`].
#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "Clamped into [0, GRID) before converting, so no digit or sign can be lost"
)]
fn cell_index(pixels: f32) -> usize {
    let scaled = (pixels.max(0.0) / CELL_PIXELS).floor();
    let bounded = scaled.clamp(0.0, GRID_EDGE - 1.0);
    bounded as usize
}

/// The last cell a rectangle's far edge touches.
///
/// The edge is exclusive: a rectangle ending exactly on a cell boundary does
/// not paint the cell beyond it, because it covers none of it.
///
/// `next_down` rather than subtracting `f32::EPSILON`: near the middle of the
/// window the gap between neighbouring floats is far wider than epsilon, so
/// the subtraction would round back to the same number and paint a cell the
/// rectangle only touches.
fn cell_index_end(pixels: f32) -> usize {
    cell_index(pixels.next_down().max(0.0))
}

const fn cell_offset(channel: usize, column: usize, row: usize) -> usize {
    PLAYER_VALUES
        .saturating_add(channel.saturating_mul(CELLS))
        .saturating_add(row.saturating_mul(GRID))
        .saturating_add(column)
}

fn write_cell(buffer: &mut [f32], channel: usize, column: usize, row: usize, value: f32) {
    if let Some(slot) = buffer.get_mut(cell_offset(channel, column, row)) {
        *slot = value;
    }
}

/// Clear the channels one hazard owns, so a cell never describes two.
fn clear_hazard_channels(buffer: &mut [f32], column: usize, row: usize) {
    for channel in [0, 1, 2, 3, 5, 6] {
        write_cell(buffer, channel, column, row, 0.0);
    }
}

#[cfg(test)]
mod tests;
