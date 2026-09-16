//! Tests for the observation encoder.
//!
//! The numbers here are the contract Python decodes against, so most of these
//! check a value rather than a shape.

#![expect(
    clippy::float_cmp,
    clippy::suboptimal_flops,
    clippy::as_conversions,
    reason = "Exact comparisons against values the encoder writes verbatim, and index arithmetic over the nine actions"
)]

use crate::collision::Collider;
use crate::enemy_types::EnemyKind;
use crate::observation::*;
use crate::scale::{PIXEL, WORLD_HALF_EXTENTS};
use crate::simulation::{
    Action, ArenaError, ArenaView, BlastView, DEFAULT_HOLD_FRAMES, EnemyView, PlayerView,
    SAMPLE_COUNT, SAMPLE_FRAMES,
};
use bevy::ecs::entity::Entity;
use bevy::math::Vec2;

fn player_at(position: Vec2, velocity: Vec2) -> PlayerView {
    PlayerView {
        entity: Entity::from_raw_u32(1).expect("a valid entity"),
        position,
        velocity,
        collider: Collider::rectangle(Vec2::splat(12.0)),
    }
}

fn view_of(player: PlayerView, enemies: Vec<EnemyView>, blasts: Vec<BlastView>) -> ArenaView {
    ArenaView {
        frame: 0,
        player: Some(player),
        enemies,
        blasts,
        half_extents: WORLD_HALF_EXTENTS,
    }
}

fn enemy_at(id: u32, kind: EnemyKind, position: Vec2, half: f32, velocity: Vec2) -> EnemyView {
    EnemyView {
        entity: Entity::from_raw_u32(id).expect("a valid entity"),
        kind,
        position,
        velocity,
        collider: Collider::rectangle(Vec2::splat(half)),
    }
}

fn blast_at(id: u32, position: Vec2, half: f32, age: f32, duration: f32) -> BlastView {
    BlastView {
        entity: Entity::from_raw_u32(id).expect("a valid entity"),
        position,
        collider: Collider::rectangle(Vec2::splat(half)),
        age,
        duration,
    }
}

fn encoded(view: &ArenaView) -> Vec<f32> {
    let mut buffer = vec![0.0; OBSERVATION_VALUES];
    encode(view, DEFAULT_HOLD_FRAMES, &mut buffer).expect("a finite view encodes");
    buffer
}

fn channel_of(buffer: &[f32], channel: usize) -> &[f32] {
    let start = PLAYER_VALUES + channel * CELLS;
    buffer.get(start..start + CELLS).expect("a whole channel")
}

fn cell(buffer: &[f32], channel: usize, column: usize, row: usize) -> f32 {
    *channel_of(buffer, channel)
        .get(row * GRID + column)
        .expect("a cell inside the grid")
}

/// Cells holding a non-zero value in one channel.
fn occupied(buffer: &[f32], channel: usize) -> Vec<(usize, usize)> {
    channel_of(buffer, channel)
        .iter()
        .enumerate()
        .filter(|(_, value)| **value != 0.0)
        .map(|(index, _)| (index % GRID, index / GRID))
        .collect()
}

/// One reference pixel, in world units.
const PX: f32 = PIXEL;

#[test]
fn the_layout_describes_the_buffer_it_produces() {
    let layout = layout(DEFAULT_HOLD_FRAMES);
    assert_eq!(layout.observation_values, 28_782, "the design's total");
    assert_eq!(layout.observation_values, OBSERVATION_VALUES);
    assert_eq!(layout.grid, GRID);
    assert_eq!(layout.channels.len(), 7);
    assert_eq!(layout.actions.len(), 9);
    assert_eq!(layout.horizons, SAMPLE_FRAMES.to_vec());
    assert_eq!(layout.hold_frames, DEFAULT_HOLD_FRAMES);

    assert_eq!(layout.player_section.offset, 0);
    assert_eq!(layout.player_section.length, 2);
    assert_eq!(layout.grid_section.offset, 2);
    assert_eq!(layout.grid_section.length, 7 * CELLS);
    assert_eq!(layout.path_section.offset, 2 + 7 * CELLS);
    assert_eq!(layout.path_section.length, 108);
    assert_eq!(
        layout.path_section.offset + layout.path_section.length,
        layout.observation_values,
        "the sections tile the buffer exactly"
    );
}

#[test]
fn the_grid_constants_agree_with_each_other() {
    assert_eq!(CELLS, GRID * GRID);
    assert!((WINDOW_PIXELS - 256.0).abs() < f32::EPSILON);
    assert!((WINDOW_HALF - 128.0).abs() < f32::EPSILON);
}

#[test]
fn a_buffer_of_the_wrong_length_is_refused() {
    let view = view_of(player_at(Vec2::ZERO, Vec2::ZERO), Vec::new(), Vec::new());
    let mut short = vec![0.0; OBSERVATION_VALUES - 1];
    assert!(matches!(
        encode(&view, DEFAULT_HOLD_FRAMES, &mut short),
        Err(ArenaError::NonFinite(_))
    ));
}

#[test]
fn the_player_sits_at_the_centre_of_its_own_window() {
    let buffer = encoded(&view_of(
        player_at(Vec2::new(1_234.0, -567.0), Vec2::ZERO),
        Vec::new(),
        Vec::new(),
    ));
    let cells = occupied(&buffer, 4);
    assert!(!cells.is_empty(), "the player paints its own channel");
    let columns: Vec<usize> = cells.iter().map(|(column, _)| *column).collect();
    let rows: Vec<usize> = cells.iter().map(|(_, row)| *row).collect();
    // A 12-unit half extent is under two pixels, so the hitbox straddles the
    // centre corner shared by cells 31 and 32.
    assert!(columns.iter().all(|column| (31..=32).contains(column)));
    assert!(rows.iter().all(|row| (31..=32).contains(row)));
}

#[test]
fn player_velocity_is_reference_pixels_a_frame_over_the_scale() {
    // Top speed is 2.5 reference pixels a frame, which is 0.625 after scaling.
    let buffer = encoded(&view_of(
        player_at(Vec2::ZERO, Vec2::new(crate::motion::top_speed(), 0.0)),
        Vec::new(),
        Vec::new(),
    ));
    let x = *buffer.first().expect("the first value");
    assert!((x - 0.625).abs() < 0.001, "expected 0.625, found {x}");
}

#[test]
fn upward_world_velocity_is_negative_in_the_observation() {
    let buffer = encoded(&view_of(
        player_at(Vec2::ZERO, Vec2::new(0.0, crate::motion::top_speed())),
        Vec::new(),
        Vec::new(),
    ));
    let y = *buffer.get(1).expect("the second value");
    assert!(
        y < 0.0,
        "the screen's Y points down, so moving up reads negative: {y}"
    );
}

#[test]
fn an_enemy_to_the_right_paints_to_the_right_of_centre() {
    let buffer = encoded(&view_of(
        player_at(Vec2::ZERO, Vec2::ZERO),
        // Twenty reference pixels right of the player.
        vec![enemy_at(
            2,
            EnemyKind::Normal,
            Vec2::new(20.0 * PX, 0.0),
            2.0 * PX,
            Vec2::ZERO,
        )],
        Vec::new(),
    ));
    let cells = occupied(&buffer, 0);
    assert!(!cells.is_empty(), "the enemy is painted");
    // Twenty pixels is five cells right of the centre corner at 32.
    assert!(
        cells.iter().all(|(column, _)| *column >= 33),
        "an enemy to the right lands right of centre: {cells:?}"
    );
    assert!(cells.iter().all(|(_, row)| (31..=32).contains(row)));
}

#[test]
fn an_enemy_above_the_player_paints_above_centre() {
    let buffer = encoded(&view_of(
        player_at(Vec2::ZERO, Vec2::ZERO),
        vec![enemy_at(
            2,
            EnemyKind::Normal,
            // Up in world axes is a smaller row in the observation.
            Vec2::new(0.0, 20.0 * PX),
            2.0 * PX,
            Vec2::ZERO,
        )],
        Vec::new(),
    ));
    let cells = occupied(&buffer, 0);
    assert!(
        cells.iter().all(|(_, row)| *row <= 30),
        "up in the world is up the screen: {cells:?}"
    );
}

#[test]
fn a_hazard_across_the_seam_is_near_not_a_world_away() {
    let half = WORLD_HALF_EXTENTS;
    // The player hugs the right edge; the enemy is just past it, which is a
    // whole world away by raw subtraction and eight pixels away in truth.
    let player = player_at(Vec2::new(half.x - 4.0 * PX, 0.0), Vec2::ZERO);
    let enemy = enemy_at(
        2,
        EnemyKind::Normal,
        Vec2::new(-half.x + 4.0 * PX, 0.0),
        2.0 * PX,
        Vec2::ZERO,
    );
    let buffer = encoded(&view_of(player, vec![enemy], Vec::new()));

    let cells = occupied(&buffer, 0);
    assert!(!cells.is_empty(), "a threat across the seam is observed");
    assert!(
        cells.iter().all(|(column, _)| (33..=40).contains(column)),
        "and lands just right of the player: {cells:?}"
    );
}

#[test]
fn a_hazard_outside_the_window_is_unobserved_rather_than_clamped_to_the_edge() {
    let buffer = encoded(&view_of(
        player_at(Vec2::ZERO, Vec2::ZERO),
        vec![enemy_at(
            2,
            EnemyKind::Normal,
            // Two hundred pixels away: outside the 128-pixel half window.
            Vec2::new(200.0 * PX, 0.0),
            2.0 * PX,
            Vec2::ZERO,
        )],
        Vec::new(),
    ));
    assert!(
        occupied(&buffer, 0).is_empty(),
        "the window is local; it does not invent an edge threat"
    );
}

#[test]
fn a_hazard_straddling_the_window_edge_paints_only_its_visible_part() {
    let buffer = encoded(&view_of(
        player_at(Vec2::ZERO, Vec2::ZERO),
        vec![enemy_at(
            2,
            EnemyKind::Normal,
            Vec2::new(126.0 * PX, 0.0),
            4.0 * PX,
            Vec2::ZERO,
        )],
        Vec::new(),
    ));
    let cells = occupied(&buffer, 0);
    assert!(!cells.is_empty(), "the visible part is painted");
    assert!(
        cells.iter().all(|(column, _)| *column < GRID),
        "and nothing spills outside the grid: {cells:?}"
    );
    assert!(
        cells.iter().any(|(column, _)| *column == GRID - 1),
        "including the last column: {cells:?}"
    );
}

#[test]
fn a_rectangle_ending_on_a_cell_boundary_does_not_paint_the_next_cell() {
    // Exactly four pixels wide, starting on a boundary: it covers one column.
    let buffer = encoded(&view_of(
        player_at(Vec2::ZERO, Vec2::ZERO),
        vec![enemy_at(
            2,
            EnemyKind::Normal,
            Vec2::new(2.0 * PX, 2.0 * PX),
            2.0 * PX,
            Vec2::ZERO,
        )],
        Vec::new(),
    ));
    let cells = occupied(&buffer, 0);
    assert_eq!(cells.len(), 1, "one cell, not four: {cells:?}");
}

#[test]
fn a_disabled_collider_is_not_a_hazard() {
    let mut enemy = enemy_at(2, EnemyKind::Normal, Vec2::ZERO, 4.0 * PX, Vec2::ZERO);
    enemy.collider.enabled = false;
    let buffer = encoded(&view_of(
        player_at(Vec2::ZERO, Vec2::ZERO),
        vec![enemy],
        Vec::new(),
    ));
    assert!(occupied(&buffer, 0).is_empty());
}

#[test]
fn a_blast_outranks_a_kamikaze_which_outranks_a_normal_enemy() {
    let at_centre = Vec2::new(40.0 * PX, 0.0);
    let buffer = encoded(&view_of(
        player_at(Vec2::ZERO, Vec2::ZERO),
        vec![
            enemy_at(2, EnemyKind::Normal, at_centre, 4.0 * PX, Vec2::ZERO),
            enemy_at(3, EnemyKind::Kamikaze, at_centre, 4.0 * PX, Vec2::ZERO),
        ],
        vec![blast_at(4, at_centre, 4.0 * PX, 0.1, 1.0)],
    ));
    let blasts = occupied(&buffer, 2);
    assert!(!blasts.is_empty(), "the blast owns the cells");
    for (column, row) in blasts {
        assert_eq!(cell(&buffer, 0, column, row), 0.0, "no normal enemy left");
        assert_eq!(cell(&buffer, 1, column, row), 0.0, "no kamikaze left");
    }
}

#[test]
fn two_hazards_of_one_kind_break_their_tie_the_same_way_every_time() {
    let at_centre = Vec2::new(40.0 * PX, 0.0);
    let small = enemy_at(
        9,
        EnemyKind::Normal,
        at_centre,
        4.0 * PX,
        Vec2::new(100.0, 0.0),
    );
    let large = enemy_at(
        3,
        EnemyKind::Normal,
        at_centre,
        6.0 * PX,
        Vec2::new(-100.0, 0.0),
    );

    let one = encoded(&view_of(
        player_at(Vec2::ZERO, Vec2::ZERO),
        vec![small, large],
        Vec::new(),
    ));
    let other = encoded(&view_of(
        player_at(Vec2::ZERO, Vec2::ZERO),
        // The same hazards, offered in the other order.
        vec![large, small],
        Vec::new(),
    ));
    assert_eq!(one, other, "query order cannot change an observation");

    // The larger hazard wins the contested cell, so its velocity is the one
    // the cell carries.
    let (column, row) = *occupied(&one, 0).first().expect("painted cells");
    assert!(
        cell(&one, 5, column, row) < 0.0,
        "the winner supplies the velocity"
    );
}

#[test]
fn a_cell_never_mixes_two_hazards() {
    let centre = Vec2::new(40.0 * PX, 0.0);
    let buffer = encoded(&view_of(
        player_at(Vec2::ZERO, Vec2::ZERO),
        vec![enemy_at(
            2,
            EnemyKind::Normal,
            centre,
            4.0 * PX,
            Vec2::new(200.0, 0.0),
        )],
        vec![blast_at(3, centre, 4.0 * PX, 0.1, 1.0)],
    ));
    for (column, row) in occupied(&buffer, 2) {
        assert_eq!(
            cell(&buffer, 0, column, row),
            0.0,
            "the overwritten enemy leaves no occupancy behind"
        );
        assert_eq!(
            cell(&buffer, 5, column, row),
            0.0,
            "nor its velocity: a blast does not travel"
        );
    }
}

#[test]
fn blast_phase_separates_a_growing_blast_from_a_shrinking_one() {
    let centre = Vec2::new(40.0 * PX, 0.0);
    let growing = encoded(&view_of(
        player_at(Vec2::ZERO, Vec2::ZERO),
        Vec::new(),
        vec![blast_at(3, centre, 4.0 * PX, 0.1, 1.0)],
    ));
    let shrinking = encoded(&view_of(
        player_at(Vec2::ZERO, Vec2::ZERO),
        Vec::new(),
        // The same width, later in its life.
        vec![blast_at(3, centre, 4.0 * PX, 0.9, 1.0)],
    ));

    let (column, row) = *occupied(&growing, 2).first().expect("painted cells");
    let early = cell(&growing, 3, column, row);
    let late = cell(&shrinking, 3, column, row);
    assert!(early > 0.0, "a young blast is still growing: {early}");
    assert!(late < 0.0, "an old one is nearly gone: {late}");
    assert!((early - 0.8).abs() < 0.001, "phase is 1 - 2t: {early}");
    assert!((late + 0.8).abs() < 0.001, "and symmetric: {late}");
}

#[test]
fn blast_phase_stays_inside_its_range_at_the_boundaries() {
    for (age, expected) in [(0.0, 1.0), (0.5, 0.0), (1.0, -1.0), (2.0, -1.0)] {
        let buffer = encoded(&view_of(
            player_at(Vec2::ZERO, Vec2::ZERO),
            Vec::new(),
            vec![blast_at(3, Vec2::new(40.0 * PX, 0.0), 4.0 * PX, age, 1.0)],
        ));
        let (column, row) = *occupied(&buffer, 2).first().expect("painted cells");
        let phase = cell(&buffer, 3, column, row);
        assert!(
            (phase - expected).abs() < 0.001,
            "age {age} should read {expected}, found {phase}"
        );
    }
}

#[test]
fn hazard_velocity_is_scaled_pixels_a_frame_with_the_screens_y() {
    // One reference pixel a frame, upward in world axes.
    let speed = PX * 60.0;
    let buffer = encoded(&view_of(
        player_at(Vec2::ZERO, Vec2::ZERO),
        vec![enemy_at(
            2,
            EnemyKind::Normal,
            Vec2::new(40.0 * PX, 0.0),
            4.0 * PX,
            Vec2::new(0.0, speed),
        )],
        Vec::new(),
    ));
    let (column, row) = *occupied(&buffer, 0).first().expect("painted cells");
    let vy = cell(&buffer, 6, column, row);
    assert!(
        (vy + 0.25).abs() < 0.001,
        "one pixel a frame upward is -1/4 after scaling: {vy}"
    );
}

#[test]
fn a_velocity_past_the_scale_is_clipped_rather_than_wrapped() {
    let buffer = encoded(&view_of(
        player_at(Vec2::ZERO, Vec2::ZERO),
        vec![enemy_at(
            2,
            EnemyKind::Normal,
            Vec2::new(40.0 * PX, 0.0),
            4.0 * PX,
            Vec2::new(PX * 60.0 * 100.0, 0.0),
        )],
        Vec::new(),
    ));
    let (column, row) = *occupied(&buffer, 0).first().expect("painted cells");
    assert!((cell(&buffer, 5, column, row) - 1.0).abs() < f32::EPSILON);
}

#[test]
fn paths_are_player_relative_offsets_divided_by_the_window_half() {
    let view = view_of(
        player_at(Vec2::new(500.0, -250.0), Vec2::ZERO),
        Vec::new(),
        Vec::new(),
    );
    let buffer = encoded(&view);
    let layout = layout(DEFAULT_HOLD_FRAMES);
    let base = layout.path_section.offset;

    // Idle from rest goes nowhere at all.
    for sample in 0..SAMPLE_COUNT {
        let x = *buffer.get(base + sample * 2).expect("a path value");
        let y = *buffer.get(base + sample * 2 + 1).expect("a path value");
        assert!(x.abs() < 1e-6 && y.abs() < 1e-6, "idle from rest is still");
    }

    // Right is positive x; up is negative y.
    let right = Action::Right.index() as usize;
    let up = Action::Up.index() as usize;
    let last = SAMPLE_COUNT - 1;
    let right_x = *buffer
        .get(base + right * SAMPLE_COUNT * 2 + last * 2)
        .expect("a path value");
    let up_y = *buffer
        .get(base + up * SAMPLE_COUNT * 2 + last * 2 + 1)
        .expect("a path value");
    assert!(right_x > 0.0, "right moves right: {right_x}");
    assert!(up_y < 0.0, "up moves up the screen: {up_y}");
}

#[test]
fn paths_carry_the_players_momentum() {
    let still = encoded(&view_of(
        player_at(Vec2::ZERO, Vec2::ZERO),
        Vec::new(),
        Vec::new(),
    ));
    let moving = encoded(&view_of(
        player_at(Vec2::ZERO, Vec2::new(crate::motion::top_speed(), 0.0)),
        Vec::new(),
        Vec::new(),
    ));
    let base = layout(DEFAULT_HOLD_FRAMES).path_section.offset;
    let idle_x = |buffer: &[f32]| *buffer.get(base).expect("a path value");
    assert!(
        idle_x(&moving) > idle_x(&still),
        "a moving player's idle path leaves the spot it is standing on"
    );
}

#[test]
fn a_path_leaving_the_window_is_reported_rather_than_clamped() {
    // A very long hold sends the far samples past the window's half width.
    let mut buffer = vec![0.0; OBSERVATION_VALUES];
    let view = view_of(player_at(Vec2::ZERO, Vec2::ZERO), Vec::new(), Vec::new());
    encode(&view, 108, &mut buffer).expect("a finite view encodes");
    let base = layout(108).path_section.offset;
    let right = Action::Right.index() as usize;
    let last = SAMPLE_COUNT - 1;
    let value = *buffer
        .get(base + right * SAMPLE_COUNT * 2 + last * 2)
        .expect("a path value");
    assert!(
        value > 1.0,
        "a sample outside the window reads past one: {value}"
    );
}

#[test]
fn the_reachable_distance_warns_about_the_window_it_is_measured_against() {
    let held_briefly = max_path_pixels(DEFAULT_HOLD_FRAMES);
    let held_throughout = max_path_pixels(108);
    assert!(
        held_briefly < WINDOW_HALF,
        "the default hold stays inside the window: {held_briefly}"
    );
    assert!(
        held_throughout > WINDOW_HALF,
        "holding throughout does not: {held_throughout}"
    );
}

#[test]
fn an_arena_without_a_player_encodes_zeroes_rather_than_failing() {
    let view = ArenaView {
        frame: 0,
        player: None,
        enemies: Vec::new(),
        blasts: Vec::new(),
        half_extents: WORLD_HALF_EXTENTS,
    };
    let buffer = encoded(&view);
    assert!(buffer.iter().all(|value| *value == 0.0));
}

#[test]
fn a_non_finite_player_is_refused() {
    let view = view_of(
        player_at(Vec2::new(f32::NAN, 0.0), Vec2::ZERO),
        Vec::new(),
        Vec::new(),
    );
    let mut buffer = vec![0.0; OBSERVATION_VALUES];
    assert!(matches!(
        encode(&view, DEFAULT_HOLD_FRAMES, &mut buffer),
        Err(ArenaError::NonFinite(_))
    ));
}

#[test]
fn encoding_is_deterministic_for_one_view() {
    let view = view_of(
        player_at(Vec2::new(12.0, -34.0), Vec2::new(100.0, -50.0)),
        vec![
            enemy_at(
                2,
                EnemyKind::Normal,
                Vec2::new(30.0 * PX, 10.0 * PX),
                3.0 * PX,
                Vec2::new(50.0, 0.0),
            ),
            enemy_at(
                3,
                EnemyKind::Kamikaze,
                Vec2::new(-20.0 * PX, -5.0 * PX),
                4.0 * PX,
                Vec2::new(0.0, -60.0),
            ),
        ],
        vec![blast_at(4, Vec2::new(0.0, 60.0 * PX), 8.0 * PX, 0.3, 1.0)],
    );
    assert_eq!(encoded(&view), encoded(&view));
}

/// Which of two entities the tie-break prefers.
///
/// `Entity::to_bits` stores the index inverted, so a lower index is a *higher*
/// bits value. Deriving the expectation from the bits keeps these tests honest
/// about the rule rather than about Bevy's packing.
fn lower_bits(left: Entity, right: Entity) -> Entity {
    if left.to_bits() <= right.to_bits() {
        left
    } else {
        right
    }
}

#[test]
fn hazards_of_equal_size_give_the_cell_to_the_lower_entity() {
    let at_centre = Vec2::new(40.0 * PX, 0.0);
    // Same kind, same footprint: only identity can separate them, and the
    // velocity each carries says which one won.
    let left = enemy_at(
        3,
        EnemyKind::Normal,
        at_centre,
        4.0 * PX,
        Vec2::new(-120.0, 0.0),
    );
    let right = enemy_at(
        9,
        EnemyKind::Normal,
        at_centre,
        4.0 * PX,
        Vec2::new(120.0, 0.0),
    );
    let winner = lower_bits(left.entity, right.entity);
    let expected_sign = if winner == left.entity { -1.0 } else { 1.0 };

    for order in [vec![left, right], vec![right, left]] {
        let buffer = encoded(&view_of(
            player_at(Vec2::ZERO, Vec2::ZERO),
            order,
            Vec::new(),
        ));
        let (column, row) = *occupied(&buffer, 0).first().expect("painted cells");
        let velocity = cell(&buffer, 5, column, row);
        assert!(
            velocity * expected_sign > 0.0,
            "the lowest entity bits win the cell, whichever order they arrive in: {velocity}"
        );
    }
}

#[test]
fn blasts_of_equal_size_and_different_phase_resolve_by_entity() {
    let at_centre = Vec2::new(40.0 * PX, 0.0);
    let growing = blast_at(2, at_centre, 4.0 * PX, 0.1, 1.0);
    let shrinking = blast_at(7, at_centre, 4.0 * PX, 0.9, 1.0);
    let winner = lower_bits(growing.entity, shrinking.entity);
    // Phase is positive while growing and negative while shrinking, so the
    // winner's sign says which blast the cell describes.
    let expected_sign = if winner == growing.entity { 1.0 } else { -1.0 };

    for order in [vec![growing, shrinking], vec![shrinking, growing]] {
        let buffer = encoded(&view_of(
            player_at(Vec2::ZERO, Vec2::ZERO),
            Vec::new(),
            order,
        ));
        let (column, row) = *occupied(&buffer, 2).first().expect("painted cells");
        let phase = cell(&buffer, 3, column, row);
        assert!(
            phase * expected_sign > 0.0,
            "one blast owns the cell, and it is the same one either way: {phase}"
        );
    }
}
