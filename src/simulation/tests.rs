//! Tests for the shared simulation and the headless arena.

use crate::collision::Collider;
use crate::enemy::{BoundaryMode, Enemy, EnemySettings, EnemyWorld, Velocity2d};
use crate::enemy_types::{EnemyKind, KamikazeBlast};
use crate::simulation::*;
use bevy::time::TimeUpdateStrategy;

/// A simulation app with no enemies, stepped one fixed frame at a time.
///
/// The clock is manual: `TimePlugin` otherwise overwrites the delta with
/// real elapsed time, whose first frame is near zero, and a test that
/// depends on the wall clock is a test that fails on a slow machine.
fn app() -> App {
    let mut app = App::new();
    app.add_plugins((MinimalPlugins, SimulationPlugin))
        .insert_resource(TimeUpdateStrategy::ManualDuration(FRAME));
    app.world_mut()
        .resource_mut::<crate::enemy_population::EnemyPopulation>()
        .target_count = 0;
    app.finish();
    app.cleanup();
    // Bevy's first update carries a zero delta whatever the update strategy
    // says: that update is where the clock takes its baseline. Spending it
    // here, before anything is spawned, makes every later update a full
    // frame -- the same priming the headless arena does before frame zero.
    app.update();
    app
}

fn step(app: &mut App) {
    app.update();
}

fn player_of(app: &mut App) -> (Vec2, Vec2) {
    let mut query = app
        .world_mut()
        .query_filtered::<(&Transform, &Velocity2d), With<Player>>();
    let (transform, velocity) = query.single(app.world()).expect("one player");
    (transform.translation.truncate(), velocity.0)
}

#[test]
fn intent_written_this_update_moves_the_player_in_the_same_update() {
    let mut app = app();
    let player = app.world_mut().spawn(spawn_player_body(Vec3::ZERO)).id();
    app.world_mut()
        .entity_mut(player)
        .insert(PlayerIntent(Vec2::X));

    step(&mut app);

    let (position, velocity) = player_of(&mut app);
    assert!(position.x > 0.0, "the player should have moved right");
    assert!(velocity.x > 0.0, "and be carrying rightward velocity");
    assert!(
        position.y.abs() < f32::EPSILON,
        "with no drift on the unasked axis"
    );
}

#[test]
fn intent_persists_across_updates_without_being_rewritten() {
    let mut app = app();
    let player = app.world_mut().spawn(spawn_player_body(Vec3::ZERO)).id();
    app.world_mut()
        .entity_mut(player)
        .insert(PlayerIntent(Vec2::X));

    step(&mut app);
    let (first, _) = player_of(&mut app);
    step(&mut app);
    let (second, _) = player_of(&mut app);

    assert!(
        second.x > first.x,
        "a held direction keeps accelerating the player"
    );
}

#[test]
fn a_defeated_player_stops_and_stays_put() {
    let mut app = app();
    let player = app.world_mut().spawn(spawn_player_body(Vec3::ZERO)).id();
    app.world_mut()
        .entity_mut(player)
        .insert(PlayerIntent(Vec2::X));
    step(&mut app);
    let (moved, velocity) = player_of(&mut app);
    assert!(velocity.x > 0.0, "the player was moving before defeat");

    app.world_mut().entity_mut(player).insert(Defeated);
    step(&mut app);

    let (after, velocity) = player_of(&mut app);
    assert_eq!(after, moved, "a defeated player does not move");
    assert_eq!(velocity, Vec2::ZERO, "and keeps no velocity");
}

#[test]
fn zero_intent_brakes_rather_than_stopping_dead() {
    let mut app = app();
    let player = app.world_mut().spawn(spawn_player_body(Vec3::ZERO)).id();
    app.world_mut()
        .entity_mut(player)
        .insert(PlayerIntent(Vec2::X));
    for _ in 0..30 {
        step(&mut app);
    }
    let (_, moving) = player_of(&mut app);

    app.world_mut()
        .entity_mut(player)
        .insert(PlayerIntent(Vec2::ZERO));
    step(&mut app);

    let (_, braking) = player_of(&mut app);
    assert!(braking.x < moving.x, "releasing input slows the player");
    assert!(braking.x > 0.0, "but momentum carries it for a while");
}

#[test]
fn the_drawing_order_of_the_player_survives_movement() {
    let mut app = app();
    let player = app
        .world_mut()
        .spawn(spawn_player_body(Vec3::new(0.0, 0.0, 10.0)))
        .id();
    app.world_mut()
        .entity_mut(player)
        .insert(PlayerIntent(Vec2::ONE));

    step(&mut app);

    let z = app
        .world()
        .entity(player)
        .get::<Transform>()
        .expect("a transform")
        .translation
        .z;
    assert!(
        (z - 10.0).abs() < f32::EPSILON,
        "Z is presentation, not motion"
    );
}

// --- the headless arena -------------------------------------------------

/// A controlled arena: no enemies unless a test places them itself.
fn empty_arena(max_frames: u32) -> HeadlessArena {
    HeadlessArena::new(ArenaConfig {
        seed: 7,
        enemy_count: 0,
        max_frames,
        ..ArenaConfig::default()
    })
    .expect("an empty arena always builds")
}

/// Put a stationary enemy exactly where a test wants it.
fn place_enemy(arena: &mut HeadlessArena, kind: EnemyKind, position: Vec2, half: f32) -> Entity {
    arena
        .world_mut()
        .spawn((
            Enemy,
            kind,
            EnemySettings {
                boundary: BoundaryMode::Wrap,
                // Detection off, so the enemy stays where the test put it.
                detection_range: 0.0,
                retention_range: 0.0,
                ..EnemySettings::default()
            },
            Collider::rectangle(Vec2::splat(half)),
            Velocity2d::default(),
            Transform::from_translation(position.extend(0.0)),
        ))
        .id()
}

fn player_position(arena: &mut HeadlessArena) -> Vec2 {
    arena.view().player.expect("a player").position
}

#[test]
fn the_first_step_advances_exactly_one_frame_of_movement() {
    let mut arena = empty_arena(600);
    arena.set_intent(Vec2::X);

    let result = arena.step().expect("the first step runs");

    assert_eq!(result.frame, 1, "the first step is frame one");
    let reference = {
        let mut velocity = Vec2::ZERO;
        crate::motion::advance_motion(Vec2::ZERO, &mut velocity, Vec2::X, FRAME_SECONDS)
    };
    let moved = player_position(&mut arena);
    assert!(
        moved.distance(reference) < 0.001,
        "one step must integrate exactly one frame: {moved:?} against {reference:?}"
    );
}

#[test]
fn the_shared_timestep_is_the_duration_the_clock_actually_uses() {
    // Not `Duration::from_secs_f32(FRAME_SECONDS) == FRAME`: a sixtieth of a
    // second is not representable, so that round trip lands a nanosecond away
    // and would assert against arithmetic rather than against the contract.
    // What matters is that prediction integrates the duration the clock ticks.
    let clock = FRAME.as_secs_f32();
    assert!(
        (clock - FRAME_SECONDS).abs() < 1e-9,
        "prediction ({FRAME_SECONDS}) must integrate the clock's frame ({clock})"
    );
}

#[test]
fn frame_zero_has_the_full_population_and_nothing_has_moved() {
    let mut arena = HeadlessArena::new(ArenaConfig {
        seed: 11,
        enemy_count: 100,
        ..ArenaConfig::default()
    })
    .expect("a hundred enemies fit");

    let view = arena.view();
    assert_eq!(view.frame, 0, "no frame has been simulated yet");
    assert_eq!(view.enemies.len(), 100, "the arena starts full");
    assert!(view.blasts.is_empty(), "nothing has detonated");
    assert_eq!(
        view.player.expect("a player").position,
        Vec2::ZERO,
        "the player has not moved before frame zero"
    );
    assert!(
        view.enemies
            .iter()
            .all(|enemy| enemy.velocity == Vec2::ZERO),
        "no enemy has been stepped before frame zero"
    );
}

#[test]
fn a_population_that_does_not_fit_the_pass_budget_is_an_error() {
    // One pass places at most the population controller's per-update budget,
    // so asking for far more than that in a single pass cannot succeed.
    let outcome = HeadlessArena::new(ArenaConfig {
        seed: 3,
        enemy_count: 200,
        max_init_passes: 1,
        ..ArenaConfig::default()
    });

    match outcome
        .err()
        .expect("one pass cannot place two hundred enemies")
    {
        ArenaError::Population { placed, requested } => {
            assert_eq!(requested, 200);
            assert!(placed < 200, "the arena reports what it managed: {placed}");
        }
        other => panic!("expected a population error, got {other:?}"),
    }
}

#[test]
fn a_budget_of_zero_passes_cannot_place_an_enemy() {
    let error = HeadlessArena::new(ArenaConfig {
        seed: 3,
        enemy_count: 1,
        max_init_passes: 0,
        ..ArenaConfig::default()
    });
    assert_eq!(
        error.err(),
        Some(ArenaError::InvalidConfig(
            "max_init_passes must be at least one to place any enemy"
        )),
    );
}

#[test]
fn a_zero_frame_budget_is_rejected() {
    let error = HeadlessArena::new(ArenaConfig {
        max_frames: 0,
        ..ArenaConfig::default()
    });
    assert_eq!(
        error.err(),
        Some(ArenaError::InvalidConfig(
            "max_frames must be at least one frame"
        ))
    );
}

#[test]
fn touching_an_enemy_terminates_the_episode() {
    let mut arena = empty_arena(600);
    place_enemy(&mut arena, EnemyKind::Normal, Vec2::ZERO, 20.0);

    let result = arena.step().expect("the step runs");

    assert!(result.hit, "an enemy on the player is a hit");
    assert!(result.terminated, "a hit ends the episode");
    assert!(!result.truncated, "the frame budget is untouched");
    assert_eq!(
        result.enemy_deaths, 0,
        "the enemy that reaches the player is not an enemy-on-enemy death"
    );
}

#[test]
fn a_blast_kills_the_player_too() {
    let mut arena = empty_arena(600);
    arena.world_mut().spawn((
        KamikazeBlast::default(),
        Collider::rectangle(Vec2::splat(200.0)),
        Transform::from_translation(Vec3::ZERO),
    ));

    let result = arena.step().expect("the step runs");

    assert!(result.hit, "the blast covers the player");
    assert!(result.terminated);
}

#[test]
fn a_finished_episode_stays_frozen_until_it_is_reset() {
    let mut arena = empty_arena(600);
    place_enemy(&mut arena, EnemyKind::Normal, Vec2::ZERO, 20.0);
    let ended = arena.step().expect("the fatal step runs");
    assert!(ended.terminated);
    let after_death = arena.view();

    assert_eq!(
        arena.step().err(),
        Some(ArenaError::Completed),
        "a completed episode refuses to advance"
    );
    assert_eq!(
        arena.view(),
        after_death,
        "and nothing about the world changed"
    );

    arena.reset(99).expect("reset builds a fresh episode");
    assert_eq!(arena.frame(), 0, "reset returns to frame zero");
    assert!(!arena.done(), "and clears the ending");
    assert!(arena.step().is_ok(), "so stepping works again");
}

#[test]
fn the_frame_budget_truncates_without_terminating() {
    let mut arena = empty_arena(3);
    for frame in 1..=3 {
        let result = arena.step().expect("steps within the budget run");
        assert_eq!(result.frame, frame);
        assert_eq!(
            result.truncated,
            frame == 3,
            "only the last frame is a truncation"
        );
        assert!(!result.terminated, "running out of frames is not a death");
    }
    assert_eq!(arena.step().err(), Some(ArenaError::Completed));
}

#[test]
fn death_and_the_frame_budget_can_land_on_the_same_frame() {
    let mut arena = empty_arena(1);
    place_enemy(&mut arena, EnemyKind::Normal, Vec2::ZERO, 20.0);

    let result = arena.step().expect("the only step runs");

    assert!(result.terminated, "the player died");
    assert!(result.truncated, "and the budget ran out");
    assert!(result.done());
}

#[test]
fn a_cluster_of_colliding_enemies_is_counted_once_each() {
    let mut arena = empty_arena(600);
    // Three overlapping enemies, well away from the player: every one of them
    // dies to the others in the same update.
    let far = Vec2::new(1_000.0, 1_000.0);
    for offset in [Vec2::ZERO, Vec2::new(5.0, 0.0), Vec2::new(0.0, 5.0)] {
        place_enemy(&mut arena, EnemyKind::Normal, far + offset, 30.0);
    }

    let result = arena.step().expect("the step runs");

    assert_eq!(
        result.enemy_deaths, 3,
        "each victim is counted exactly once"
    );
    assert!(!result.hit, "none of this reached the player");
    assert!(
        arena.view().enemies.is_empty(),
        "and the cluster is gone from the view"
    );
}

#[test]
fn per_step_counters_do_not_leak_into_the_next_step() {
    let mut arena = empty_arena(600);
    let far = Vec2::new(1_000.0, 1_000.0);
    for offset in [Vec2::ZERO, Vec2::new(5.0, 0.0)] {
        place_enemy(&mut arena, EnemyKind::Normal, far + offset, 30.0);
    }

    let first = arena.step().expect("the collision step runs");
    assert_eq!(first.enemy_deaths, 2);
    let second = arena.step().expect("the next step runs");
    assert_eq!(second.enemy_deaths, 0, "the tally starts again each frame");
}

#[test]
fn the_view_reports_world_units_identity_and_blast_phase() {
    let mut arena = empty_arena(600);
    let enemy = place_enemy(
        &mut arena,
        EnemyKind::Kamikaze,
        Vec2::new(500.0, -250.0),
        15.0,
    );
    arena.world_mut().spawn((
        KamikazeBlast { age: 0.25 },
        Collider::rectangle(Vec2::splat(40.0)),
        Transform::from_translation(Vec3::new(-100.0, 0.0, 0.0)),
    ));

    let view = arena.view();

    let seen = view
        .enemies
        .iter()
        .find(|candidate| candidate.entity == enemy)
        .expect("the placed enemy is in the view");
    assert_eq!(seen.kind, EnemyKind::Kamikaze);
    assert_eq!(seen.position, Vec2::new(500.0, -250.0));
    assert_eq!(seen.collider.half_extents, Vec2::splat(15.0));
    assert_eq!(view.half_extents, crate::scale::WORLD_HALF_EXTENTS);

    let blast = view.blasts.first().expect("the blast is in the view");
    assert!((blast.age - 0.25).abs() < f32::EPSILON);
    assert!(blast.duration > 0.0, "the phase needs the full duration");
}

#[test]
fn dying_enemies_leave_the_view_before_they_can_be_painted() {
    let mut arena = empty_arena(600);
    let far = Vec2::new(1_000.0, 1_000.0);
    for offset in [Vec2::ZERO, Vec2::new(5.0, 0.0)] {
        place_enemy(&mut arena, EnemyKind::Normal, far + offset, 30.0);
    }

    arena.step().expect("the collision step runs");

    assert!(
        arena.view().enemies.is_empty(),
        "a hazard that cannot kill is not reported as one"
    );
}

#[test]
fn the_same_seed_and_actions_replay_the_same_run() {
    let intents = [Vec2::X, Vec2::Y, Vec2::ZERO, -Vec2::X, Vec2::ONE];
    let run = |seed: u64| {
        let mut arena = HeadlessArena::new(ArenaConfig {
            seed,
            enemy_count: 40,
            max_frames: 600,
            ..ArenaConfig::default()
        })
        .expect("the arena fills");
        for frame in 0..600 {
            arena.set_intent(intents[frame % intents.len()]);
            if arena.done() {
                break;
            }
            arena.step().expect("a step inside the budget");
        }
        arena.view()
    };

    assert_eq!(run(2_024), run(2_024), "one seed is one run");
}

#[test]
fn different_seeds_place_and_move_enemies_differently() {
    let positions = |seed: u64| {
        let mut arena = HeadlessArena::new(ArenaConfig {
            seed,
            enemy_count: 40,
            max_frames: 60,
            ..ArenaConfig::default()
        })
        .expect("the arena fills");
        for _ in 0..30 {
            arena.step().expect("a step inside the budget");
        }
        arena
            .view()
            .enemies
            .iter()
            .map(|enemy| enemy.position)
            .collect::<Vec<_>>()
    };

    assert_ne!(
        positions(1),
        positions(2),
        "a different seed is a different arena, not just a different number"
    );
}

#[test]
fn reset_reproduces_a_seed_exactly() {
    let mut first = HeadlessArena::new(ArenaConfig {
        seed: 5,
        enemy_count: 20,
        max_frames: 120,
        ..ArenaConfig::default()
    })
    .expect("the arena fills");
    for _ in 0..20 {
        first.set_intent(Vec2::X);
        first.step().expect("a step inside the budget");
    }
    let expected = first.view();

    let mut second = HeadlessArena::new(ArenaConfig {
        seed: 999,
        enemy_count: 20,
        max_frames: 120,
        ..ArenaConfig::default()
    })
    .expect("the arena fills");
    second.step().expect("a step on the old seed");
    second.reset(5).expect("reset to the first arena's seed");
    for _ in 0..20 {
        second.set_intent(Vec2::X);
        second.step().expect("a step inside the budget");
    }

    assert_eq!(
        second.view(),
        expected,
        "reset is a fresh episode on a seed"
    );
}

#[test]
fn enemies_wrap_at_the_seam_rather_than_stopping() {
    let arena = HeadlessArena::new(ArenaConfig {
        seed: 4,
        enemy_count: 10,
        max_frames: 60,
        ..ArenaConfig::default()
    })
    .expect("the arena fills");
    let population = arena_population(arena);
    assert!(
        population
            .iter()
            .all(|boundary| *boundary == BoundaryMode::Wrap),
        "the catalog's wrapping must survive the move into the simulation"
    );
}

fn arena_population(mut arena: HeadlessArena) -> Vec<BoundaryMode> {
    let world = arena.world_mut();
    world
        .query_filtered::<&EnemySettings, With<Enemy>>()
        .iter(world)
        .map(|settings| settings.boundary)
        .collect()
}

#[test]
fn the_arena_world_wraps_at_the_documented_extents() {
    let mut arena = empty_arena(600);
    assert_eq!(
        arena.world_mut().resource::<EnemyWorld>().half_extents,
        Some(crate::scale::WORLD_HALF_EXTENTS)
    );
}

#[test]
fn an_empty_population_is_allowed_for_controlled_tests() {
    let mut arena = empty_arena(60);
    assert!(arena.view().enemies.is_empty());
    assert!(arena.step().is_ok());
}

// --- actions and predicted paths ---------------------------------------

/// Walk a real arena with one action held for `hold_frames`, sampling where
/// the player actually is at each horizon.
fn simulated_path(
    action: Action,
    hold_frames: u32,
    start: Vec2,
    warmup: Option<(Vec2, u32)>,
) -> [Vec2; SAMPLE_COUNT] {
    let mut arena = HeadlessArena::new(ArenaConfig {
        seed: 1,
        enemy_count: 0,
        max_frames: 10_000,
        ..ArenaConfig::default()
    })
    .expect("an empty arena builds");
    if start != Vec2::ZERO {
        let player = arena.player();
        arena
            .world_mut()
            .entity_mut(player)
            .insert(Transform::from_translation(start.extend(0.0)));
    }
    // An optional run-up, so the prediction starts from a moving player.
    if let Some((direction, frames)) = warmup {
        arena.set_intent(direction);
        for _ in 0..frames {
            arena.step().expect("warm-up steps run");
        }
    }
    let mut samples = [Vec2::ZERO; SAMPLE_COUNT];
    let mut next = 0;
    for frame in 1..=HORIZON {
        arena.set_intent(if frame <= hold_frames {
            action.direction()
        } else {
            Vec2::ZERO
        });
        arena.step().expect("a step inside the budget");
        while SAMPLE_FRAMES.get(next).is_some_and(|at| *at == frame) {
            if let Some(slot) = samples.get_mut(next) {
                *slot = arena.view().player.expect("a player").position;
            }
            next = next.saturating_add(1);
        }
    }
    samples
}

fn predicted_path(
    action: Action,
    hold_frames: u32,
    start: Vec2,
    warmup: Option<(Vec2, u32)>,
) -> [Vec2; SAMPLE_COUNT] {
    let (position, velocity) = match warmup {
        None => (start, Vec2::ZERO),
        Some((direction, frames)) => {
            let mut arena = HeadlessArena::new(ArenaConfig {
                seed: 1,
                enemy_count: 0,
                max_frames: 10_000,
                ..ArenaConfig::default()
            })
            .expect("an empty arena builds");
            let player = arena.player();
            arena
                .world_mut()
                .entity_mut(player)
                .insert(Transform::from_translation(start.extend(0.0)));
            arena.set_intent(direction);
            for _ in 0..frames {
                arena.step().expect("warm-up steps run");
            }
            let view = arena.view();
            let player = view.player.expect("a player");
            (player.position, player.velocity)
        }
    };
    predict_path(position, velocity, action.direction(), hold_frames).expect("finite inputs")
}

fn assert_paths_agree(action: Action, hold: u32, start: Vec2, warmup: Option<(Vec2, u32)>) {
    let predicted = predicted_path(action, hold, start, warmup);
    let simulated = simulated_path(action, hold, start, warmup);
    for (index, (left, right)) in predicted.iter().zip(simulated.iter()).enumerate() {
        let frame = SAMPLE_FRAMES[index];
        // A world unit is a sixth of a reference pixel; agreement well inside
        // one unit is agreement inside the encoder's cell by a wide margin.
        assert!(
            left.distance(*right) < 0.5,
            "{} hold {hold} at frame {frame}: predicted {left:?}, simulated {right:?}",
            action.name()
        );
    }
}

#[test]
fn every_action_predicts_where_the_arena_actually_goes_from_rest() {
    for action in Action::ALL {
        assert_paths_agree(action, DEFAULT_HOLD_FRAMES, Vec2::ZERO, None);
    }
}

#[test]
fn prediction_agrees_with_the_arena_from_a_moving_start() {
    // Run right for half a second first, then predict each action from the
    // momentum that leaves behind.
    for action in Action::ALL {
        assert_paths_agree(action, DEFAULT_HOLD_FRAMES, Vec2::ZERO, Some((Vec2::X, 30)));
    }
}

#[test]
fn prediction_agrees_for_every_hold_length() {
    for hold in [0, 1, DEFAULT_HOLD_FRAMES, 108] {
        assert_paths_agree(Action::Right, hold, Vec2::ZERO, None);
        assert_paths_agree(Action::UpLeft, hold, Vec2::ZERO, Some((Vec2::Y, 20)));
    }
}

#[test]
fn a_released_action_brakes_instead_of_stopping_dead() {
    let path = predicted_path(Action::Right, 4, Vec2::ZERO, None);
    let travelled: Vec<f32> = path.iter().map(|point| point.x).collect();
    assert!(travelled[0] > 0.0, "the held frames move the player");
    for pair in travelled.windows(2) {
        assert!(
            pair[1] >= pair[0] - 0.001,
            "coasting carries on in the same direction: {travelled:?}"
        );
    }
    // In reference pixels, which is what the design reasons in: four frames of
    // input plus coasting is about ten pixels of momentum, not a journey.
    let last = travelled[SAMPLE_COUNT - 1] / crate::scale::PIXEL;
    assert!(
        (8.0..12.0).contains(&last),
        "four frames of input carries about ten pixels, not {last}"
    );
}

#[test]
fn idle_predicts_braking_from_a_moving_start() {
    let moving = predicted_path(
        Action::Idle,
        DEFAULT_HOLD_FRAMES,
        Vec2::ZERO,
        Some((Vec2::X, 30)),
    );
    let gaps: Vec<f32> = moving
        .windows(2)
        .map(|pair| pair[1].x - pair[0].x)
        .collect();
    assert!(gaps[0] > 0.0, "momentum carries the player on");
    assert!(
        gaps.last().copied().unwrap_or(1.0) < gaps[0],
        "and decays rather than continuing forever: {gaps:?}"
    );
}

#[test]
fn a_reversal_turns_the_player_around() {
    let path = predicted_path(
        Action::Left,
        DEFAULT_HOLD_FRAMES,
        Vec2::ZERO,
        Some((Vec2::X, 30)),
    );
    assert!(
        path[SAMPLE_COUNT - 1].x < path[0].x,
        "holding left against rightward momentum ends up further left: {path:?}"
    );
}

#[test]
fn a_diagonal_is_no_faster_than_a_cardinal() {
    let straight = predicted_path(Action::Right, 108, Vec2::ZERO, None);
    let diagonal = predicted_path(Action::UpRight, 108, Vec2::ZERO, None);
    let straight_distance = straight[SAMPLE_COUNT - 1].length();
    let diagonal_distance = diagonal[SAMPLE_COUNT - 1].length();
    assert!(
        (straight_distance - diagonal_distance).abs() < 0.5,
        "normalisation keeps both at the same speed: {straight_distance} against {diagonal_distance}"
    );
}

#[test]
fn predicted_paths_cross_both_seams_the_way_the_arena_does() {
    let half = crate::scale::WORLD_HALF_EXTENTS;
    // Start a few units inside each edge, heading out of the world.
    let cases = [
        (Action::Right, Vec2::new(half.x - 5.0, 0.0)),
        (Action::Up, Vec2::new(0.0, half.y - 5.0)),
    ];
    for (action, start) in cases {
        let path = predicted_path(action, 108, start, None);
        for point in path {
            assert!(
                point.x.abs() <= half.x && point.y.abs() <= half.y,
                "a wrapped path stays inside the world: {point:?}"
            );
        }
        assert_paths_agree(action, 108, start, None);
    }
}

#[test]
fn prediction_leaves_the_arena_untouched() {
    let mut arena = HeadlessArena::new(ArenaConfig {
        seed: 12,
        enemy_count: 20,
        max_frames: 600,
        ..ArenaConfig::default()
    })
    .expect("the arena fills");
    arena.set_intent(Vec2::X);
    arena.step().expect("one step before predicting");
    let before = arena.view();

    let paths = arena
        .predict_paths(DEFAULT_HOLD_FRAMES)
        .expect("finite state");

    assert_eq!(arena.view(), before, "predicting is not playing");
    assert_eq!(paths.len(), 9, "one path per action");
    let path_of = |action: Action| {
        *paths
            .get(usize::from(action.index()))
            .expect("every action has a path")
    };
    assert_ne!(
        path_of(Action::Left),
        path_of(Action::Right),
        "and the actions are actually different"
    );
}

#[test]
fn predicted_paths_start_from_the_players_momentum() {
    let at_rest =
        predict_path(Vec2::ZERO, Vec2::ZERO, Action::Idle.direction(), 24).expect("finite inputs");
    let moving = predict_path(
        Vec2::ZERO,
        Vec2::new(900.0, 0.0),
        Action::Idle.direction(),
        24,
    )
    .expect("finite inputs");
    assert!(
        moving[0].x > at_rest[0].x,
        "a moving player's idle path is not a standing one"
    );
}

#[test]
fn action_bytes_are_the_protocol_order() {
    let names: Vec<&str> = Action::ALL.iter().map(|action| action.name()).collect();
    assert_eq!(
        names,
        vec![
            "idle",
            "left",
            "right",
            "up",
            "down",
            "up-left",
            "up-right",
            "down-left",
            "down-right"
        ]
    );
    for (index, action) in Action::ALL.iter().enumerate() {
        let byte = u8::try_from(index).expect("nine actions fit a byte");
        assert_eq!(Action::from_byte(byte), Some(*action));
        assert_eq!(action.index(), byte);
    }
    assert_eq!(Action::from_byte(9), None, "there is no tenth action");
}

#[test]
fn an_action_outside_the_table_is_rejected_rather_than_idled() {
    let mut arena = empty_arena(60);
    assert_eq!(
        arena.set_action(9).err(),
        Some(ArenaError::InvalidAction(9))
    );
    assert!(arena.set_action(8).is_ok(), "the ninth action is valid");
}

#[test]
fn non_finite_prediction_inputs_fail_instead_of_reaching_an_observation() {
    let bad = predict_path(Vec2::new(f32::NAN, 0.0), Vec2::ZERO, Vec2::X, 24);
    assert!(matches!(bad, Err(ArenaError::NonFinite(_))));
    let bad = predict_path(Vec2::ZERO, Vec2::new(0.0, f32::INFINITY), Vec2::X, 24);
    assert!(matches!(bad, Err(ArenaError::NonFinite(_))));
    let bad = predict_path(Vec2::ZERO, Vec2::ZERO, Vec2::new(f32::NAN, 1.0), 24);
    assert!(matches!(bad, Err(ArenaError::NonFinite(_))));
}
