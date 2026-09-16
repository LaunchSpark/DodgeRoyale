//! The gameplay simulation, without a renderer.
//!
//! The browser and native games add art, menus and a camera on top of this; the
//! headless arena that trains an agent runs exactly the same systems with none
//! of that. Keeping one copy is what lets a trained policy move the player the
//! way a human's keyboard does: both write [`PlayerIntent`], and
//! [`move_players`] is the only thing that integrates it.

use bevy::prelude::*;

use crate::collision::Collider;
use crate::enemy::{EnemyPlugin, EnemySet, EnemyTarget, EnemyWorld, Velocity2d};
use crate::enemy_population::EnemyPopulationPlugin;
use crate::enemy_types::{Defeated, ReferenceEnemyPlugin};
use crate::motion::{PLAYER_HALF_SIZE, advance_motion};
use crate::rng::SeededRngPlugin;
use crate::scale::WORLD_HALF_EXTENTS;

/// The actor a player or a policy steers.
#[derive(Component, Debug, Default, Clone, Copy)]
pub struct Player;

/// The direction movement is asked for this update.
///
/// A direction, never a speed: length above one grants no extra speed, because
/// [`advance_motion`] normalizes before applying the fixed top speed. Writers
/// run before [`PlayerSet::Move`], and the value is read, not consumed, so a
/// held key and a policy that repeats an action look identical.
#[derive(Component, Debug, Default, Clone, Copy, PartialEq)]
pub struct PlayerIntent(pub Vec2);

/// Player movement, ordered before the enemies react to it.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlayerSet {
    Move,
}

/// Everything the simulation advances, as one gate.
///
/// The game runs this only while playing; the headless arena runs it every
/// update. Run conditions on this set reach every system inside it.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SimulationSet;

/// Installs the simulation: seeded randomness, the player, and the enemies.
///
/// Add this once. The graphical app gates [`SimulationSet`] on its playing
/// screen and decorates the actors afterwards.
pub struct SimulationPlugin;

impl Plugin for SimulationPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            SeededRngPlugin,
            EnemyPlugin,
            EnemyPopulationPlugin,
            ReferenceEnemyPlugin,
        ))
        // The enemy plugin defaults to these bounds; stating them here keeps the
        // arena's size a property of the simulation rather than a coincidence.
        .insert_resource(EnemyWorld {
            half_extents: Some(WORLD_HALF_EXTENTS),
            ..default()
        })
        // The enemy sets already chain among themselves. This nests them, and
        // player movement, inside one gate, and puts movement first so enemies
        // steer toward where the player has just moved.
        .configure_sets(
            Update,
            (
                PlayerSet::Move,
                EnemySet::Prepare,
                EnemySet::Steer,
                EnemySet::Modifiers,
                EnemySet::Move,
                EnemySet::Deaths,
                EnemySet::Contacts,
                EnemySet::Effects,
                EnemySet::Cleanup,
                EnemySet::Replenish,
            )
                .chain()
                .in_set(SimulationSet),
        )
        .add_systems(Update, move_players.in_set(PlayerSet::Move));
    }
}

/// The simulation half of a player: no sprite, no trail, no shadow.
///
/// The graphical game spawns this and adds its own art to the same entity.
#[must_use]
pub fn spawn_player_body(translation: Vec3) -> impl Bundle {
    (
        Player,
        EnemyTarget,
        Collider::rectangle(Vec2::splat(PLAYER_HALF_SIZE)),
        Velocity2d::default(),
        PlayerIntent::default(),
        Transform::from_translation(translation),
    )
}

/// Integrate every player's intent into velocity and position.
///
/// A defeated player stops dead and stays put until the screen is reset, which
/// is what the graphical game relied on before movement moved here.
#[expect(
    clippy::needless_pass_by_value,
    reason = "Bevy injects system parameters by value"
)]
pub fn move_players(
    time: Res<Time>,
    mut players: Query<
        (
            &mut Transform,
            &mut Velocity2d,
            &PlayerIntent,
            Option<&Defeated>,
        ),
        With<Player>,
    >,
) {
    let seconds = time.delta_secs();
    for (mut transform, mut velocity, intent, defeated) in &mut players {
        if defeated.is_some() {
            velocity.0 = Vec2::ZERO;
            continue;
        }
        let position = advance_motion(
            transform.translation.truncate(),
            &mut velocity.0,
            intent.0,
            seconds,
        );
        // Only the plane moves: Z is the drawing order the game chose.
        transform.translation.x = position.x;
        transform.translation.y = position.y;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::time::TimeUpdateStrategy;
    use std::time::Duration;

    /// One 60 Hz frame, the only timestep the simulation is stepped by.
    const FRAME: Duration = Duration::from_nanos(16_666_667);

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
}
