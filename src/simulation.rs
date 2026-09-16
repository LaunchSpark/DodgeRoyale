//! The gameplay simulation, without a renderer.
//!
//! The browser and native games add art, menus and a camera on top of this; the
//! headless arena that trains an agent runs exactly the same systems with none
//! of that. Keeping one copy is what lets a trained policy move the player the
//! way a human's keyboard does: both write [`PlayerIntent`], and
//! [`move_players`] is the only thing that integrates it.

use bevy::ecs::schedule::{ScheduleLabel, SingleThreadedExecutor};
use bevy::prelude::*;
use bevy::time::TimeUpdateStrategy;
use std::time::Duration;

use crate::collision::Collider;
use crate::enemy::{Dying, Enemy, EnemyPlugin, EnemySet, EnemyTarget, EnemyWorld, Velocity2d};
use crate::enemy_population::{EnemyPopulation, EnemyPopulationPlugin, replenish_enemies};
use crate::enemy_types::{
    Defeated, EnemyKind, KamikazeBlast, KamikazeSettings, PlayerHit, ReferenceEnemyPlugin,
};
use crate::motion::{PLAYER_HALF_SIZE, advance_motion};
use crate::rng::{GameSeed, SeededRngPlugin};
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

/// One 60 Hz frame, as a duration a `Duration` can represent exactly.
///
/// The simulation is stepped by this and nothing else: manual time advances by
/// it, and path prediction integrates by it, so a predicted position and a
/// simulated one are the same arithmetic.
pub const FRAME: Duration = Duration::from_nanos(16_666_667);

/// [`FRAME`] in seconds, for arithmetic that wants a float.
pub const FRAME_SECONDS: f32 = 16_666_667.0 / 1_000_000_000.0;

/// How many spawn-only passes an arena may run before it gives up filling.
///
/// Placement samples random positions and can fail, so filling is not a fixed
/// number of passes; this bounds the retries rather than looping forever.
pub const DEFAULT_MAX_INIT_PASSES: u32 = 64;

/// Where the player draws, and the Z its transform keeps.
const PLAYER_Z: f32 = 10.0;

/// What a [`HeadlessArena`] is built from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArenaConfig {
    /// Fixes the run: the same seed and the same actions replay exactly.
    pub seed: u64,
    /// Enemies the arena is kept stocked with. Zero is allowed, for tests.
    pub enemy_count: usize,
    /// Frames an episode may last before it is truncated. Never zero.
    pub max_frames: u32,
    /// Spawn-only passes allowed while filling the arena before frame zero.
    pub max_init_passes: u32,
}

impl Default for ArenaConfig {
    fn default() -> Self {
        Self {
            seed: 0,
            enemy_count: 100,
            max_frames: 3_600,
            max_init_passes: DEFAULT_MAX_INIT_PASSES,
        }
    }
}

impl ArenaConfig {
    /// Reject a configuration that cannot produce a playable episode.
    const fn validate(self) -> Result<Self, ArenaError> {
        if self.max_frames == 0 {
            return Err(ArenaError::InvalidConfig(
                "max_frames must be at least one frame",
            ));
        }
        if self.max_init_passes == 0 && self.enemy_count > 0 {
            return Err(ArenaError::InvalidConfig(
                "max_init_passes must be at least one to place any enemy",
            ));
        }
        Ok(self)
    }
}

/// Why an arena could not start or step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArenaError {
    /// The configuration could never produce an episode.
    InvalidConfig(&'static str),
    /// The arena could not place the requested population before frame zero.
    Population { placed: usize, requested: usize },
    /// The episode has already ended; reset before stepping again.
    Completed,
}

impl core::fmt::Display for ArenaError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match *self {
            Self::InvalidConfig(reason) => write!(formatter, "invalid arena config: {reason}"),
            Self::Population { placed, requested } => write!(
                formatter,
                "placed {placed} of {requested} enemies before frame zero"
            ),
            Self::Completed => write!(formatter, "the episode has ended; reset before stepping"),
        }
    }
}

impl core::error::Error for ArenaError {}

/// What one [`HeadlessArena::step`] produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StepResult {
    /// Frames simulated in this episode, counting this one.
    pub frame: u32,
    /// The controlled player was hit during this frame.
    pub hit: bool,
    /// Enemies killed by enemy-on-enemy collision during this frame.
    pub enemy_deaths: u32,
    /// The episode ended because the player died.
    pub terminated: bool,
    /// The episode ended because it ran out of frames.
    pub truncated: bool,
}

impl StepResult {
    /// Whether this frame ended the episode, for either reason.
    #[must_use]
    pub const fn done(&self) -> bool {
        self.terminated || self.truncated
    }
}

/// The player, in world units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlayerView {
    pub entity: Entity,
    pub position: Vec2,
    pub velocity: Vec2,
    pub collider: Collider,
}

/// One live enemy, in world units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnemyView {
    /// Identity, so a cell contested by two hazards breaks its tie the same
    /// way on every run.
    pub entity: Entity,
    pub kind: EnemyKind,
    pub position: Vec2,
    pub velocity: Vec2,
    pub collider: Collider,
}

/// One kamikaze blast, in world units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlastView {
    pub entity: Entity,
    pub position: Vec2,
    pub collider: Collider,
    /// Seconds since it detonated. With [`Self::duration`] this says whether
    /// the blast is still growing or already shrinking, which its size alone
    /// cannot: the same width occurs twice.
    pub age: f32,
    /// Seconds the blast lives in total.
    pub duration: f32,
}

/// Everything the encoder needs from one frame, in world units.
#[derive(Debug, Clone, PartialEq)]
pub struct ArenaView {
    pub frame: u32,
    pub player: Option<PlayerView>,
    pub enemies: Vec<EnemyView>,
    pub blasts: Vec<BlastView>,
    /// Half the arena's width and height; the world wraps at these edges.
    pub half_extents: Vec2,
}

/// Per-frame tallies, cleared before each step.
#[derive(Resource, Debug, Default, Clone, Copy)]
struct FrameCounters {
    hit: bool,
    enemy_deaths: u32,
}

/// Whether the simulation systems may run this update.
///
/// Startup and population filling both need the app to update without any play
/// happening, so the gate is closed until frame zero is ready.
#[derive(Resource, Debug, Default, Clone, Copy, PartialEq, Eq)]
struct SimulationGate(bool);

/// The initialisation schedule: placement only, no movement or contacts.
#[derive(ScheduleLabel, Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ArenaInit;

/// A deterministic, windowless game, stepped one frame at a time.
///
/// Built for training: `step` advances exactly one 60 Hz frame, `view` reports
/// the world in world units, and the same seed with the same intents replays
/// the same run.
pub struct HeadlessArena {
    app: App,
    config: ArenaConfig,
    player: Entity,
    frame: u32,
    terminated: bool,
    truncated: bool,
}

impl HeadlessArena {
    /// Build an arena and fill it, leaving it ready for its first step.
    ///
    /// # Errors
    ///
    /// [`ArenaError::InvalidConfig`] for a configuration that cannot play, and
    /// [`ArenaError::Population`] when the arena could not be filled within
    /// its pass budget.
    pub fn new(config: ArenaConfig) -> Result<Self, ArenaError> {
        let config = config.validate()?;
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, SimulationPlugin))
            .insert_resource(TimeUpdateStrategy::ManualDuration(FRAME))
            .insert_resource(GameSeed::new(config.seed))
            .init_resource::<FrameCounters>()
            .init_resource::<SimulationGate>()
            .configure_sets(Update, SimulationSet.run_if(simulation_running))
            .add_systems(
                Update,
                (
                    // Between the two death sources: enemy-on-enemy kills are
                    // marked in Deaths, and the enemy that reaches the player
                    // is marked later, in Effects.
                    count_enemy_deaths
                        .after(EnemySet::Deaths)
                        .before(EnemySet::Contacts),
                    // Blasts report in Prepare and contacts in Effects, so
                    // reading after Effects catches both.
                    record_player_hits.after(EnemySet::Effects),
                )
                    .in_set(SimulationSet),
            );
        app.world_mut()
            .resource_mut::<EnemyPopulation>()
            .target_count = config.enemy_count;
        // Placement is the only thing that may happen before frame zero.
        app.add_schedule(Schedule::new(ArenaInit));
        app.edit_schedule(ArenaInit, |schedule| {
            schedule.add_systems(replenish_enemies);
        });
        single_threaded(&mut app);
        app.finish();
        app.cleanup();

        // Bevy's first update takes the clock's baseline and reports a zero
        // delta whatever the update strategy says. Spending it with the gate
        // closed means frame zero is a full frame, and nothing has moved.
        app.update();

        let player = app
            .world_mut()
            .spawn(spawn_player_body(Vec3::new(0.0, 0.0, PLAYER_Z)))
            .id();

        let mut arena = Self {
            app,
            config,
            player,
            frame: 0,
            terminated: false,
            truncated: false,
        };
        arena.fill()?;
        arena.app.world_mut().resource_mut::<SimulationGate>().0 = true;
        Ok(arena)
    }

    /// Run spawn-only passes until the arena is full, or the budget runs out.
    fn fill(&mut self) -> Result<(), ArenaError> {
        for _ in 0..self.config.max_init_passes {
            if self.enemy_count() >= self.config.enemy_count {
                return Ok(());
            }
            self.app.world_mut().run_schedule(ArenaInit);
        }
        let placed = self.enemy_count();
        if placed >= self.config.enemy_count {
            Ok(())
        } else {
            Err(ArenaError::Population {
                placed,
                requested: self.config.enemy_count,
            })
        }
    }

    fn enemy_count(&mut self) -> usize {
        let world = self.app.world_mut();
        world
            .query_filtered::<(), With<Enemy>>()
            .iter(world)
            .count()
    }

    /// Ask for a direction on the next step. Length is ignored; zero coasts.
    pub fn set_intent(&mut self, direction: Vec2) {
        if let Some(mut intent) = self.app.world_mut().get_mut::<PlayerIntent>(self.player) {
            intent.0 = direction;
        }
    }

    /// Advance exactly one 60 Hz frame.
    ///
    /// # Errors
    ///
    /// [`ArenaError::Completed`] once the episode has ended. A finished episode
    /// stays frozen rather than quietly simulating past its own ending.
    pub fn step(&mut self) -> Result<StepResult, ArenaError> {
        if self.done() {
            return Err(ArenaError::Completed);
        }
        *self.app.world_mut().resource_mut::<FrameCounters>() = FrameCounters::default();
        self.app.update();
        let counters = *self.app.world().resource::<FrameCounters>();

        self.frame = self.frame.saturating_add(1);
        // Latched: a hit ends the episode even though the world could carry on.
        self.terminated |= counters.hit;
        self.truncated = self.frame >= self.config.max_frames;
        Ok(StepResult {
            frame: self.frame,
            hit: counters.hit,
            enemy_deaths: counters.enemy_deaths,
            terminated: self.terminated,
            // Death and the frame budget can land on the same frame; both are
            // reported, and the consumer decides which it bootstraps from.
            truncated: self.truncated,
        })
    }

    /// Frames simulated in this episode.
    #[must_use]
    pub const fn frame(&self) -> u32 {
        self.frame
    }

    /// Whether this episode has ended, for either reason.
    #[must_use]
    pub const fn done(&self) -> bool {
        self.terminated || self.truncated
    }

    /// The configuration this arena was built with.
    #[must_use]
    pub const fn config(&self) -> ArenaConfig {
        self.config
    }

    /// The entity the arena steers.
    #[must_use]
    pub const fn player(&self) -> Entity {
        self.player
    }

    /// Start a new episode on a new seed, discarding everything before it.
    ///
    /// # Errors
    ///
    /// As [`HeadlessArena::new`].
    pub fn reset(&mut self, seed: u64) -> Result<(), ArenaError> {
        *self = Self::new(ArenaConfig {
            seed,
            ..self.config
        })?;
        Ok(())
    }

    /// The world this frame, in world units.
    pub fn view(&mut self) -> ArenaView {
        let half_extents = self
            .app
            .world()
            .resource::<EnemyWorld>()
            .half_extents
            .unwrap_or(WORLD_HALF_EXTENTS);
        let blast_duration = self
            .app
            .world()
            .resource::<KamikazeSettings>()
            .blast_seconds;
        let frame = self.frame;
        let world = self.app.world_mut();

        let player = world
            .query_filtered::<(Entity, &Transform, &Velocity2d, &Collider), With<Player>>()
            .iter(world)
            .next()
            .map(|(entity, transform, velocity, collider)| PlayerView {
                entity,
                position: transform.translation.truncate(),
                velocity: velocity.0,
                collider: *collider,
            });

        // Dying enemies are excluded: they are removed this update and cannot
        // kill anything, so an encoder that painted them would invent a threat.
        let mut enemies: Vec<EnemyView> = world
            .query_filtered::<(Entity, &EnemyKind, &Transform, &Velocity2d, &Collider), (With<Enemy>, Without<Dying>)>()
            .iter(world)
            .map(|(entity, kind, transform, velocity, collider)| EnemyView {
                entity,
                kind: *kind,
                position: transform.translation.truncate(),
                velocity: velocity.0,
                collider: *collider,
            })
            .collect();
        enemies.sort_unstable_by_key(|enemy| enemy.entity);

        let mut blasts: Vec<BlastView> = world
            .query::<(Entity, &KamikazeBlast, &Transform, &Collider)>()
            .iter(world)
            .map(|(entity, blast, transform, collider)| BlastView {
                entity,
                position: transform.translation.truncate(),
                collider: *collider,
                age: blast.age,
                duration: blast_duration,
            })
            .collect();
        blasts.sort_unstable_by_key(|blast| blast.entity);

        ArenaView {
            frame,
            player,
            enemies,
            blasts,
            half_extents,
        }
    }

    /// Direct world access, for tests that need a hazard in an exact place.
    #[cfg(test)]
    pub(crate) fn world_mut(&mut self) -> &mut World {
        self.app.world_mut()
    }
}

#[expect(
    clippy::needless_pass_by_value,
    reason = "Bevy injects system parameters by value"
)]
fn simulation_running(gate: Res<SimulationGate>) -> bool {
    gate.0
}

/// Count enemies marked for death by enemy-on-enemy collision this frame.
///
/// Read between Deaths and Contacts: the enemy that reaches the player is
/// marked later, in Effects, so it is not counted here. Population size cannot
/// measure this, because replacements spawn in the same update.
fn count_enemy_deaths(
    dying: Query<(), (With<Enemy>, With<Dying>)>,
    mut counters: ResMut<FrameCounters>,
) {
    counters.enemy_deaths = u32::try_from(dying.iter().count()).unwrap_or(u32::MAX);
}

/// Record whether the controlled player was hit this frame.
fn record_player_hits(
    mut hits: MessageReader<PlayerHit>,
    players: Query<(), With<Player>>,
    mut counters: ResMut<FrameCounters>,
) {
    for hit in hits.read() {
        if players.contains(hit.target) {
            counters.hit = true;
        }
    }
}

/// Keep every schedule on this thread: an arena is owned by one worker, and a
/// pool per arena would multiply threads by the number of arenas.
fn single_threaded(app: &mut App) {
    for label in [
        First.intern(),
        PreUpdate.intern(),
        Update.intern(),
        PostUpdate.intern(),
        Last.intern(),
        Startup.intern(),
    ] {
        app.edit_schedule(label, |schedule| {
            schedule.set_executor(SingleThreadedExecutor::new());
        });
    }
}

#[cfg(test)]
mod tests;
