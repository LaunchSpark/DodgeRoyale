//! Seeded randomness, split into independent per-domain streams.
//!
//! One [`GameSeed`] fixes a whole run. Each domain derives its own stream from
//! it, so the number of draws one domain makes never shifts another's sequence.
//! Without that separation, adding a particle burst would change every later
//! enemy spawn and a recorded seed would stop reproducing its run.

use bevy::prelude::*;

use crate::enemy_population::EnemySpawnQueue;

/// A part of the game that draws randomness on its own schedule.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Domain {
    /// Choosing and placing enemies.
    Spawn,
    /// Scattering pieces when an enemy shatters.
    Shatter,
}

impl Domain {
    /// A fixed, arbitrary constant that separates this domain from the others.
    const fn salt(self) -> u64 {
        match self {
            Self::Spawn => 0x5350_4157_4e5f_3031,
            Self::Shatter => 0x5348_4154_5445_5230,
        }
    }
}

/// The single seed every stream in a run derives from.
#[derive(Resource, Clone, Copy, PartialEq, Eq, Debug)]
pub struct GameSeed(u64);

impl Default for GameSeed {
    /// Ordinary play varies; pass an explicit seed to replay a run.
    fn default() -> Self {
        Self::from_entropy()
    }
}

impl GameSeed {
    /// Fix a run to an exact seed.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// Draw an unpredictable seed, for ordinary play.
    #[must_use]
    pub fn from_entropy() -> Self {
        Self(fastrand::u64(..))
    }

    /// The seed itself, for logging or reporting back to a player.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }

    /// The derived seed for one domain's stream.
    #[must_use]
    pub const fn stream(self, domain: Domain) -> u64 {
        mix(self.0 ^ mix(domain.salt()))
    }

    /// A fresh generator for one domain, positioned at the start of its stream.
    #[must_use]
    pub const fn rng(self, domain: Domain) -> fastrand::Rng {
        fastrand::Rng::with_seed(self.stream(domain))
    }
}

/// Installs the run's seed and points every stream at it.
///
/// Resources default to entropy when their own plugin initialises them, so this
/// reseeds them from [`GameSeed`] before the first frame.
pub struct SeededRngPlugin;

impl Plugin for SeededRngPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<GameSeed>()
            .add_systems(PreStartup, seed_spawn_queue);
    }
}

/// Point enemy spawning at this run's spawn stream.
#[expect(
    clippy::needless_pass_by_value,
    reason = "Bevy injects system parameters by value"
)]
fn seed_spawn_queue(seed: Res<GameSeed>, mut queue: ResMut<EnemySpawnQueue>) {
    *queue = EnemySpawnQueue::with_seed(seed.stream(Domain::Spawn));
}

/// The splitmix64 finaliser, which spreads neighbouring inputs across the range.
///
/// Seeds 1 and 2 would otherwise open their streams with near-identical values.
const fn mix(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Take a short sample from one domain's stream.
    fn sample(seed: u64, domain: Domain) -> Vec<u32> {
        let mut rng = GameSeed::new(seed).rng(domain);
        (0..8).map(|_| rng.u32(..)).collect()
    }

    #[test]
    fn the_same_seed_and_domain_always_derive_the_same_stream() {
        let seed = GameSeed::new(12_345);
        assert_eq!(seed.stream(Domain::Spawn), seed.stream(Domain::Spawn));
        assert_eq!(sample(12_345, Domain::Spawn), sample(12_345, Domain::Spawn));
    }

    #[test]
    fn different_seeds_produce_different_streams() {
        assert_ne!(sample(1, Domain::Spawn), sample(2, Domain::Spawn));
    }

    #[test]
    fn domains_of_one_seed_are_independent() {
        let seed = GameSeed::new(12_345);
        assert_ne!(seed.stream(Domain::Spawn), seed.stream(Domain::Shatter));
        assert_ne!(
            sample(12_345, Domain::Spawn),
            sample(12_345, Domain::Shatter)
        );
    }

    #[test]
    fn exhausting_one_domain_leaves_another_untouched() {
        let seed = GameSeed::new(7);
        let before = sample(7, Domain::Shatter);
        let mut spawn = seed.rng(Domain::Spawn);
        for _ in 0..1_000 {
            spawn.u32(..);
        }
        assert_eq!(before, sample(7, Domain::Shatter));
    }

    #[test]
    fn neighbouring_seeds_are_thoroughly_mixed() {
        // Without mixing, seeds 0, 1, 2 would open with near-identical values.
        let openings: Vec<u32> = (0..3)
            .map(|seed| GameSeed::new(seed).rng(Domain::Spawn).u32(..))
            .collect();
        assert_ne!(openings.first(), openings.get(1));
        assert_ne!(openings.get(1), openings.get(2));
    }

    #[test]
    fn a_seed_survives_a_round_trip() {
        assert_eq!(GameSeed::new(42).get(), 42);
    }

    #[test]
    fn entropy_seeds_vary_between_runs() {
        let seeds: Vec<u64> = (0..4).map(|_| GameSeed::from_entropy().get()).collect();
        assert!(
            seeds.iter().any(|value| Some(value) != seeds.first()),
            "entropy seeding produced four identical seeds"
        );
    }
}
