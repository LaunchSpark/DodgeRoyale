//! Player-facing settings, mirroring the reference cart's own options.
//!
//! The theme and the enemy count both take effect. Difficulty and powerups are
//! stored faithfully, with the cart's tuning values attached, but nothing reads
//! them yet. The config screen marks those rows as unimplemented rather than
//! pretending otherwise.

/// The largest enemy count the config screen accepts.
pub const MAX_ENEMIES: usize = 999;

/// How many themes the reference cart offers.
///
/// The palette itself is presentation, so it lives elsewhere; a test there keeps
/// this count honest.
pub const THEME_COUNT: usize = 13;

/// The three difficulty levels, carrying the cart's tuning multipliers.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Difficulty {
    Easy,
    #[default]
    Normal,
    Hard,
}

impl Difficulty {
    /// The name shown on the config screen.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Easy => "EASY",
            Self::Normal => "NORMAL",
            Self::Hard => "HARD",
        }
    }

    /// Enemy speed multiplier, from the cart's `startspd` table.
    #[must_use]
    pub const fn enemy_speed(self) -> f32 {
        match self {
            Self::Easy => 0.8,
            Self::Normal => 1.0,
            Self::Hard => 1.6,
        }
    }

    /// Spawn interval multiplier, from the cart's `startest` table.
    #[must_use]
    pub const fn spawn_interval(self) -> f32 {
        match self {
            Self::Easy => 1.4,
            Self::Normal => 1.0,
            Self::Hard => 0.8,
        }
    }

    /// The next level, wrapping from hard back to easy.
    #[must_use]
    pub const fn next(self) -> Self {
        match self {
            Self::Easy => Self::Normal,
            Self::Normal => Self::Hard,
            Self::Hard => Self::Easy,
        }
    }

    /// The previous level, wrapping from easy back to hard.
    #[must_use]
    pub const fn previous(self) -> Self {
        match self {
            Self::Easy => Self::Hard,
            Self::Normal => Self::Easy,
            Self::Hard => Self::Normal,
        }
    }
}

/// Everything the config screen can change.
#[derive(Clone, Copy, Debug)]
pub struct Settings {
    /// Index into [`THEMES`].
    pub theme: usize,
    pub difficulty: Difficulty,
    /// Enemies the arena is kept stocked with.
    pub starting_enemies: usize,
    pub powerups: bool,
}

impl Default for Settings {
    /// The cart starts on the blue theme at normal difficulty, everything on; the
    /// enemy count is raised from the cart's 24 to 100.
    fn default() -> Self {
        Self {
            theme: 0,
            difficulty: Difficulty::Normal,
            starting_enemies: 100,
            powerups: true,
        }
    }
}

impl Settings {
    /// Step to the next or previous theme, wrapping at both ends.
    pub const fn cycle_theme(&mut self, forward: bool) {
        self.theme = if forward {
            // THEME_COUNT is a non-zero constant, so this cannot divide by zero.
            self.theme.saturating_add(1) % THEME_COUNT
        } else if self.theme == 0 {
            THEME_COUNT.saturating_sub(1)
        } else {
            self.theme.saturating_sub(1)
        };
    }

    /// Append a typed digit, ignoring anything that would exceed [`MAX_ENEMIES`].
    pub fn push_enemy_digit(&mut self, digit: u32) {
        let Ok(digit) = usize::try_from(digit) else {
            return;
        };
        if digit > 9 {
            return;
        }
        let next = self
            .starting_enemies
            .saturating_mul(10)
            .saturating_add(digit);
        if next <= MAX_ENEMIES {
            self.starting_enemies = next;
        }
    }

    /// Remove the last typed digit.
    pub const fn pop_enemy_digit(&mut self) {
        self.starting_enemies /= 10;
    }

    /// Step the count by one, staying within zero and [`MAX_ENEMIES`].
    pub const fn bump_enemies(&mut self, forward: bool) {
        self.starting_enemies = if forward {
            let next = self.starting_enemies.saturating_add(1);
            if next > MAX_ENEMIES {
                MAX_ENEMIES
            } else {
                next
            }
        } else {
            self.starting_enemies.saturating_sub(1)
        };
    }

    /// Replace the count, clamped to [`MAX_ENEMIES`].
    pub const fn set_enemies(&mut self, value: usize) {
        self.starting_enemies = if value > MAX_ENEMIES {
            MAX_ENEMIES
        } else {
            value
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn difficulty_tuning_matches_the_reference_cart() {
        // The cart's startspd and startest tables, indexed easy, normal, hard.
        for (level, speed, interval) in [
            (Difficulty::Easy, 0.8, 1.4),
            (Difficulty::Normal, 1.0, 1.0),
            (Difficulty::Hard, 1.6, 0.8),
        ] {
            assert!((level.enemy_speed() - speed).abs() < f32::EPSILON);
            assert!((level.spawn_interval() - interval).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn difficulty_cycles_in_both_directions_and_wraps() {
        assert_eq!(Difficulty::Easy.next(), Difficulty::Normal);
        assert_eq!(Difficulty::Normal.next(), Difficulty::Hard);
        assert_eq!(Difficulty::Hard.next(), Difficulty::Easy);
        assert_eq!(Difficulty::Easy.previous(), Difficulty::Hard);
    }

    #[test]
    fn defaults_match_the_reference_cart_except_enemy_count() {
        let settings = Settings::default();
        assert_eq!(settings.theme, 0);
        assert_eq!(settings.difficulty, Difficulty::Normal);
        assert_eq!(settings.starting_enemies, 100);
        assert!(settings.powerups);
    }

    #[test]
    fn theme_cycling_wraps_across_every_theme() {
        let mut settings = Settings::default();
        for _ in 0..THEME_COUNT {
            settings.cycle_theme(true);
        }
        assert_eq!(settings.theme, 0, "a full cycle returns to the first theme");
        settings.cycle_theme(false);
        assert_eq!(
            settings.theme,
            THEME_COUNT - 1,
            "stepping back wraps to the last"
        );
    }

    #[test]
    fn typing_digits_appends_and_stops_at_the_cap() {
        let mut settings = Settings::default();
        settings.set_enemies(0);
        for digit in [1_u32, 2, 3] {
            settings.push_enemy_digit(digit);
        }
        assert_eq!(settings.starting_enemies, 123);
        // A fourth digit would exceed the cap, so the keystroke is ignored.
        settings.push_enemy_digit(9);
        assert_eq!(settings.starting_enemies, 123);
        assert_eq!(MAX_ENEMIES, 999);
    }

    #[test]
    fn backspace_removes_the_last_digit() {
        let mut settings = Settings::default();
        settings.set_enemies(123);
        settings.pop_enemy_digit();
        assert_eq!(settings.starting_enemies, 12);
        settings.pop_enemy_digit();
        settings.pop_enemy_digit();
        assert_eq!(settings.starting_enemies, 0);
        // Removing from zero is harmless.
        settings.pop_enemy_digit();
        assert_eq!(settings.starting_enemies, 0);
    }

    #[test]
    fn bumping_stays_inside_the_allowed_range() {
        let mut settings = Settings::default();
        settings.set_enemies(0);
        settings.bump_enemies(false);
        assert_eq!(settings.starting_enemies, 0, "cannot go below zero");
        settings.bump_enemies(true);
        assert_eq!(settings.starting_enemies, 1);
        settings.set_enemies(MAX_ENEMIES);
        settings.bump_enemies(true);
        assert_eq!(
            settings.starting_enemies, MAX_ENEMIES,
            "cannot exceed the cap"
        );
    }

    #[test]
    fn setting_a_value_clamps_to_the_cap() {
        let mut settings = Settings::default();
        settings.set_enemies(100_000);
        assert_eq!(settings.starting_enemies, MAX_ENEMIES);
    }
}
