//! Configuration for the OHH [`Anonymizer`].
//!
//! The [`AnonymizeConfig`] struct controls which fields are rewritten
//! and how. Defaults provide a reasonable privacy baseline: names are
//! consistently renamed across the stream, site/network/version and
//! table identifiers are rotated to plausible real-world values, game
//! numbers are replaced with a sequential counter, and timestamps are
//! fuzzed with a global shift plus a small per-hand jitter.
//!
//! [`Anonymizer`]: super::Anonymizer
use std::time::Duration;

/// Strategy for replacing player names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NameStrategy {
    /// Leave player names untouched.
    Keep,
    /// Each hand gets a fresh, independent set of anonymous names.
    ///
    /// A name that appears in hand N has no relationship to the same
    /// name in hand N+1. This is the strongest privacy mode but makes
    /// it impossible to track a player across hands in the anonymized
    /// output.
    PerHand,
    /// Names are assigned on first sight and then stay stable for the
    /// rest of the stream.
    ///
    /// "Alice" maps to "SilverEagle42" once, and every later hand that
    /// contains "Alice" will also see "SilverEagle42". This preserves
    /// cross-hand player identity (useful for analysis) while hiding
    /// the real handles. The mapping is learned incrementally, so this
    /// works on streams much larger than memory would hold raw.
    Stable,
}

/// Time-fuzzing parameters.
///
/// A single random `global_shift` in `[-max_global_shift,
/// +max_global_shift]` is chosen once per [`Anonymizer`] and applied
/// to every timestamp in the stream. Each hand additionally gets an
/// independent jitter in `[-max_per_hand_jitter, +max_per_hand_jitter]`
/// added to its own `start_date_utc` so inter-hand spacing cannot be
/// used to fingerprint the original session.
///
/// [`Anonymizer`]: super::Anonymizer
#[derive(Debug, Clone)]
pub struct TimeFuzzConfig {
    /// Maximum absolute shift applied to every timestamp in the stream.
    pub max_global_shift: Duration,
    /// Maximum absolute jitter added on top of the global shift for
    /// each individual hand's `start_date_utc`.
    pub max_per_hand_jitter: Duration,
}

impl Default for TimeFuzzConfig {
    /// 30 minutes of global shift, 5 seconds of per-hand jitter.
    fn default() -> Self {
        Self {
            max_global_shift: Duration::from_secs(30 * 60),
            max_per_hand_jitter: Duration::from_secs(5),
        }
    }
}

/// Full configuration for the anonymizer.
///
/// Construct with [`AnonymizeConfig::default`] and flip the fields you
/// care about. The default enables every transformation so the hand
/// history is stripped of identifying details with a single call.
#[derive(Debug, Clone)]
pub struct AnonymizeConfig {
    /// How to replace player names. See [`NameStrategy`].
    pub name_strategy: NameStrategy,
    /// Custom pool of replacement names.
    ///
    /// When `None`, the built-in pool of curated poker handles is used.
    /// When the pool is exhausted, names are minted procedurally as
    /// `Anon{N}` regardless of strategy.
    pub name_pool: Option<Vec<String>>,
    /// Rewrite `site_name` to a plausible real-world value.
    pub rotate_site: bool,
    /// Rewrite `network_name` to a plausible real-world value.
    pub rotate_network: bool,
    /// Rewrite `internal_version` to a plausible version string.
    pub rotate_internal_version: bool,
    /// Rewrite `table_name` (and `table_handle`, if present) to
    /// plausible neutral names.
    pub rotate_table_name: bool,
    /// Replace `game_number` and `tournament_info.tournament_number`
    /// with a sequential counter anchored at a random base.
    ///
    /// Stable across the stream: the same original game number always
    /// maps to the same replacement.
    pub rotate_game_numbers: bool,
    /// When `Some`, fuzz `start_date_utc` on every hand.
    pub time_fuzz: Option<TimeFuzzConfig>,
    /// Seed for the internal RNG. `None` picks a seed from the OS.
    pub seed: Option<u64>,
}

impl Default for AnonymizeConfig {
    fn default() -> Self {
        Self {
            name_strategy: NameStrategy::Stable,
            name_pool: None,
            rotate_site: true,
            rotate_network: true,
            rotate_internal_version: true,
            rotate_table_name: true,
            rotate_game_numbers: true,
            time_fuzz: Some(TimeFuzzConfig::default()),
            seed: None,
        }
    }
}
