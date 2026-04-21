//! Stream-stable rotation for game numbers and tournament identifiers.
//!
//! Unlike site/network rotation, these fields don't have a natural
//! dictionary of replacement values — they're just opaque strings (a
//! game number is usually a large integer as text). We replace them
//! with a sequential counter anchored at a random base so that
//! cross-references between hands survive but the absolute values
//! cannot be traced back to the original site's records.
use std::collections::HashMap;

use rand::RngExt;
use rand::rngs::StdRng;

/// Rotates game-number-like fields ("12345678" → "74910001",
/// "74910002", ...). The base is randomized once; subsequent
/// assignments are purely sequential.
pub struct GameNumberRotator {
    rotate: bool,
    base: u64,
    counter: u64,
    mapping: HashMap<String, String>,
}

impl GameNumberRotator {
    /// Create a rotator. When `rotate` is `false`, [`map`] returns its
    /// argument unchanged.
    ///
    /// The random base is drawn from `[10^7, 10^8)` so the generated
    /// numbers resemble real-world game numbers in magnitude.
    ///
    /// [`map`]: GameNumberRotator::map
    pub fn new(rotate: bool, rng: &mut StdRng) -> Self {
        let base = if rotate {
            rng.random_range(10_000_000u64..100_000_000u64)
        } else {
            0
        };
        Self {
            rotate,
            base,
            counter: 0,
            mapping: HashMap::new(),
        }
    }

    /// Map an original identifier to its sequential replacement.
    pub fn map(&mut self, original: &str) -> String {
        if !self.rotate {
            return original.to_string();
        }
        if let Some(v) = self.mapping.get(original) {
            return v.clone();
        }
        self.counter += 1;
        let replacement = (self.base + self.counter).to_string();
        self.mapping
            .insert(original.to_string(), replacement.clone());
        replacement
    }
}

/// Rotates tournament-name strings like "Sunday Million" or "Daily
/// $50K GTD" into opaque "Anonymous Tournament N" labels.
///
/// Stable within the stream: every hand that belongs to the same
/// original tournament continues to share a replacement name.
pub struct TournamentNameRotator {
    rotate: bool,
    counter: u64,
    mapping: HashMap<String, String>,
}

impl TournamentNameRotator {
    /// Create a rotator. When `rotate` is `false`, [`map`] returns its
    /// argument unchanged.
    ///
    /// [`map`]: TournamentNameRotator::map
    pub fn new(rotate: bool) -> Self {
        Self {
            rotate,
            counter: 0,
            mapping: HashMap::new(),
        }
    }

    /// Map an original tournament name to a sequential replacement.
    pub fn map(&mut self, original: &str) -> String {
        if !self.rotate {
            return original.to_string();
        }
        if let Some(v) = self.mapping.get(original) {
            return v.clone();
        }
        self.counter += 1;
        let replacement = format!("Anonymous Tournament {}", self.counter);
        self.mapping
            .insert(original.to_string(), replacement.clone());
        replacement
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn rng() -> StdRng {
        StdRng::seed_from_u64(99)
    }

    #[test]
    fn disabled_game_rotator_is_identity() {
        let mut g = GameNumberRotator::new(false, &mut rng());
        assert_eq!(g.map("42"), "42");
    }

    #[test]
    fn enabled_game_rotator_is_stable_and_distinct() {
        let mut g = GameNumberRotator::new(true, &mut rng());
        let a = g.map("100");
        let b = g.map("100");
        let c = g.map("200");
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn game_rotator_base_is_plausible_size() {
        let mut g = GameNumberRotator::new(true, &mut rng());
        let out: u64 = g.map("first").parse().unwrap();
        assert!(out >= 10_000_000);
    }

    #[test]
    fn disabled_tournament_rotator_is_identity() {
        let mut t = TournamentNameRotator::new(false);
        assert_eq!(t.map("Sunday Million"), "Sunday Million");
    }

    #[test]
    fn enabled_tournament_rotator_is_stable_and_distinct() {
        let mut t = TournamentNameRotator::new(true);
        let a = t.map("Sunday Million");
        let b = t.map("Sunday Million");
        let c = t.map("Nightly Hundred");
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert!(a.starts_with("Anonymous Tournament"));
    }
}
