//! Site, network, and version rotation.
//!
//! Rewrites `site_name`, `network_name`, and `internal_version` to
//! plausible, real-world-looking values. Each distinct input maps to
//! the same output for the lifetime of the rotator so a single stream
//! does not report different "sites" on each hand.
use std::collections::HashMap;

use rand::prelude::SliceRandom;
use rand::rngs::StdRng;

/// Well-known poker operators, used as the replacement pool for
/// `site_name`.
///
/// These are major public brands; using them as anonymized labels is
/// intentional — the goal of anonymization is to blend in, not to
/// invent fictional sites that would themselves be identifying.
const DEFAULT_SITES: &[&str] = &[
    "PokerStars",
    "888 Poker",
    "partypoker",
    "GGPoker",
    "Americas Cardroom",
    "Ignition Casino",
    "Bet365 Poker",
    "Unibet Poker",
    "Winamax",
    "Natural8",
    "WPT Global",
    "PokerKing",
    "BlackChip Poker",
    "Paddy Power Poker",
    "William Hill Poker",
];

/// Well-known poker networks (the layer above sites).
///
/// Many sites share a network — e.g., ACR, BlackChip, and PokerKing
/// all run on WPN — so we keep this pool separate from the site pool.
const DEFAULT_NETWORKS: &[&str] = &[
    "PokerStars",
    "888",
    "partypoker",
    "GGNetwork",
    "Winning Poker Network",
    "iPoker",
    "PaiWangLuo",
    "Chico Poker Network",
    "Winamax",
    "MPN",
];

/// Plausible client-version strings.
///
/// Intentionally generic — no site actually publishes a stable scheme
/// for these, so hand-rolled semver-ish strings are more convincing
/// than anything we could derive from the original.
const DEFAULT_VERSIONS: &[&str] = &[
    "1.0.0", "1.2.4", "1.4.7", "2.0.1", "2.3.0", "2.5.9", "3.0.0", "3.1.2", "3.4.7", "4.0.0",
    "4.2.1", "5.1.3", "2024.1", "2024.3", "2024.4", "2025.1", "2025.2",
];

/// Rewrites site, network, and version fields.
///
/// Each rotated field is assigned a replacement on first sight and
/// then returns the same replacement for every subsequent occurrence
/// in the stream.
pub struct SiteRotator {
    site_pool: Vec<String>,
    network_pool: Vec<String>,
    version_pool: Vec<String>,
    site_cursor: usize,
    network_cursor: usize,
    version_cursor: usize,
    sites: HashMap<String, String>,
    networks: HashMap<String, String>,
    versions: HashMap<String, String>,
    rotate_site: bool,
    rotate_network: bool,
    rotate_version: bool,
}

impl SiteRotator {
    /// Create a rotator driven by `rng`.
    ///
    /// The three `rotate_*` flags control whether each field is
    /// replaced; when all three are false the rotator is a no-op and
    /// can be cheaply skipped.
    pub fn new(
        rotate_site: bool,
        rotate_network: bool,
        rotate_version: bool,
        rng: &mut StdRng,
    ) -> Self {
        Self {
            site_pool: shuffled(DEFAULT_SITES, rng),
            network_pool: shuffled(DEFAULT_NETWORKS, rng),
            version_pool: shuffled(DEFAULT_VERSIONS, rng),
            site_cursor: 0,
            network_cursor: 0,
            version_cursor: 0,
            sites: HashMap::new(),
            networks: HashMap::new(),
            versions: HashMap::new(),
            rotate_site,
            rotate_network,
            rotate_version,
        }
    }

    /// `true` when at least one of the three fields is being rewritten.
    pub fn is_active(&self) -> bool {
        self.rotate_site || self.rotate_network || self.rotate_version
    }

    /// Return the replacement for `site`, or the original if site
    /// rotation is disabled.
    pub fn map_site(&mut self, site: &str) -> String {
        if !self.rotate_site {
            return site.to_string();
        }
        assign(
            &mut self.sites,
            &mut self.site_pool,
            &mut self.site_cursor,
            site,
            "Site",
        )
    }

    /// Return the replacement for `network`.
    pub fn map_network(&mut self, network: &str) -> String {
        if !self.rotate_network {
            return network.to_string();
        }
        assign(
            &mut self.networks,
            &mut self.network_pool,
            &mut self.network_cursor,
            network,
            "Network",
        )
    }

    /// Return the replacement for `version`.
    pub fn map_version(&mut self, version: &str) -> String {
        if !self.rotate_version {
            return version.to_string();
        }
        assign(
            &mut self.versions,
            &mut self.version_pool,
            &mut self.version_cursor,
            version,
            "v",
        )
    }
}

/// Shuffled clone of a `&'static [&'static str]` into owned `String`s.
fn shuffled(pool: &[&str], rng: &mut StdRng) -> Vec<String> {
    let mut out: Vec<String> = pool.iter().map(|s| (*s).to_string()).collect();
    out.shuffle(rng);
    out
}

/// Look up `original` in `table`, inserting a new replacement from
/// `pool` (or a fallback of `fallback_prefix{N}`) on first sight.
///
/// Shared by all three rotated fields because the pattern is
/// identical.
fn assign(
    table: &mut HashMap<String, String>,
    pool: &mut [String],
    cursor: &mut usize,
    original: &str,
    fallback_prefix: &str,
) -> String {
    if let Some(v) = table.get(original) {
        return v.clone();
    }
    let replacement = if *cursor < pool.len() {
        let v = pool[*cursor].clone();
        *cursor += 1;
        v
    } else {
        let v = format!("{fallback_prefix}{}", table.len() + 1);
        v
    };
    table.insert(original.to_string(), replacement.clone());
    replacement
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn rng() -> StdRng {
        StdRng::seed_from_u64(7)
    }

    #[test]
    fn rotator_is_inactive_when_all_flags_false() {
        let r = SiteRotator::new(false, false, false, &mut rng());
        assert!(!r.is_active());
    }

    #[test]
    fn disabled_rotation_is_identity() {
        let mut r = SiteRotator::new(false, false, false, &mut rng());
        assert_eq!(r.map_site("X"), "X");
        assert_eq!(r.map_network("Y"), "Y");
        assert_eq!(r.map_version("Z"), "Z");
    }

    #[test]
    fn same_input_yields_same_replacement() {
        let mut r = SiteRotator::new(true, true, true, &mut rng());
        let a = r.map_site("real");
        let b = r.map_site("real");
        assert_eq!(a, b);
    }

    #[test]
    fn distinct_inputs_yield_distinct_replacements() {
        let mut r = SiteRotator::new(true, false, false, &mut rng());
        assert_ne!(r.map_site("a"), r.map_site("b"));
    }

    #[test]
    fn exhausted_pool_falls_back_to_prefixed_counter() {
        let mut r = SiteRotator::new(true, false, false, &mut rng());
        // Consume entire pool
        for i in 0..DEFAULT_SITES.len() {
            r.map_site(&format!("site{i}"));
        }
        let extra = r.map_site("extra");
        assert!(extra.starts_with("Site"));
    }
}
