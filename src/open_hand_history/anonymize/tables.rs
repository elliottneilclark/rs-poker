//! Table name rotation.
//!
//! Replaces `table_name` and `table_handle` with consistent,
//! neutral-sounding names drawn from a pool of mythological and
//! celestial labels common across real poker sites.
use std::collections::HashMap;

use rand::prelude::SliceRandom;
use rand::rngs::StdRng;

/// Pool of neutral table names.
///
/// The mix of mythology, constellations, and gems mirrors the naming
/// schemes used by major operators (PokerStars and partypoker most
/// notably), so replacements look natural in the output.
const DEFAULT_TABLE_NAMES: &[&str] = &[
    // Greco-Roman mythology
    "Apollo",
    "Athena",
    "Hera",
    "Hermes",
    "Iris",
    "Juno",
    "Minerva",
    "Neptune",
    "Orion",
    "Poseidon",
    "Venus",
    "Vulcan",
    "Zeus",
    "Artemis",
    "Hestia",
    "Demeter",
    "Hades",
    "Helios",
    "Jason",
    "Perseus",
    // Constellations
    "Andromeda",
    "Cassiopeia",
    "Draco",
    "Lyra",
    "Pegasus",
    "Phoenix",
    "Sagittarius",
    "Taurus",
    "Ursa",
    "Vega",
    "Altair",
    "Antares",
    "Betelgeuse",
    "Cygnus",
    "Rigel",
    // Gems & minerals
    "Amethyst",
    "Beryl",
    "Citrine",
    "Diamond",
    "Emerald",
    "Garnet",
    "Jade",
    "Kunzite",
    "Lapis",
    "Malachite",
    "Onyx",
    "Opal",
    "Pearl",
    "Quartz",
    "Ruby",
    "Sapphire",
    "Topaz",
    "Turquoise",
    "Zircon",
    // Northern cities
    "Oslo",
    "Bergen",
    "Helsinki",
    "Reykjavik",
    "Tallinn",
    "Riga",
    "Vilnius",
    "Gdansk",
    "Tromso",
    "Uppsala",
];

/// Rewrites table identifiers. Stable within the stream.
pub struct TableNamer {
    pool: Vec<String>,
    cursor: usize,
    table_names: HashMap<String, String>,
    table_handles: HashMap<String, String>,
    active: bool,
}

impl TableNamer {
    /// Create a namer. When `active` is `false`, all methods return
    /// the input unchanged.
    pub fn new(active: bool, rng: &mut StdRng) -> Self {
        let mut pool: Vec<String> = DEFAULT_TABLE_NAMES
            .iter()
            .map(|s| (*s).to_string())
            .collect();
        pool.shuffle(rng);
        Self {
            pool,
            cursor: 0,
            table_names: HashMap::new(),
            table_handles: HashMap::new(),
            active,
        }
    }

    /// Return the replacement for `name`, or the original when
    /// inactive.
    pub fn map_name(&mut self, name: &str) -> String {
        if !self.active {
            return name.to_string();
        }
        Self::lookup(
            &mut self.table_names,
            &mut self.pool,
            &mut self.cursor,
            name,
            "Table",
        )
    }

    /// Return the replacement for an `Option` table handle.
    ///
    /// Handles are separate identifiers from names in the OHH spec
    /// (though many sites duplicate them), so we keep a distinct map.
    pub fn map_handle(&mut self, handle: Option<&str>) -> Option<String> {
        if !self.active {
            return handle.map(|h| h.to_string());
        }
        handle.map(|h| {
            Self::lookup(
                &mut self.table_handles,
                &mut self.pool,
                &mut self.cursor,
                h,
                "Handle",
            )
        })
    }

    fn lookup(
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
            format!("{fallback_prefix}{}", table.len() + 1)
        };
        table.insert(original.to_string(), replacement.clone());
        replacement
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn rng() -> StdRng {
        StdRng::seed_from_u64(3)
    }

    #[test]
    fn inactive_is_identity() {
        let mut t = TableNamer::new(false, &mut rng());
        assert_eq!(t.map_name("Foo"), "Foo");
        assert_eq!(t.map_handle(Some("bar")), Some("bar".to_string()));
        assert_eq!(t.map_handle(None), None);
    }

    #[test]
    fn same_input_is_stable() {
        let mut t = TableNamer::new(true, &mut rng());
        assert_eq!(t.map_name("Foo"), t.map_name("Foo"));
    }

    #[test]
    fn distinct_inputs_are_distinct() {
        let mut t = TableNamer::new(true, &mut rng());
        assert_ne!(t.map_name("Foo"), t.map_name("Bar"));
    }
}
