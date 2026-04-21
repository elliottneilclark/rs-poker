//! Player-name replacement.
//!
//! This module provides the [`NameMapper`] trait and two concrete
//! implementations that correspond to [`NameStrategy::PerHand`] and
//! [`NameStrategy::Stable`]. Both draw from a shared, shuffled pool of
//! neutral poker-style handles; when the pool is exhausted they fall
//! back to procedural `Anon{N}` names.
//!
//! [`NameStrategy::PerHand`]: super::config::NameStrategy::PerHand
//! [`NameStrategy::Stable`]: super::config::NameStrategy::Stable
use std::collections::HashMap;

use rand::prelude::SliceRandom;
use rand::rngs::StdRng;

/// Maps original player names to anonymized replacements.
///
/// Implementations may carry per-hand or per-stream state. The
/// [`Anonymizer`] always calls [`begin_hand`] before renaming the
/// players of a new hand so per-hand implementations can reset.
///
/// [`Anonymizer`]: super::Anonymizer
/// [`begin_hand`]: NameMapper::begin_hand
pub trait NameMapper {
    /// Called once at the start of every hand, before any [`map_name`]
    /// call for that hand.
    ///
    /// [`map_name`]: NameMapper::map_name
    fn begin_hand(&mut self);

    /// Return the anonymized replacement for `original`.
    ///
    /// The same input must map to the same output within a single
    /// hand; cross-hand stability depends on the strategy.
    fn map_name(&mut self, original: &str) -> String;
}

/// Curated pool of neutral, poker-style handles.
///
/// Chosen to look plausible as real screen names without suggesting
/// any particular site, locale, or persona. Kept intentionally small:
/// streams that need more unique names fall through to `Anon{N}`.
const DEFAULT_NAME_POOL: &[&str] = &[
    "SilverEagle",
    "CrimsonFox",
    "QuietRiver",
    "BlueHarbor",
    "IronOak",
    "NorthernPine",
    "AmberWolf",
    "CopperHawk",
    "GoldenKite",
    "JadeOtter",
    "VioletFinch",
    "BronzeLynx",
    "StoneHeron",
    "RustyAnchor",
    "MysticOwl",
    "ScarletBadger",
    "PaleMoon",
    "EmeraldCrow",
    "MarbleDeer",
    "HiddenVale",
    "LoneCedar",
    "MossyStone",
    "WildMeadow",
    "ThornyElm",
    "LuckyHollow",
    "BarrenRidge",
    "OpenPlains",
    "SwiftHare",
    "BrightDawn",
    "DuskRanger",
    "HollowDrum",
    "NorthStar",
    "PaperTiger",
    "SaltyHarbor",
    "SpicyReef",
    "BrassCompass",
    "CrystalLake",
    "DeepRiver",
    "FrostyPeak",
    "GlassOcean",
    "HighBluff",
    "InkyShade",
    "JollyRoger",
    "KindredPath",
    "LazyRiver",
    "MistyForest",
    "NimbleFox",
    "OldLighthouse",
    "PrismCove",
    "QuickSilver",
    "RisingTide",
    "SilentCanyon",
    "TawnyOwl",
    "UmberMesa",
    "VelvetSky",
    "WhiteSpruce",
    "XenonGlow",
    "YellowBirch",
    "ZenGarden",
    "AlderGrove",
    "BirchTrail",
    "ClayPot",
    "DuneWalker",
    "ElderWood",
    "FernLake",
    "GraniteCliff",
    "HazelHill",
    "IvyBend",
    "JasmineKey",
    "KelpGarden",
    "LimePoint",
    "MapleRidge",
    "NettleField",
    "OakHarbor",
    "PinePass",
    "QuartzCove",
    "RowanHollow",
    "SageCreek",
    "ThistlePath",
    "UrchinBay",
    "VineyardRow",
    "WillowBend",
    "AmberLantern",
    "BasaltRock",
    "CedarGrove",
    "DriftwoodBeach",
    "EmberCoal",
    "FallenLog",
    "GraniteShore",
    "HeronPoint",
    "IronGate",
    "JadeLantern",
    "KaleidoscopeSky",
    "LanternGlow",
    "MoonlitBay",
    "NeedlePoint",
    "OakenShield",
    "PebbleBeach",
    "QuillFeather",
    "RiverStone",
    "SilverMist",
    "TorchFlame",
    "UnderGrove",
    "VesperStar",
    "WhisperWind",
    "ArcticFern",
    "BoulderPass",
    "CinderPath",
    "DewDrop",
    "EclipseRidge",
    "FrostVine",
    "GlacierRun",
    "HollyGrove",
    "IndigoLake",
    "JunkyardRose",
    "KiteRunner",
    "LotusBloom",
    "MoltenCore",
    "NightFrost",
    "OrchidWisp",
    "PlumePass",
    "QuiverArrow",
    "RootedTree",
    "SeedlingPath",
    "ThunderCloud",
    "UphillClimb",
    "VerdantMeadow",
    "WanderingStar",
    "XylemFlow",
    "YonderHill",
    "ZephyrBreeze",
    "AlpineEcho",
    "BramblePatch",
    "CascadeFall",
    "DuneCrest",
    "ElmShadow",
    "FieldStone",
    "GoldenRod",
    "HarborLight",
    "IvoryGate",
    "JuniperHill",
    "KestrelKite",
    "LichenStone",
    "MistyHarbor",
    "NightOrchid",
    "OakenPath",
    "PrairieWind",
    "QuellingCalm",
    "RiverBirch",
    "SlateRock",
    "TulipGarden",
    "UmbraMoon",
    "ValleyStream",
    "WalnutGrove",
    "XanaduRest",
    "YewBranch",
    "ZebraRock",
    "AutumnLeaf",
    "BrambleWood",
    "CliffEdge",
    "DaisyChain",
    "EchoVale",
    "FlintSpark",
    "GrainField",
    "HolyOak",
    "IceBloom",
    "JadeFern",
    "KnotwoodBow",
    "LoamEarth",
    "MarshLily",
    "NorthernGale",
    "OatField",
    "PineNeedle",
    "QuietDusk",
    "RosePetal",
    "SnowDrift",
    "TinderBox",
    "UrsineTrack",
    "VoltGust",
    "WispFlame",
    "XyloRhythm",
    "YarrowRoot",
    "ZinniaBloom",
    "AshenVale",
    "BarkGrain",
    "ClearStream",
    "DriftSand",
    "EdenBloom",
    "FrostWind",
    "GoldMeadow",
    "HollowReed",
    "InkyDusk",
    "JuteCord",
    "KindleFire",
    "LatticeVine",
    "MallowField",
    "NightingaleSong",
    "OpalShore",
    "PinePitch",
    "QuailNest",
    "RustyGate",
    "SaltMarsh",
    "ThistleDown",
    "UplandMoor",
    "VesselHarbor",
    "WeatherVane",
    "AmaranthBloom",
    "BlueSpruce",
    "CloverMead",
    "DuskLark",
];

/// Provides a shuffled stream of candidate names, falling back to
/// `Anon{N}` when the curated pool runs out.
///
/// Used internally by both [`PerHandNameMapper`] and
/// [`StableNameMapper`]. Keeping the pool logic in one place ensures
/// the two strategies produce qualitatively similar outputs.
struct NamePool {
    names: Vec<String>,
    cursor: usize,
    anon_counter: u64,
}

impl NamePool {
    fn new(custom: Option<Vec<String>>, rng: &mut StdRng) -> Self {
        let mut names: Vec<String> =
            custom.unwrap_or_else(|| DEFAULT_NAME_POOL.iter().map(|s| (*s).to_string()).collect());
        names.shuffle(rng);
        Self {
            names,
            cursor: 0,
            anon_counter: 0,
        }
    }

    fn next(&mut self) -> String {
        if self.cursor < self.names.len() {
            let name = self.names[self.cursor].clone();
            self.cursor += 1;
            name
        } else {
            self.anon_counter += 1;
            format!("Anon{}", self.anon_counter)
        }
    }
}

/// Per-hand name mapper: the mapping table is cleared at the start of
/// every hand, so two hands never share a replacement.
pub struct PerHandNameMapper {
    pool: NamePool,
    current: HashMap<String, String>,
}

impl PerHandNameMapper {
    /// Create a new per-hand mapper.
    ///
    /// `custom_pool` overrides the built-in pool; `rng` seeds the
    /// shuffle so two mappers with the same seed produce the same
    /// sequence.
    pub fn new(custom_pool: Option<Vec<String>>, rng: &mut StdRng) -> Self {
        Self {
            pool: NamePool::new(custom_pool, rng),
            current: HashMap::new(),
        }
    }
}

impl NameMapper for PerHandNameMapper {
    fn begin_hand(&mut self) {
        self.current.clear();
    }

    fn map_name(&mut self, original: &str) -> String {
        if let Some(name) = self.current.get(original) {
            return name.clone();
        }
        let name = self.pool.next();
        self.current.insert(original.to_string(), name.clone());
        name
    }
}

/// Stream-stable name mapper: every player gets a replacement on first
/// sight and keeps it for the rest of the stream.
///
/// Memory usage grows with the number of distinct original names seen,
/// not with the number of hands, so this scales to arbitrarily large
/// JSONL inputs as long as the cardinality of players is bounded.
pub struct StableNameMapper {
    pool: NamePool,
    assigned: HashMap<String, String>,
}

impl StableNameMapper {
    /// Create a new stable mapper. See [`PerHandNameMapper::new`].
    pub fn new(custom_pool: Option<Vec<String>>, rng: &mut StdRng) -> Self {
        Self {
            pool: NamePool::new(custom_pool, rng),
            assigned: HashMap::new(),
        }
    }
}

impl NameMapper for StableNameMapper {
    fn begin_hand(&mut self) {}

    fn map_name(&mut self, original: &str) -> String {
        if let Some(name) = self.assigned.get(original) {
            return name.clone();
        }
        let name = self.pool.next();
        self.assigned.insert(original.to_string(), name.clone());
        name
    }
}

/// Pass-through mapper that leaves names unchanged.
///
/// Used when [`NameStrategy::Keep`] is selected; centralising this
/// variant as a `NameMapper` keeps the [`Anonymizer`] free of `if`
/// branches over the strategy.
///
/// [`NameStrategy::Keep`]: super::config::NameStrategy::Keep
/// [`Anonymizer`]: super::Anonymizer
pub struct KeepNameMapper;

impl NameMapper for KeepNameMapper {
    fn begin_hand(&mut self) {}

    fn map_name(&mut self, original: &str) -> String {
        original.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn rng() -> StdRng {
        StdRng::seed_from_u64(42)
    }

    #[test]
    fn stable_mapper_gives_same_name_across_calls() {
        let mut m = StableNameMapper::new(None, &mut rng());
        m.begin_hand();
        let a1 = m.map_name("alice");
        m.begin_hand();
        let a2 = m.map_name("alice");
        assert_eq!(a1, a2);
    }

    #[test]
    fn per_hand_mapper_resets_between_hands() {
        let mut m = PerHandNameMapper::new(None, &mut rng());
        m.begin_hand();
        let a1 = m.map_name("alice");
        m.begin_hand();
        let a2 = m.map_name("alice");
        // Two independent hands are very unlikely to coincidentally
        // draw the same slot given a 200-entry shuffled pool.
        assert_ne!(a1, a2);
    }

    #[test]
    fn distinct_originals_get_distinct_replacements() {
        let mut m = StableNameMapper::new(None, &mut rng());
        m.begin_hand();
        assert_ne!(m.map_name("alice"), m.map_name("bob"));
    }

    #[test]
    fn exhausting_pool_falls_back_to_anon_counter() {
        let custom = vec!["only".to_string()];
        let mut m = StableNameMapper::new(Some(custom), &mut rng());
        m.begin_hand();
        assert_eq!(m.map_name("first"), "only");
        assert_eq!(m.map_name("second"), "Anon1");
        assert_eq!(m.map_name("third"), "Anon2");
    }

    #[test]
    fn seed_produces_deterministic_output() {
        let mut a = StableNameMapper::new(None, &mut StdRng::seed_from_u64(1));
        let mut b = StableNameMapper::new(None, &mut StdRng::seed_from_u64(1));
        a.begin_hand();
        b.begin_hand();
        assert_eq!(a.map_name("alice"), b.map_name("alice"));
    }

    #[test]
    fn keep_mapper_is_identity() {
        let mut m = KeepNameMapper;
        m.begin_hand();
        assert_eq!(m.map_name("alice"), "alice");
    }
}
