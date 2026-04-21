//! The top-level [`Anonymizer`] that composes every transformation.
//!
//! [`Anonymizer`] owns an RNG, a [`NameMapper`], a [`SiteRotator`], a
//! [`TableNamer`], optional [`TimeFuzzer`], and identifier rotators,
//! and exposes a single [`Anonymizer::anonymize`] method that mutates
//! a [`HandHistory`] in place. Transforming in place lets callers
//! stream hands through without extra allocations.
//!
//! # Example
//!
//! ```
//! use rs_poker::open_hand_history::anonymize::{
//!     AnonymizeConfig, Anonymizer, NameStrategy,
//! };
//!
//! let config = AnonymizeConfig {
//!     name_strategy: NameStrategy::Stable,
//!     seed: Some(42),
//!     ..AnonymizeConfig::default()
//! };
//! let mut anonymizer = Anonymizer::new(config);
//! # let mut hand = rs_poker::open_hand_history::HandHistory {
//! #     spec_version: "1.4.7".into(), site_name: "Real".into(),
//! #     network_name: "Real".into(), internal_version: "1.0".into(),
//! #     tournament: false, tournament_info: None,
//! #     game_number: "1".into(), start_date_utc: None,
//! #     table_name: "t".into(), table_handle: None, table_skin: None,
//! #     game_type: rs_poker::open_hand_history::GameType::Holdem,
//! #     bet_limit: None, table_size: 2, currency: "USD".into(),
//! #     dealer_seat: 1, small_blind_amount: 1.0, big_blind_amount: 2.0,
//! #     ante_amount: 0.0, hero_player_id: None,
//! #     players: vec![], rounds: vec![], pots: vec![],
//! #     tournament_bounties: None,
//! # };
//! anonymizer.anonymize(&mut hand);
//! ```
use rand::SeedableRng;
use rand::rngs::StdRng;

use super::config::{AnonymizeConfig, NameStrategy};
use super::identifiers::{GameNumberRotator, TournamentNameRotator};
use super::names::{KeepNameMapper, NameMapper, PerHandNameMapper, StableNameMapper};
use super::sites::SiteRotator;
use super::tables::TableNamer;
use super::times::TimeFuzzer;
use crate::open_hand_history::HandHistory;

/// Stateful transformer that rewrites identifying fields of a
/// [`HandHistory`] in place.
///
/// A single `Anonymizer` is intended to process a whole stream of
/// related hands: state like the stable name map, the site rotation
/// table, and the global time shift persists across calls to
/// [`anonymize`](Anonymizer::anonymize).
pub struct Anonymizer {
    rng: StdRng,
    name_mapper: Box<dyn NameMapper>,
    site_rotator: SiteRotator,
    table_namer: TableNamer,
    time_fuzzer: Option<TimeFuzzer>,
    game_numbers: GameNumberRotator,
    tournament_names: TournamentNameRotator,
}

impl Anonymizer {
    /// Build a new anonymizer from a configuration.
    pub fn new(config: AnonymizeConfig) -> Self {
        let mut rng = match config.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_rng(&mut rand::rng()),
        };

        let name_mapper: Box<dyn NameMapper> = match config.name_strategy {
            NameStrategy::Keep => Box::new(KeepNameMapper),
            NameStrategy::PerHand => {
                Box::new(PerHandNameMapper::new(config.name_pool.clone(), &mut rng))
            }
            NameStrategy::Stable => {
                Box::new(StableNameMapper::new(config.name_pool.clone(), &mut rng))
            }
        };

        let site_rotator = SiteRotator::new(
            config.rotate_site,
            config.rotate_network,
            config.rotate_internal_version,
            &mut rng,
        );
        let table_namer = TableNamer::new(config.rotate_table_name, &mut rng);
        let time_fuzzer = config
            .time_fuzz
            .as_ref()
            .map(|c| TimeFuzzer::new(c.max_global_shift, c.max_per_hand_jitter, &mut rng));
        let game_numbers = GameNumberRotator::new(config.rotate_game_numbers, &mut rng);
        let tournament_names = TournamentNameRotator::new(config.rotate_game_numbers);

        Self {
            rng,
            name_mapper,
            site_rotator,
            table_namer,
            time_fuzzer,
            game_numbers,
            tournament_names,
        }
    }

    /// Rewrite the identifying fields of `hand` in place.
    ///
    /// Calling this with every hand from a JSONL stream is the
    /// standard usage; see [`anonymize_stream`][super::anonymize_stream]
    /// for a ready-made streaming driver.
    pub fn anonymize(&mut self, hand: &mut HandHistory) {
        self.rewrite_site(hand);
        self.rewrite_table(hand);
        self.rewrite_game_number(hand);
        self.rewrite_tournament(hand);
        self.rewrite_start_date(hand);
        self.rewrite_players(hand);
    }

    fn rewrite_site(&mut self, hand: &mut HandHistory) {
        if !self.site_rotator.is_active() {
            return;
        }
        hand.site_name = self.site_rotator.map_site(&hand.site_name);
        hand.network_name = self.site_rotator.map_network(&hand.network_name);
        hand.internal_version = self.site_rotator.map_version(&hand.internal_version);
        if let Some(skin) = hand.table_skin.as_deref() {
            hand.table_skin = Some(self.site_rotator.map_site(skin));
        }
    }

    fn rewrite_table(&mut self, hand: &mut HandHistory) {
        hand.table_name = self.table_namer.map_name(&hand.table_name);
        hand.table_handle = self.table_namer.map_handle(hand.table_handle.as_deref());
    }

    fn rewrite_game_number(&mut self, hand: &mut HandHistory) {
        hand.game_number = self.game_numbers.map(&hand.game_number);
    }

    fn rewrite_tournament(&mut self, hand: &mut HandHistory) {
        let Some(info) = hand.tournament_info.as_mut() else {
            return;
        };
        info.tournament_number = self.game_numbers.map(&info.tournament_number);
        info.name = self.tournament_names.map(&info.name);
        if let Some(fuzzer) = &self.time_fuzzer {
            info.start_date_utc = fuzzer.shift_only(info.start_date_utc);
        }
    }

    fn rewrite_start_date(&mut self, hand: &mut HandHistory) {
        if let Some(fuzzer) = &self.time_fuzzer {
            hand.start_date_utc = fuzzer.shift_with_jitter(hand.start_date_utc, &mut self.rng);
        }
    }

    fn rewrite_players(&mut self, hand: &mut HandHistory) {
        self.name_mapper.begin_hand();
        for player in &mut hand.players {
            player.name = self.name_mapper.map_name(&player.name);
            if let Some(display) = player.display.as_ref() {
                player.display = Some(self.name_mapper.map_name(display));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::open_hand_history::{GameType, HandHistory, PlayerObj};
    use chrono::{DateTime, Utc};

    fn sample_hand() -> HandHistory {
        HandHistory {
            spec_version: "1.4.7".into(),
            site_name: "RealSite".into(),
            network_name: "RealNet".into(),
            internal_version: "9.9.9".into(),
            tournament: false,
            tournament_info: None,
            game_number: "555000001".into(),
            start_date_utc: Some(
                DateTime::parse_from_rfc3339("2024-05-01T12:00:00Z")
                    .unwrap()
                    .with_timezone(&Utc),
            ),
            table_name: "RealTable".into(),
            table_handle: Some("Handle".into()),
            table_skin: Some("RealSite.it".into()),
            game_type: GameType::Holdem,
            bet_limit: None,
            table_size: 2,
            currency: "USD".into(),
            dealer_seat: 1,
            small_blind_amount: 1.0,
            big_blind_amount: 2.0,
            ante_amount: 0.0,
            hero_player_id: Some(1),
            players: vec![
                PlayerObj {
                    id: 1,
                    seat: 1,
                    name: "Alice".into(),
                    display: Some("AliceDisplay".into()),
                    starting_stack: 200.0,
                    player_bounty: None,
                    is_sitting_out: None,
                },
                PlayerObj {
                    id: 2,
                    seat: 2,
                    name: "Bob".into(),
                    display: None,
                    starting_stack: 200.0,
                    player_bounty: None,
                    is_sitting_out: None,
                },
            ],
            rounds: vec![],
            pots: vec![],
            tournament_bounties: None,
        }
    }

    fn deterministic_config() -> AnonymizeConfig {
        AnonymizeConfig {
            seed: Some(17),
            ..AnonymizeConfig::default()
        }
    }

    #[test]
    fn default_config_rewrites_every_identifying_field() {
        let mut hand = sample_hand();
        let mut a = Anonymizer::new(deterministic_config());
        a.anonymize(&mut hand);

        assert_ne!(hand.site_name, "RealSite");
        assert_ne!(hand.network_name, "RealNet");
        assert_ne!(hand.internal_version, "9.9.9");
        assert_ne!(hand.table_name, "RealTable");
        assert_ne!(hand.table_handle.as_deref(), Some("Handle"));
        assert_ne!(hand.table_skin.as_deref(), Some("RealSite.it"));
        assert_ne!(hand.game_number, "555000001");
        assert_ne!(hand.players[0].name, "Alice");
        assert_ne!(hand.players[1].name, "Bob");
        assert_ne!(hand.players[0].display.as_deref(), Some("AliceDisplay"));
    }

    #[test]
    fn keep_strategy_preserves_names() {
        let mut hand = sample_hand();
        let cfg = AnonymizeConfig {
            name_strategy: NameStrategy::Keep,
            ..deterministic_config()
        };
        Anonymizer::new(cfg).anonymize(&mut hand);
        assert_eq!(hand.players[0].name, "Alice");
        assert_eq!(hand.players[1].name, "Bob");
    }

    #[test]
    fn stable_strategy_shares_names_across_hands() {
        let mut a = Anonymizer::new(deterministic_config());
        let mut hand1 = sample_hand();
        let mut hand2 = sample_hand();
        a.anonymize(&mut hand1);
        a.anonymize(&mut hand2);
        assert_eq!(hand1.players[0].name, hand2.players[0].name);
        assert_eq!(hand1.players[1].name, hand2.players[1].name);
    }

    #[test]
    fn per_hand_strategy_refreshes_names_each_hand() {
        let cfg = AnonymizeConfig {
            name_strategy: NameStrategy::PerHand,
            ..deterministic_config()
        };
        let mut a = Anonymizer::new(cfg);
        let mut hand1 = sample_hand();
        let mut hand2 = sample_hand();
        a.anonymize(&mut hand1);
        a.anonymize(&mut hand2);
        assert_ne!(hand1.players[0].name, hand2.players[0].name);
    }

    #[test]
    fn seeded_runs_are_reproducible() {
        let mut h1 = sample_hand();
        let mut h2 = sample_hand();
        Anonymizer::new(deterministic_config()).anonymize(&mut h1);
        Anonymizer::new(deterministic_config()).anonymize(&mut h2);
        assert_eq!(h1, h2);
    }

    #[test]
    fn time_fuzz_shift_is_constant_across_hands() {
        // With zero jitter, hands at the same original time land at
        // the same anonymized time.
        let cfg = AnonymizeConfig {
            time_fuzz: Some(super::super::config::TimeFuzzConfig {
                max_global_shift: std::time::Duration::from_secs(3600),
                max_per_hand_jitter: std::time::Duration::ZERO,
            }),
            ..deterministic_config()
        };
        let mut a = Anonymizer::new(cfg);
        let mut h1 = sample_hand();
        let mut h2 = sample_hand();
        a.anonymize(&mut h1);
        a.anonymize(&mut h2);
        assert_eq!(h1.start_date_utc, h2.start_date_utc);
    }

    #[test]
    fn hero_player_id_is_preserved() {
        // We don't remap numeric player IDs — only names — so
        // hero_player_id should still point at the same slot.
        let mut hand = sample_hand();
        let hero = hand.hero_player_id;
        Anonymizer::new(deterministic_config()).anonymize(&mut hand);
        assert_eq!(hand.hero_player_id, hero);
    }
}
