//! # Anonymization tools for Open Hand History
//!
//! Strip identifying information from OHH hand-history streams so
//! they can be shared publicly without exposing the players, sites,
//! or precise timing of real sessions.
//!
//! ## What gets rewritten
//!
//! | Field                              | Default behavior                                         |
//! |------------------------------------|----------------------------------------------------------|
//! | `players[].name`, `.display`       | Replaced with names drawn from a neutral pool            |
//! | `site_name`, `table_skin`          | Rotated to a real-world brand (PokerStars, 888, etc.)    |
//! | `network_name`                     | Rotated to a real-world network (GGNetwork, WPN, etc.)   |
//! | `internal_version`                 | Rotated to a plausible version string                    |
//! | `table_name`, `table_handle`       | Rotated to a mythological / celestial label              |
//! | `game_number`, `tournament_number` | Replaced with a sequential counter anchored at a random base |
//! | `tournament_info.name`             | Replaced with "Anonymous Tournament N"                   |
//! | `start_date_utc` (hand)            | Shifted by a random global offset plus per-hand jitter   |
//! | `tournament_info.start_date_utc`   | Shifted by the same global offset (no jitter)            |
//!
//! Everything else — cards, actions, amounts, player IDs — is left
//! untouched so downstream analysis still works.
//!
//! ## Name strategies
//!
//! Two modes, selected via [`NameStrategy`]:
//!
//! * [`NameStrategy::PerHand`] — every hand is reshuffled. Strongest
//!   privacy, but cross-hand player identity is lost.
//! * [`NameStrategy::Stable`] — names are learned on first sight and
//!   stay stable for the rest of the stream. Preserves player
//!   identity without reading the full input into memory, which is
//!   why the streaming driver exists.
//!
//! ## Streaming
//!
//! [`anonymize_stream`] reads JSONL OHH records line-by-line and
//! writes them back out, so multi-gigabyte inputs are handled with
//! bounded memory. The learned name map grows only with the number
//! of distinct players seen, not with the number of hands.
//!
//! ## Example
//!
//! ```no_run
//! use std::fs::File;
//! use std::io::{BufReader, BufWriter};
//! use rs_poker::open_hand_history::anonymize::{
//!     AnonymizeConfig, Anonymizer, anonymize_stream,
//! };
//!
//! let input = BufReader::new(File::open("in.ohh")?);
//! let output = BufWriter::new(File::create("out.ohh")?);
//! let mut anonymizer = Anonymizer::new(AnonymizeConfig::default());
//! let hands = anonymize_stream(input, output, &mut anonymizer)?;
//! println!("anonymized {hands} hands");
//! # Ok::<_, Box<dyn std::error::Error>>(())
//! ```
mod anonymizer;
mod config;
mod identifiers;
mod names;
mod sites;
mod stream;
mod tables;
mod times;

pub use anonymizer::Anonymizer;
pub use config::{AnonymizeConfig, NameStrategy, TimeFuzzConfig};
pub use names::{KeepNameMapper, NameMapper, PerHandNameMapper, StableNameMapper};
pub use stream::{StreamError, anonymize_stream};
