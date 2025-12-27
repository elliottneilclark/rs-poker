//! # Open Hand History (OHH) Format Support
//!
//! This module provides support for the Open Hand History format v1.4.7,
//! a standardized JSON format for poker hand histories.
//!
//! ## Features
//!
//! - **Data Model**: Complete Rust structs matching the OHH specification
//! - **Serialization**: Convert to/from JSON using serde
//! - **Arena Integration**: Convert arena simulations to OHH format (requires `arena` feature)
//! - **File Writing**: Append hand histories to files in JSON Lines format
//!
//! ## Specification
//!
//! See <https://hh-specs.handhistory.org/> for the full OHH specification.
//!
//! ## Usage with Arena
//!
//! ```no_run
//! # #[cfg(all(feature = "open-hand-history", feature = "arena"))] {
//! use rs_poker::arena::{HoldemSimulationBuilder, GameState};
//! use rs_poker::arena::historian::OpenHandHistoryHistorian;
//! use std::path::PathBuf;
//!
//! let historian = OpenHandHistoryHistorian::new(PathBuf::from("hands.jsonl"));
//! // Add historian to your simulation...
//! # }
//! ```
mod hand_history;
mod serde_utils;
mod writer;

#[cfg(feature = "open-hand-history-test-util")]
mod test_util;

#[cfg(feature = "arena")]
mod converter;

pub use hand_history::*;
pub use writer::*;

#[cfg(feature = "arena")]
pub use converter::*;

#[cfg(feature = "open-hand-history-test-util")]
pub use test_util::*;
