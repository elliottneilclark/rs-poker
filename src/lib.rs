//! RS-Poker is a library for poker.
//! It's mostly meant for Holdem games, however the core functionality
//! should work for all game types.
//!
//! # Implemented:
//! Currently RS-Poker supports:
//!
//! * Hand Iteration.
//! * Hand Ranking.
//! * Hand Range parsing.
//! * Hand Range generation.
//! * Holdem Game State.
//!
#![deny(clippy::all)]
extern crate rand;

/// Allow all the core poker functionality to be used
/// externally. Everything in core should be agnostic
/// to poker style.
pub mod core;
/// The holdem specific code. This contains range
/// parsing, game state, and starting hand code.
pub mod holdem;

/// Given a tournament calculate the implied
/// equity in the total tournament.
pub mod simulated_icm;

/// Simulate poker games via agents that
/// play. Then determine who wins the most over
/// time
pub mod arena;
