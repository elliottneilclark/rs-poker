//! Furry Fiesta is a library for poker.
//! It's mostly meant for Holdem games, however the core functionality
//! should work for all game types.
//!
//! # Implemented:
//! Currently Furry Fiesta supports:
//!
//! * Hand Iteration.
//! * Hand Ranking.
//! * Hand Range parsing.
//! * Hand Range generation.
//!
//! # Planned:
//! * Holdem Game State.
//! * Multi-threading
//!
extern crate rand;

/// Allow all the core poker functionality to be used
/// externally. Everything in core should be agnostic
/// to poker style.
pub mod core;
/// The holdem specific code. This contains range
/// parsing, game state, and starting hand code.
pub mod holdem;
