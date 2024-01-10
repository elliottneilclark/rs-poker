//! This is the arena module for simulation via agents.

/// Public module containing types for actions that agents can take in the
/// arena.
pub mod action;
pub mod agent;
pub mod competition;
pub mod errors;
pub mod game_state;
pub mod simulation;

#[cfg(any(test, feature = "arena-test-util"))]
pub mod test_util;

pub use agent::Agent;
pub use game_state::GameState;
pub use simulation::{HoldemSimulation, HoldemSimulationBuilder, RngHoldemSimulationBuilder};
