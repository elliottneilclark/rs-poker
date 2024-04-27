//! This is the arena module for simulation via agents.
//!
//! # Single Simulation
//!
//! The tools allow explicit control over the
//! simulation all the way down to the rng.
//!
//! ## Single Simulation Example
//!
//! ```
//! use rand::{rngs::StdRng, SeedableRng};
//! use rs_poker::arena::agent::CallingAgent;
//! use rs_poker::arena::agent::RandomAgent;
//! use rs_poker::arena::game_state::GameState;
//! use rs_poker::arena::RngHoldemSimulationBuilder;
//!
//! let stacks = vec![100.0, 100.0];
//! let agents: Vec<Box<dyn rs_poker::arena::Agent>> = vec![
//!     Box::<CallingAgent>::default(),
//!     Box::<RandomAgent>::default(),
//! ];
//! let rng = StdRng::seed_from_u64(420);
//!
//! let game_state = GameState::new(stacks, 10.0, 5.0, 0.0, 0);
//! let mut sim = RngHoldemSimulationBuilder::default()
//!     .game_state(game_state)
//!     .rng(rng)
//!     .agents(agents)
//!     .build()
//!     .unwrap();
//!
//! let result = sim.run();
//! ```
//!
//! # Competition Examples
//!
//! ## `HoldemCompetition` Example
//!
//! It's also possible to run a competition where the
//! same agents compete in multiple simulations
//! with tabulated results
//!
//! ```
//! use rs_poker::arena::agent::CallingAgentGenerator;
//! use rs_poker::arena::agent::FoldingAgentGenerator;
//! use rs_poker::arena::agent::RandomAgentGenerator;
//! use rs_poker::arena::competition::HoldemCompetition;
//! use rs_poker::arena::competition::StandardSimulationGenerator;
//! use rs_poker::arena::game_state::RandomGameStateGenerator;
//! use rs_poker::arena::AgentGenerator;
//!
//! // We are not limited to just heads up. We can have up to full ring of 9 agents.
//! let agent_gens: Vec<Box<dyn AgentGenerator>> = vec![
//!     Box::<CallingAgentGenerator>::default(),
//!     Box::<FoldingAgentGenerator>::default(),
//!     Box::<RandomAgentGenerator>::default(),
//! ];
//!
//! let game_state_gen = RandomGameStateGenerator::new(3, 100.0, 500.0, 10.0, 5.0, 0.0);
//! let sim_gen = StandardSimulationGenerator::new(agent_gens, vec![], game_state_gen);
//!
//! let mut competition = HoldemCompetition::new(sim_gen);
//!
//! let _first_results = competition.run(100).unwrap();
//! let recent_results = competition.run(100).unwrap();
//!
//! // The holdem competition tabulates the results accross multiple runs.
//! println!("{:?}", recent_results);
//! ```
//!
//! ## `SingleTableTournament` Example
//!
//! It's also possible to run a single table tournament where the
//! game state continues on until one player has all the money.
//!
//! ```
//! use rs_poker::arena::agent::RandomAgentGenerator;
//! use rs_poker::arena::competition::SingleTableTournamentBuilder;
//! use rs_poker::arena::game_state::GameState;
//! use rs_poker::arena::AgentGenerator;
//!
//! // We are not limited to just heads up. We can have up to full ring of 9 agents.
//! let agent_gens: Vec<Box<dyn AgentGenerator>> = vec![
//!     Box::<RandomAgentGenerator>::default(),
//!     Box::<RandomAgentGenerator>::default(),
//!     Box::<RandomAgentGenerator>::default(),
//!     Box::<RandomAgentGenerator>::default(),
//! ];
//! let stacks = vec![100.0; 4];
//!
//! // This is the starting game state.
//! let game_state = GameState::new(stacks, 10.0, 5.0, 1.0, 0);
//!
//! let tournament = SingleTableTournamentBuilder::default()
//!     .agent_generators(agent_gens)
//!     .starting_game_state(game_state)
//!     .build()
//!     .unwrap();
//!
//! let results = tournament.run().unwrap();
//! ```

pub mod action;
pub mod agent;
pub mod competition;
pub mod errors;
pub mod game_state;
pub mod historian;
pub mod sim_builder;
pub mod simulation;

#[cfg(any(test, feature = "arena-test-util"))]
pub mod test_util;

pub use agent::{Agent, AgentGenerator, CloneAgentGenerator};
pub use game_state::{CloneGameStateGenerator, GameState, GameStateGenerator};
pub use historian::{CloneHistorianGenerator, Historian, HistorianError, HistorianGenerator};
pub use sim_builder::{HoldemSimulationBuilder, RngHoldemSimulationBuilder};
pub use simulation::HoldemSimulation;
