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
//! # Competition Example
//!
//! It's also possible to run a competition where the
//! same agents compete in multiple simulations
//! with tabulated results
//!
//! ```
//! use rs_poker::arena::agent::CallingAgent;
//! use rs_poker::arena::agent::FoldingAgent;
//! use rs_poker::arena::agent::RandomAgent;
//! use rs_poker::arena::competition::CloneAgent;
//! use rs_poker::arena::competition::CloningAgentsGenerator;
//! use rs_poker::arena::competition::EmptyHistorianGenerator;
//! use rs_poker::arena::competition::HoldemCompetition;
//! use rs_poker::arena::competition::RandomGameStateGenerator;
//! use rs_poker::arena::competition::StandardSimulationGenerator;
//!
//! // We are not limited to just heads up. We can have up to full ring of 9 agents.
//! let agents: Vec<Box<dyn CloneAgent>> = vec![
//!     Box::<CallingAgent>::default(),
//!     Box::<RandomAgent>::default(),
//!     Box::<FoldingAgent>::default(),
//! ];
//!
//! let agent_gen = CloningAgentsGenerator::new(agents);
//! let game_state_gen = RandomGameStateGenerator::new(3, 100.0, 500.0, 10.0, 5.0, 0.0);
//! let sim_gen =
//!     StandardSimulationGenerator::new(agent_gen, game_state_gen, EmptyHistorianGenerator);
//!
//! let mut competition = HoldemCompetition::new(sim_gen);
//!
//! let _first_results = competition.run(100).unwrap();
//! let recent_results = competition.run(100).unwrap();
//!
//! // The holdem competition tabulates the results accross multiple runs.
//! println!("{:?}", recent_results);
//! ```

pub mod action;
pub mod agent;
pub mod competition;
pub mod errors;
pub mod game_state;
pub mod historian;
pub mod sim_builder;
pub mod simulation;
pub mod tournament;

#[cfg(any(test, feature = "arena-test-util"))]
pub mod test_util;

pub use agent::{Agent, AgentBuilder};
pub use game_state::GameState;
pub use historian::{Historian, HistorianError};
pub use sim_builder::{HoldemSimulationBuilder, RngHoldemSimulationBuilder};
pub use simulation::HoldemSimulation;
