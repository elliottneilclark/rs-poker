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
//! use rand::{SeedableRng, rngs::StdRng};
//! use rs_poker::arena::HoldemSimulationBuilder;
//! use rs_poker::arena::GameStateBuilder;
//! use rs_poker::arena::agent::CallingAgent;
//! use rs_poker::arena::agent::RandomAgent;
//!
//! let rt = tokio::runtime::Runtime::new().unwrap();
//! rt.block_on(async {
//!     let agents: Vec<Box<dyn rs_poker::arena::Agent>> = vec![
//!         Box::<CallingAgent>::default(),
//!         Box::<RandomAgent>::default(),
//!     ];
//!
//!     let game_state = GameStateBuilder::new()
//!         .num_players_with_stack(2, 100.0)
//!         .blinds(10.0, 5.0)
//!         .build()
//!         .unwrap();
//!     let mut sim = HoldemSimulationBuilder::default()
//!         .game_state(game_state)
//!         .agents(agents)
//!         .build_with_rng(StdRng::seed_from_u64(420))
//!         .unwrap();
//!
//!     sim.run().await;
//! });
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
//! use rs_poker::arena::AgentGenerator;
//! use rs_poker::arena::agent::CallingAgentGenerator;
//! use rs_poker::arena::agent::FoldingAgentGenerator;
//! use rs_poker::arena::agent::RandomAgentGenerator;
//! use rs_poker::arena::competition::HoldemCompetition;
//! use rs_poker::arena::competition::StandardSimulationIterator;
//! use rs_poker::arena::game_state::RandomGameStateGenerator;
//!
//! // We are not limited to just heads up. We can have up to full ring of 9 agents.
//! let agent_gens: Vec<Box<dyn AgentGenerator>> = vec![
//!     Box::<CallingAgentGenerator>::default(),
//!     Box::<FoldingAgentGenerator>::default(),
//!     Box::<RandomAgentGenerator>::default(),
//! ];
//!
//! let game_state_gen = RandomGameStateGenerator::new(3, 100.0, 500.0, 10.0, 5.0, 0.0);
//! let sim_gen = StandardSimulationIterator::new(agent_gens, vec![], game_state_gen);
//!
//! let mut competition = HoldemCompetition::new(sim_gen);
//!
//! let rt = tokio::runtime::Runtime::new().unwrap();
//! rt.block_on(async {
//!     let _first_results = competition.run(100).await.unwrap();
//!     let recent_results = competition.run(100).await.unwrap();
//!
//!     // The holdem competition tabulates the results accross multiple runs.
//!     println!("{:?}", recent_results);
//! });
//! ```
//!
//! ## `SingleTableTournament` Example
//!
//! It's also possible to run a single table tournament where the
//! game state continues on until one player has all the money.
//!
//! ```
//! use rs_poker::arena::AgentGenerator;
//! use rs_poker::arena::GameStateBuilder;
//! use rs_poker::arena::agent::RandomAgentGenerator;
//! use rs_poker::arena::competition::SingleTableTournamentBuilder;
//!
//! // We are not limited to just heads up. We can have up to full ring of 9 agents.
//! let agent_gens: Vec<Box<dyn AgentGenerator>> = vec![
//!     Box::<RandomAgentGenerator>::default(),
//!     Box::<RandomAgentGenerator>::default(),
//!     Box::<RandomAgentGenerator>::default(),
//!     Box::<RandomAgentGenerator>::default(),
//! ];
//!
//! // This is the starting game state.
//! let game_state = GameStateBuilder::new()
//!     .num_players_with_stack(4, 100.0)
//!     .blinds(10.0, 5.0)
//!     .ante(1.0)
//!     .build()
//!     .unwrap();
//!
//! let tournament = SingleTableTournamentBuilder::default()
//!     .agent_generators(agent_gens)
//!     .starting_game_state(game_state)
//!     .build(rand::rng())
//!     .unwrap();
//!
//! let rt = tokio::runtime::Runtime::new().unwrap();
//! let results = rt.block_on(tournament.run()).unwrap();
//! ```
//!
//! ##  Counter Factual Regret Minimization (CFR) Example
//!
//! rs-poker has an implementation of CFR that can be used to implement agents
//! that decide their actions based on the regret minimization algorithm. For
//! that you can use the `CFRAgent` along with the `CFRHistorian` and `CFRState`
//! structs.
//!
//! The strategy is implemented by the `ActionGenerator` trait, which is used to
//! generate potential actions for a given game state. The
//! `BasicCFRActionGenerator` is a simple implementation that generates fold,
//! call, and All-In actions.
//!
//! Tree recursion is controlled by `MaxWidth` (a `Budget` impl):
//! `MaxWidth.recursive_widths` gives the per-depth wave width, and depths
//! past the end of that vec use the cheap fast-forward reward path. The
//! number of waves at each depth is governed by the rest of the `Budget`.
//!
//! The Agent then chooses the action based upon the regret minimization.
//!
//! ```
//! use rs_poker::arena::cfr::CFRAgent;
//! ```
pub mod action;
pub mod agent;
pub mod cfr;
pub mod competition;
pub mod errors;
pub mod game_state;
pub mod hand_estimator;
pub mod historian;
pub mod rng;
pub mod sim_builder;
pub mod simulation;

#[cfg(any(test, feature = "arena-test-util"))]
pub mod test_util;

#[cfg(any(test, feature = "open-hand-history"))]
pub mod comparison;

pub use agent::{Agent, AgentGenerator, CloneAgentGenerator, ConfigAgentBuilder};
pub use game_state::{
    CloneGameStateGenerator, GameState, GameStateBuilder, GameStateGenerator,
    RandomGameStateGenerator,
};
pub use hand_estimator::{GameLog, HandDistributionEstimator, OpponentRanges};
pub use historian::{CloneHistorianGenerator, Historian, HistorianError, HistorianGenerator};
pub use rng::seeded_rng;
pub use sim_builder::HoldemSimulationBuilder;
pub use simulation::HoldemSimulation;
