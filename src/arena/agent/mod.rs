//! `Agent`s are the automatic playes in the poker simulations. They are the
//! logic and strategies behind figuring out expected value.
//!
//! Some basic agents are provided as a way of testing baseline value.
mod all_in;
mod calling;
mod clone;
mod config;
mod folding;
mod generator;
mod random;
mod replay;

use crate::arena::{GameState, Historian, action::AgentAction};
use async_trait::async_trait;

/// This is the trait that you need to implement in order to implenet
/// different strategies. It's up to you to to implement the logic and state.
///
/// Agents are async so that an `act` implementation may await IO (an HTTP
/// endpoint, a batched ML inference call) or drive concurrent exploration.
/// The `Send` bound lets a simulation (and its owned agents) be spawned onto
/// the tokio runtime for recursive sub-simulations. `Sync` is not required.
#[async_trait]
pub trait Agent: Send {
    /// This is the method that will be called by the game to get the action
    async fn act(&mut self, id: u128, game_state: &GameState) -> AgentAction;

    /// Every agent should expose a human-readable name for logging/stats.
    fn name(&self) -> &str;

    // Some Agents may need to be able to see the changes in the game
    // state. This is the method that will be called to create historians
    // when starting a new simulation game.
    fn historian(&self) -> Option<Box<dyn Historian>> {
        None
    }
}

pub use all_in::{AllInAgent, AllInAgentGenerator};
pub use calling::{CallingAgent, CallingAgentGenerator};
pub use clone::{CloneAgent, CloneAgentGenerator};
pub use config::{AgentConfig, AgentConfigError, CfrExploration, ConfigAgentBuilder};
pub use folding::{FoldingAgent, FoldingAgentGenerator};
pub use generator::AgentGenerator;
pub use random::{RandomAgent, RandomAgentGenerator, RandomPotControlAgent};
pub use replay::{SliceReplayAgent, VecReplayAgent};
