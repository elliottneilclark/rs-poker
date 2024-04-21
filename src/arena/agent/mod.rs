//! `Agent`s are the automatic playes in the poker simulations. They are the
//! logic and strategies behind figuring out expected value.
//!
//! Some basic agents are provided as a way of testing baseline value.
mod calling;
mod folding;
mod random;
mod replay;

use super::{action::AgentAction, game_state::GameState};
/// This is the trait that you need to implement in order to implenet
/// different strategies. It's up to you to to implement the logic and state.
///
/// Agents must implment Clone. This punts all mutex or reference counting
/// issues to the writer of agent but also allows single threaded simulations
/// not to need `Arc<Mutex<T>>`'s overhead.
pub trait Agent {
    /// This is the method that will be called by the game to get the action
    fn act(&mut self, id: &uuid::Uuid, game_state: &GameState) -> AgentAction;
}

/// AgentBuilder is a trait that is used to build agents for tournaments
/// where each simulation needs a new agent.
pub trait AgentBuilder {
    /// This method is called before each game to build a new agent.
    fn build(&self, game_state: &GameState) -> Box<dyn Agent>;
}

pub use calling::{CallingAgent, CallingAgentBuilder};
pub use folding::{FoldingAgent, FoldingAgentBuilder};
pub use random::{RandomAgent, RandomAgentBuilder, RandomPotControlAgent};
pub use replay::{SliceReplayAgent, VecReplayAgent};
