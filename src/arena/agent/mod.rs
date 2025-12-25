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
/// This is the trait that you need to implement in order to implenet
/// different strategies. It's up to you to to implement the logic and state.
///
/// Agents must implment Clone. This punts all mutex or reference counting
/// issues to the writer of agent but also allows single threaded simulations
/// not to need `Arc<Mutex<T>>`'s overhead.
pub trait Agent {
    /// This is the method that will be called by the game to get the action
    fn act(&mut self, id: u128, game_state: &GameState) -> AgentAction;

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
pub use config::{AgentConfig, AgentConfigError, ConfigAgentGenerator};
pub use folding::{FoldingAgent, FoldingAgentGenerator};
pub use generator::AgentGenerator;
pub use random::{RandomAgent, RandomAgentGenerator, RandomPotControlAgent};
pub use replay::{SliceReplayAgent, VecReplayAgent};
