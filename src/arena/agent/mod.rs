//! `Agent`s are the automatic playes in the poker simulations. They are the
//! logic and strategies behind figuring out expected value.
//!
//! Some basic agents are provided as a way of testing baseline value.
mod calling;
mod folding;
mod random;
mod replay;

use crate::arena::{action::AgentAction, game_state::GameState};

/// This is the trait that you need to implement in order to implenet
/// different strategies. It's up to you to to implement the logic and state.
pub trait Agent {
    fn act(&mut self, game_state: &GameState) -> AgentAction;
}

pub use calling::CallingAgent;
pub use folding::FoldingAgent;
pub use random::RandomAgent;
pub use replay::{SliceReplayAgent, VecReplayAgent};
