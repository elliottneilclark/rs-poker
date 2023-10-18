//! `Agent`s are the automatic playes in the poker simulations. They are the
//! logic and strategies behind figuring out expected value.
//!
//! Some basic agents are provided as a way of testing baseline value.
mod calling;
mod folding;
mod random;
mod replay;

use crate::arena::{action::AgentAction, game_state::GameState};
use dyn_clone::DynClone;
/// This is the trait that you need to implement in order to implenet
/// different strategies. It's up to you to to implement the logic and state.
///
/// Agents must implment Clone. This punts all mutex or reference counting
/// issues to the writer of agent but also allows single threaded simulations
/// not to need Arc<Mutex<T>>'s overhead.
pub trait Agent: DynClone {
    fn act(&mut self, game_state: &GameState) -> AgentAction;
}

dyn_clone::clone_trait_object!(Agent);

pub use calling::CallingAgent;
pub use folding::FoldingAgent;
pub use random::{RandomAgent, RandomPotControlAgent};
pub use replay::{SliceReplayAgent, VecReplayAgent};
