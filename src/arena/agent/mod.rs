//! `Agent`s are the automatic playes in the poker simulations. They are the
//! logic and strategies behind figuring out expected value.
//!
//! Some basic agents are provided as a way of testing baseline value.
mod calling;
mod fn_observer;
mod folding;
mod random;
mod replay;

use super::{
    action::{Action, AgentAction},
    game_state::GameState,
};
use dyn_clone::DynClone;
/// This is the trait that you need to implement in order to implenet
/// different strategies. It's up to you to to implement the logic and state.
///
/// Agents must implment Clone. This punts all mutex or reference counting
/// issues to the writer of agent but also allows single threaded simulations
/// not to need Arc<Mutex<T>>'s overhead.
pub trait Agent: DynClone {
    /// This is the method that will be called by the game to get the action
    fn act(&mut self, game_state: &GameState) -> AgentAction;
    /// When some action happens that changes the game state, the agent can be
    /// notified The game state will be the new state after the action has
    /// been applied
    fn record_action(&mut self, _game_state: &GameState, _action: &Action) {}
}

dyn_clone::clone_trait_object!(Agent);

pub use calling::CallingAgent;
pub use fn_observer::FnObserverRandomAgent;
pub use folding::FoldingAgent;
pub use random::{RandomAgent, RandomPotControlAgent};
pub use replay::{SliceReplayAgent, VecReplayAgent};
