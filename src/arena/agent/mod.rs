//! `Agent`s are the automatic playes in the poker simulations. They are the
//! logic and strategies behind figuring out expected value.
//!
//! Some basic agents are provided as a way of testing baseline value.
mod all_in;
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
    fn act(&mut self, id: u128, game_state: &GameState) -> AgentAction;
}

/// AgentBuilder is a trait that is used to build agents for tournaments
/// where each simulation needs a new agent.
pub trait AgentGenerator {
    /// This method is called before each game to build a new agent.
    fn generate(&self, game_state: &GameState) -> Box<dyn Agent>;
}

pub trait CloneAgent: Agent {
    fn clone_box(&self) -> Box<dyn Agent>;
}

impl<T> CloneAgent for T
where
    T: 'static + Agent + Clone,
{
    fn clone_box(&self) -> Box<dyn Agent> {
        Box::new(self.clone())
    }
}

pub struct CloneAgentGenerator<T> {
    agent: T,
}

impl<T> CloneAgentGenerator<T>
where
    T: CloneAgent,
{
    pub fn new(agent: T) -> Self {
        CloneAgentGenerator { agent }
    }
}

impl<T> AgentGenerator for CloneAgentGenerator<T>
where
    T: CloneAgent,
{
    fn generate(&self, _game_state: &GameState) -> Box<dyn Agent> {
        self.agent.clone_box()
    }
}

pub use all_in::{AllInAgent, AllInAgentGenerator};
pub use calling::{CallingAgent, CallingAgentGenerator};
pub use folding::{FoldingAgent, FoldingAgentGenerator};
pub use random::{RandomAgent, RandomAgentGenerator, RandomPotControlAgent};
pub use replay::{SliceReplayAgent, VecReplayAgent};
