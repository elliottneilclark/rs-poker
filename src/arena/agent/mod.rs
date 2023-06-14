mod calling;
mod folding;
mod random;
mod replay;

use crate::arena::{action::AgentAction, game_state::GameState};

pub trait Agent {
    fn act(&mut self, game_state: &GameState) -> AgentAction;
}

pub use calling::CallingAgent;
pub use folding::FoldingAgent;
pub use random::RandomAgent;
pub use replay::{SliceReplayAgent, VecReplayAgent};
