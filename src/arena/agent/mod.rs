mod calling;
mod folding;
mod random;

use crate::arena::{action::AgentAction, game_state::GameState};

pub trait Agent {
    fn act(&self, game_state: &GameState) -> AgentAction;
}

pub use calling::CallingAgent;
pub use folding::FoldingAgent;
pub use random::RandomAgent;
