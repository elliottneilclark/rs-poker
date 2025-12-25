use crate::arena::{Agent, GameState};

/// AgentBuilder is a trait that is used to build agents for tournaments
/// where each simulation needs a new agent.
pub trait AgentGenerator {
    /// This method is called before each game to build a new agent.
    /// The `player_idx` parameter indicates which player position this agent is for.
    fn generate(&self, player_idx: usize, game_state: &GameState) -> Box<dyn Agent>;
}
