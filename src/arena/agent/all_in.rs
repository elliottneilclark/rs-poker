use crate::arena::{GameState, action::AgentAction};

use super::{Agent, AgentGenerator};

/// A simple agent that always calls. This can
/// stand in for a player who is a calling
/// station for the rest of a hand.
#[derive(Debug, Clone, Copy, Default)]
pub struct AllInAgent;

impl Agent for AllInAgent {
    fn act(self: &mut AllInAgent, _id: &uuid::Uuid, game_state: &GameState) -> AgentAction {
        AgentAction::Bet(game_state.current_player_stack() + game_state.current_round_bet())
    }
}

/// Default `AgentGenerator` for `AllInAgent`.
#[derive(Debug, Clone, Copy, Default)]
pub struct AllInAgentGenerator;

impl AgentGenerator for AllInAgentGenerator {
    fn generate(&self, _game_state: &GameState) -> Box<dyn Agent> {
        Box::new(AllInAgent)
    }
}
