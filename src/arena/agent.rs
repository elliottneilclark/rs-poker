use super::action::AgentAction;
use super::game_state::GameState;

pub trait Agent {
    fn act(&self, game_state: &GameState) -> AgentAction;
}

pub struct FoldingAgent {}

impl Agent for FoldingAgent {
    fn act(&self, game_state: &GameState) -> AgentAction {
        if game_state.current_round_data().num_active_players() == 1 {
            AgentAction::Bet(game_state.current_round_data().bet)
        } else {
            dbg!("folding");
            AgentAction::Fold
        }
    }
}

pub struct CallingAgent {}

impl Agent for CallingAgent {
    fn act(&self, game_state: &GameState) -> AgentAction {
        AgentAction::Bet(game_state.current_round_data().bet)
    }
}
