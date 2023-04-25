use crate::arena::{action::AgentAction, game_state::GameState};

use super::Agent;

#[derive(Debug, Clone, Copy)]
pub struct CallingAgent {}

impl Agent for CallingAgent {
    fn act(&self, game_state: &GameState) -> AgentAction {
        AgentAction::Bet(game_state.current_round_data().bet)
    }
}

#[cfg(test)]
mod tests {
    use crate::arena::{GameState, HoldemSimulation};

    use super::*;

    #[test]
    fn test_call_agents() {
        let stacks = vec![100; 4];
        let game_state = GameState::new(stacks, 10, 5, 0);
        let mut sim = HoldemSimulation::new_with_agents(
            game_state,
            vec![
                Box::new(CallingAgent {}),
                Box::new(CallingAgent {}),
                Box::new(CallingAgent {}),
                Box::new(CallingAgent {}),
            ],
        );

        sim.run();

        assert_eq!(sim.game_state.num_active_players(), 4);

        assert_ne!(0, sim.game_state.player_winnings.iter().sum());
        assert_eq!(40, sim.game_state.player_winnings.iter().sum());
    }
}
