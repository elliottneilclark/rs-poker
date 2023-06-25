use crate::arena::{action::AgentAction, game_state::GameState};

use super::Agent;

#[derive(Debug, Clone, Copy)]
pub struct CallingAgent {}

impl Agent for CallingAgent {
    fn act(self: &mut CallingAgent, game_state: &GameState) -> AgentAction {
        AgentAction::Bet(game_state.current_round_data().bet)
    }
}

#[cfg(test)]
mod tests {
    use rand::thread_rng;

    use crate::arena::{GameState, HoldemSimulationBuilder};

    use super::*;

    #[test]
    fn test_call_agents() {
        let stacks = vec![100; 4];
        let game_state = GameState::new(stacks, 10, 5, 0);
        let mut sim = HoldemSimulationBuilder::default()
            .rng(thread_rng())
            .game_state(game_state)
            .agents(vec![
                Box::new(CallingAgent {}),
                Box::new(CallingAgent {}),
                Box::new(CallingAgent {}),
                Box::new(CallingAgent {}),
            ])
            .build()
            .unwrap();

        sim.run();

        assert_eq!(sim.game_state.num_active_players(), 4);

        assert_ne!(0, sim.game_state.player_winnings.iter().sum());
        assert_eq!(40, sim.game_state.player_winnings.iter().sum());
    }
}
