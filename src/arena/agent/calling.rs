use crate::arena::{action::AgentAction, game_state::GameState};

use super::{Agent, AgentGenerator};

/// A simple agent that always calls. This can
/// stand in for a player who is a calling
/// station for the rest of a hand.
#[derive(Debug, Clone, Copy, Default)]
pub struct CallingAgent;

impl Agent for CallingAgent {
    fn act(self: &mut CallingAgent, _id: &uuid::Uuid, game_state: &GameState) -> AgentAction {
        AgentAction::Bet(game_state.current_round_bet())
    }
}

/// Default `AgentGenerator` for `CallingAgent`.
#[derive(Debug, Clone, Copy, Default)]
pub struct CallingAgentGenerator;

impl AgentGenerator for CallingAgentGenerator {
    fn generate(&self, _game_state: &GameState) -> Box<dyn Agent> {
        Box::new(CallingAgent)
    }
}

#[cfg(test)]
mod tests {
    use rand::rng;

    use crate::arena::HoldemSimulationBuilder;

    use super::*;

    #[test_log::test]
    fn test_call_agents() {
        let stacks = vec![100.0; 4];
        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        let mut sim = HoldemSimulationBuilder::default()
            .rng(rng())
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

        assert_ne!(0.0, sim.game_state.player_winnings.iter().sum::<f32>());
        assert_eq!(40.0, sim.game_state.player_winnings.iter().sum::<f32>());
    }
}
