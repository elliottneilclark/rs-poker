use crate::arena::{action::AgentAction, game_state::GameState};

use super::{Agent, AgentBuilder};

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

/// Default Builder for `CallingAgent`.
pub struct CallingAgentBuilder;

impl AgentBuilder for CallingAgentBuilder {
    fn build(&self, _game_state: &GameState) -> Box<dyn Agent> {
        Box::new(CallingAgent)
    }
}

#[cfg(test)]
mod tests {
    use rand::thread_rng;

    use crate::arena::HoldemSimulationBuilder;

    use super::*;

    #[test_log::test]
    fn test_call_agents() {
        let stacks = vec![100.0; 4];
        let game_state = GameState::new(stacks, 10.0, 5.0, 0.0, 0);
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

        assert_ne!(
            0.0_f32,
            sim.game_state.player_winnings.clone().into_iter().sum()
        );
        assert_eq!(40.0_f32, sim.game_state.player_winnings.iter().sum());
    }
}
