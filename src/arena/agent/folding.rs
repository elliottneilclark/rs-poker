use crate::arena::{action::AgentAction, game_state::GameState};

use super::Agent;

/// A simple agent that folds unless there is only one active player left.
#[derive(Default, Debug, Clone, Copy)]
pub struct FoldingAgent {}

impl Agent for FoldingAgent {
    fn act(self: &mut FoldingAgent, _id: &uuid::Uuid, game_state: &GameState) -> AgentAction {
        if game_state.current_round_num_active_players() == 1 {
            AgentAction::Bet(game_state.current_round_bet())
        } else {
            AgentAction::Fold
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use rand::{rngs::StdRng, SeedableRng};

    use crate::arena::{game_state::Round, RngHoldemSimulationBuilder};

    use super::*;

    #[test_log::test]
    fn test_folding_agents() {
        let stacks = vec![100.0; 2];
        let rng = StdRng::seed_from_u64(420);

        let game_state = GameState::new(stacks, 10.0, 5.0, 0.0, 0);
        let mut sim = RngHoldemSimulationBuilder::default()
            .rng(rng)
            .game_state(game_state)
            .agents(vec![Box::new(FoldingAgent {}), Box::new(FoldingAgent {})])
            .build()
            .unwrap();

        sim.run();

        assert_eq!(sim.game_state.num_active_players(), 1);
        assert_eq!(sim.game_state.round, Round::Complete);

        assert_relative_eq!(15.0_f32, sim.game_state.player_bet.iter().sum());

        assert_relative_eq!(15.0_f32, sim.game_state.player_winnings.iter().sum());
        assert_relative_eq!(15.0_f32, sim.game_state.player_winnings[1]);
    }
}
