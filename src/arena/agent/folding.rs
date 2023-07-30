use crate::arena::{action::AgentAction, game_state::GameState};

use super::Agent;

/// A simple agent that folds unless there is only one active player left.
#[derive(Default, Debug, Clone, Copy)]
pub struct FoldingAgent {}

impl Agent for FoldingAgent {
    fn act(self: &mut FoldingAgent, game_state: &GameState) -> AgentAction {
        if game_state.current_round_data().num_active_players() == 1 {
            AgentAction::Bet(game_state.current_round_data().bet)
        } else {
            AgentAction::Fold
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::{rngs::StdRng, SeedableRng};

    use crate::arena::{game_state::Round, GameState, RngHoldemSimulationBuilder};

    use super::*;

    #[test_log::test]
    fn test_folding_agents() {
        let stacks = vec![100; 2];
        let rng = StdRng::seed_from_u64(420);

        let game_state = GameState::new(stacks, 10, 5, 0);
        let mut sim = RngHoldemSimulationBuilder::default()
            .rng(rng)
            .game_state(game_state)
            .agents(vec![Box::new(FoldingAgent {}), Box::new(FoldingAgent {})])
            .build()
            .unwrap();

        sim.run();

        assert_eq!(sim.game_state.num_active_players(), 1);
        assert_eq!(sim.game_state.round, Round::Complete);

        assert_eq!(15, sim.game_state.player_bet.iter().sum());

        assert_eq!(15, sim.game_state.player_winnings.iter().sum());
        assert_eq!(15, sim.game_state.player_winnings[0]);
    }
}
