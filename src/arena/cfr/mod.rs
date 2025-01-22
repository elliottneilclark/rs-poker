mod action_generator;
mod agent;
mod historian;
mod node;
mod state;

pub use action_generator::{ActionGenerator, BasicCFRActionGenerator};
pub use agent::CFRAgent;
pub use historian::CFRHistorian;
pub use node::{Node, NodeData, PlayerData, TerminalData};
pub use state::{CFRState, TraversalState};

#[cfg(test)]
mod tests {
    use std::vec;

    use crate::arena::cfr::BasicCFRActionGenerator;
    use crate::arena::game_state::{Round, RoundData};

    use crate::arena::{Agent, GameState, Historian, HoldemSimulationBuilder};
    use crate::core::{Hand, PlayerBitSet};

    use super::{CFRAgent, CFRState};

    #[test]
    fn test_should_fold_all_in() {
        let num_agents = 2;

        // Player 0 has a pair of kings
        let hand_zero = Hand::new_from_str("AsKsKcAcTh4d8d").unwrap();
        // Player 1 has a pair of tens
        let hand_one = Hand::new_from_str("JdTcKcAcTh4d8d").unwrap();

        // The board is the last 5 cards of the hand
        let board = hand_zero.iter().skip(2).collect();
        // Zero is all in.
        let stacks: Vec<f32> = vec![0.0, 900.0];
        let player_bet = vec![1000.0, 100.0];
        // Create a game state where player 0 is all in and player 1 should make a
        // decision to call or fold
        let round_data = RoundData::new_with_bets(
            num_agents,
            100.0,
            PlayerBitSet::new(num_agents),
            1,
            player_bet.clone(),
        );
        let game_state = GameState::new(
            Round::River,
            round_data,
            board,
            vec![hand_zero, hand_one],
            stacks,
            player_bet,
            5.0,
            0.0,
            0.0,
            0,
        );

        let states: Vec<_> = (0..num_agents)
            .map(|_| CFRState::new(game_state.clone()))
            .collect();

        let agents: Vec<_> = states
            .iter()
            .enumerate()
            .map(|(i, s)| Box::new(CFRAgent::<BasicCFRActionGenerator>::new(s.clone(), i)))
            .collect();

        let historians: Vec<Box<dyn Historian>> = agents
            .iter()
            .map(|a| Box::new(a.historian()) as Box<dyn Historian>)
            .collect();

        let dyn_agents = agents.into_iter().map(|a| a as Box<dyn Agent>).collect();

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(dyn_agents)
            .historians(historians)
            .build()
            .unwrap();

        sim.run();

        // Player 1 should not put any more bets in and should fold
        assert_eq!(sim.game_state.player_bet[1], 100.0);

        // Player 0 should win the pot
        assert_eq!(sim.game_state.stacks[0], 1100.0);

        // Player 1 didn't put any more in and didn't win
        assert_eq!(sim.game_state.stacks[1], 900.0);
    }

    #[test]
    fn test_should_go_all_in() {
        let num_agents = 2;

        // Player 0 has a pair of tens
        let hand_zero = Hand::new_from_str("JdTcKcAcTh4d8d").unwrap();
        // Player 1 has three of a kind, kings
        let hand_one = Hand::new_from_str("KcKsKdAcTh4d8d").unwrap();

        // The board is the last 5 cards of the hand
        let board = hand_zero.iter().skip(2).collect();
        // Zero is all in.
        let stacks: Vec<f32> = vec![0.0, 900.0];
        let player_bet = vec![1000.0, 100.0];
        let round_data = RoundData::new(num_agents, 100.0, PlayerBitSet::new(num_agents), 1);
        let game_state = GameState::new(
            Round::River,
            round_data,
            board,
            vec![hand_zero, hand_one],
            stacks,
            player_bet,
            5.0,
            0.0,
            0.0,
            0,
        );

        // Each agent keeps it's own reward state.
        let states: Vec<_> = (0..num_agents)
            .map(|_| CFRState::new(game_state.clone()))
            .collect();

        let agents: Vec<_> = states
            .iter()
            .enumerate()
            .map(|(i, s)| Box::new(CFRAgent::<BasicCFRActionGenerator>::new(s.clone(), i)))
            .collect();

        let historians: Vec<Box<dyn Historian>> = agents
            .iter()
            .map(|a| Box::new(a.historian()) as Box<dyn Historian>)
            .collect();

        let dyn_agents = agents.into_iter().map(|a| a as Box<dyn Agent>).collect();

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(dyn_agents)
            .historians(historians)
            .build()
            .unwrap();

        sim.run();

        // Player 1 should not put any more bets in and should fold
        assert_eq!(sim.game_state.player_bet[1], 1000.0);

        // Player 1 should win the pot
        assert_eq!(sim.game_state.stacks[1], 2000.0);
    }
}
