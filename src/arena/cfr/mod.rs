//! The CFR module implements a small CFR simulation of poker when combined with
//! the arena module, it provides the tools to solve poker games.
//!
//! # Overview
//!
//! CFR Works by traversing a tree of game states and updating the regret
//! values for each action taken.
//!
//! ## State Structure
//!
//! Trees in rust are hard because of the borrow checker. Instead of ref counted
//! pointers we use an arena to store the nodes of the tree. This arena (vector
//! of nodes) is then used via address. Rather than a pointer to a node we store
//! the index.
//!
//! See `CFRStateInternal` for more details on the arena structure.
//!
//! ## Historian
//!
//! Arenas simulate a single game. For each player there's an agent. That agent
//! is responsible for deciding which action to take when it is their turn. For
//! that the agent looks in the tree. The tree needs to be up to date with the
//! current game state. That is the job of the historian. The historian is
//! responsible for updating the tree with the current game state. However
//! the tree is lazily created.
//!
//! ## Action Generator
//!
//! The action generator is responsible for generating possible actions, mapping
//! actions into indices in the children array of the nodes, and deciding on the
//! least regretted action to take.
//!
//! ActionGenerator must be stateless, so that the same action
//! generator can be used as a type parameter for agents and historians.
//!
//! ## Agent
//!
//! The agent is responsible for deciding which action to take when it is
//! their turn. For that the agent looks in the tree. Then it will simulate all
//! the possible actions and update the regret values for each action taken.
//! Then it will use the CFR+ algorithm to choose the action to take.
mod action_generator;
mod agent;
mod export;
mod historian;
mod node;
mod state;

pub use action_generator::{ActionGenerator, BasicCFRActionGenerator};
pub use agent::CFRAgent;
pub use export::{ExportFormat, export_cfr_state, export_to_dot, export_to_png, export_to_svg};
pub use historian::CFRHistorian;
pub use node::{Node, NodeData, PlayerData, TerminalData};
pub use state::{CFRState, TraversalState};

#[cfg(test)]
mod tests {
    use std::vec;

    use crate::arena::cfr::BasicCFRActionGenerator;
    use crate::arena::game_state::{Round, RoundData};

    use crate::arena::historian::DirectoryHistorian;
    use crate::arena::{
        Agent, GameState, Historian, HoldemSimulation, HoldemSimulationBuilder, test_util,
    };
    use crate::core::{Hand, PlayerBitSet};

    use super::{CFRAgent, CFRState};

    #[test]
    fn test_should_fold_all_in() {
        let num_agents = 2;

        // Player 0 has a pair of kings
        let hand_zero = Hand::new_from_str("AsKsKcAcTh4d8d").unwrap();
        // Player 1 has a pair of tens
        let hand_one = Hand::new_from_str("JdTcKcAcTh4d8d").unwrap();

        let board = (hand_zero.clone() & hand_one.clone()).iter().collect();
        // Zero is all in.
        let stacks: Vec<f32> = vec![0.0, 900.0];
        let player_bet = vec![1000.0, 100.0];
        let player_bet_round = vec![900.0, 0.0];
        // Create a game state where player 0 is all in and player 1 should make a
        // decision to call or fold
        let round_data = RoundData::new_with_bets(
            num_agents,
            100.0,
            PlayerBitSet::new(num_agents),
            1,
            player_bet_round,
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

        let sim = run(game_state);

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

        let board = (hand_zero.clone() & hand_one.clone()).iter().collect();
        // Zero is all in.
        let stacks: Vec<f32> = vec![0.0, 900.0];
        let player_bet = vec![1000.0, 100.0];
        let player_bet_round = vec![900.0, 0.0];
        let round_data = RoundData::new_with_bets(
            num_agents,
            100.0,
            PlayerBitSet::new(num_agents),
            1,
            player_bet_round,
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

        let sim = run(game_state);

        // Player 1 should not put any more bets in and should fold
        assert_eq!(sim.game_state.player_bet[1], 1000.0);

        // Player 1 should win the pot
        assert_eq!(sim.game_state.stacks[1], 2000.0);
    }

    #[test]
    fn test_should_fold_with_one_round_to_go() {
        // Player 0 has 3 of a kind, aces
        let hand_zero = Hand::new_from_str("AdAcAs5h9hJcKd").unwrap();
        // Player 1 has a pair of kings
        let hand_one = Hand::new_from_str("Kc2cAs5h9hJcKd").unwrap();

        let game_state = build_from_hands(hand_zero, hand_one, Round::Turn);
        let result = run(game_state);

        // Player 1 should not put any more bets in and should fold
        assert_eq!(result.game_state.player_bet[1], 100.0);
    }

    #[test]
    fn test_should_fold_with_two_rounds_to_go() {
        let hand_zero = Hand::new_from_str("AsAhAdAcTh").unwrap();
        let hand_one = Hand::new_from_str("JsTcAdAcTh").unwrap();

        let game_state = build_from_hands(hand_zero, hand_one, Round::Flop);

        let result = run(game_state);

        // Player 1 should not put any more bets in and should fold
        assert_eq!(result.game_state.player_bet[1], 100.0);
    }

    #[test]
    fn test_should_fold_after_preflop() {
        let hand_zero = Hand::new_from_str("AsAh").unwrap();
        let hand_one = Hand::new_from_str("2c7d").unwrap();

        let game_state = build_from_hands(hand_zero, hand_one, Round::Preflop);
        let result = run(game_state);

        // Player 1 should not put any more bets in and should fold
        assert_eq!(result.game_state.player_bet[1], 100.0);
    }

    fn build_from_hands(hand_zero: Hand, hand_one: Hand, round: Round) -> GameState {
        let board = (hand_zero.clone() & hand_one.clone()).iter().collect();
        let num_agents = 2;

        // Zero is all in.
        let stacks: Vec<f32> = vec![0.0, 900.0];
        let player_bet = vec![1000.0, 100.0];
        let player_bet_round = vec![900.0, 0.0];
        let round_data = RoundData::new_with_bets(
            num_agents,
            100.0,
            PlayerBitSet::new(num_agents),
            1,
            player_bet_round,
        );
        GameState::new(
            round,
            round_data,
            board,
            vec![hand_zero, hand_one],
            stacks,
            player_bet,
            5.0,
            0.0,
            0.0,
            0,
        )
    }

    fn run(game_state: GameState) -> HoldemSimulation {
        // Each agent keeps it's own reward state.
        let states: Vec<_> = (0..game_state.num_players)
            .map(|_| CFRState::new(game_state.clone()))
            .collect();

        let agents: Vec<_> = states
            .iter()
            .enumerate()
            .map(|(i, s)| Box::new(CFRAgent::<BasicCFRActionGenerator>::new(s.clone(), i)))
            .collect();

        let mut historians: Vec<Box<dyn Historian>> = agents
            .iter()
            .map(|a| Box::new(a.historian()) as Box<dyn Historian>)
            .collect();

        historians.push(Box::new(DirectoryHistorian::new("/tmp/cfr_test".into())));

        let dyn_agents = agents.into_iter().map(|a| a as Box<dyn Agent>).collect();

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(dyn_agents)
            .historians(historians)
            .build()
            .unwrap();

        sim.run();

        assert_eq!(Round::Complete, sim.game_state.round);

        test_util::assert_valid_game_state(&sim.game_state);

        sim
    }
}
