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
//! See `NodeArena` for more details on the arena structure.
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
//! ## Action Index Mapper
//!
//! The `ActionIndexMapper` provides a fixed absolute-amount mapping for actions
//! to indices. It maps actions to indices 0-51:
//! - Index 0: Fold
//! - Index 1: Call/Check
//! - Indices 2-50: Raises (logarithmic distribution)
//! - Index 51: All-in
//!
//! ## Agent
//!
//! The agent is responsible for deciding which action to take when it is
//! their turn. For that the agent looks in the tree. Then it will simulate all
//! the possible actions and update the regret values for each action taken.
//! Then it will use the CFR+ algorithm to choose the action to take.
//!
//! ## Preflop Chart Action Generator
//!
//! For situations where preflop exploration is too expensive, the
//! `PreflopChartActionGenerator` limits exploration to only actions that
//! have non-zero probability in pre-configured charts for the current
//! hand/position. See `PreflopChartConfig` for chart configuration.
mod action_bit_set;
mod action_generator;
mod action_index_mapper;
mod action_picker;
mod action_validator;
mod agent;
mod export;
mod gamestate_iterator_gen;
mod historian;
mod node;
mod node_arena;
mod state;
mod traversal_state;

pub use action_generator::{
    ActionGenerator, BasicCFRActionGenerator, ConfigurableActionConfig,
    ConfigurableActionGenerator, PreflopChartActionConfig, PreflopChartActionGenerator,
    PreflopChartConfig, RoundActionConfig, SimpleActionGenerator,
};
pub use action_index_mapper::{
    ACTION_IDX_ALL_IN, ACTION_IDX_CALL, ACTION_IDX_FOLD, ACTION_IDX_RAISE_MAX,
    ACTION_IDX_RAISE_MIN, ActionIndexMapper, ActionIndexMapperConfig, NUM_ACTION_INDICES,
};
pub use action_picker::{ActionPicker, get_regret_matcher_from_node};
pub use action_validator::{ValidatorMode, validate_actions};
pub use agent::{CFRAgent, CFRAgentBuilder};
pub use export::{ExportFormat, export_cfr_state, export_to_dot, export_to_png, export_to_svg};
pub use gamestate_iterator_gen::{
    DepthBasedIteratorGen, DepthBasedIteratorGenConfig, GameStateIteratorGen,
};
pub use historian::CFRHistorian;
pub use node::{Node, NodeData, PlayerData, TerminalData};
pub use state::CFRState;
pub use traversal_state::{TraversalSet, TraversalState};

#[cfg(test)]
mod tests {
    use std::vec;

    use crate::arena::cfr::{
        BasicCFRActionGenerator, DepthBasedIteratorGen, DepthBasedIteratorGenConfig, TraversalSet,
    };
    use crate::arena::game_state::{Round, RoundData};

    use crate::arena::{
        Agent, GameState, GameStateBuilder, HoldemSimulation, HoldemSimulationBuilder, test_util,
    };
    use crate::core::{Hand, PlayerBitSet};

    use super::CFRAgentBuilder;

    #[test]
    fn test_should_fold_all_in() {
        let num_agents = 2;

        // Player 0 has a pair of kings
        let hand_zero = Hand::new_from_str("AsKsKcAcTh4d8d").unwrap();
        // Player 1 has a pair of tens
        let hand_one = Hand::new_from_str("JdTcKcAcTh4d8d").unwrap();

        let board = (hand_zero & hand_one).iter().collect();
        // Zero is all in.
        let stacks: Vec<f32> = vec![0.0, 900.0];
        let player_bet = vec![1000.0, 100.0];
        let player_bet_round = vec![900.0, 0.0];
        // Create a game state where player 0 is all in and player 1 should make a
        // decision to call or fold
        let round_data =
            RoundData::new_with_bets(100.0, PlayerBitSet::new(num_agents), 1, player_bet_round);
        let game_state = GameStateBuilder::new()
            .round(Round::River)
            .round_data(round_data)
            .board(board)
            .hands(vec![hand_zero, hand_one])
            .stacks(stacks)
            .player_bet(player_bet)
            .big_blind(5.0)
            .small_blind(0.0)
            .build()
            .unwrap();

        // Increase iterations significantly for 52-action space convergence
        // These tests are inherently stochastic - higher iterations = more reliable
        let sim = run(game_state, 5000);

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

        let board = (hand_zero & hand_one).iter().collect();
        // Zero is all in.
        let stacks: Vec<f32> = vec![0.0, 900.0];
        let player_bet = vec![1000.0, 100.0];
        let player_bet_round = vec![900.0, 0.0];
        let round_data =
            RoundData::new_with_bets(100.0, PlayerBitSet::new(num_agents), 1, player_bet_round);
        let game_state = GameStateBuilder::new()
            .round(Round::River)
            .round_data(round_data)
            .board(board)
            .hands(vec![hand_zero, hand_one])
            .stacks(stacks)
            .player_bet(player_bet)
            .big_blind(5.0)
            .small_blind(0.0)
            .build()
            .unwrap();

        // Increase iterations significantly for 52-action space convergence
        // These tests are inherently stochastic - higher iterations = more reliable
        let sim = run(game_state, 50000);

        // Player 1 should call the all-in with three of a kind
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
        let result = run(game_state, 200);

        // Player 1 should not put any more bets in and should fold
        assert_eq!(result.game_state.player_bet[1], 100.0);
    }

    #[test]
    fn test_should_fold_with_two_rounds_to_go() {
        let hand_zero = Hand::new_from_str("AsAhAdAcTh").unwrap();
        let hand_one = Hand::new_from_str("JsTcAdAcTh").unwrap();

        let game_state = build_from_hands(hand_zero, hand_one, Round::Flop);

        let result = run(game_state, 200);

        // Player 1 should not put any more bets in and should fold
        assert_eq!(result.game_state.player_bet[1], 100.0);
    }

    #[test]
    fn test_should_fold_after_preflop() {
        let hand_zero = Hand::new_from_str("AsAh").unwrap();
        let hand_one = Hand::new_from_str("2s7h").unwrap();

        let game_state = build_from_hands(hand_zero, hand_one, Round::Preflop);
        // Increase iterations significantly for 52-action space convergence
        // These tests are inherently stochastic - higher iterations = more reliable
        let result = run(game_state, 50000);

        // Player 1 should not put any more bets in and should fold
        assert_eq!(result.game_state.player_bet[1], 100.0);
    }

    fn build_from_hands(hand_zero: Hand, hand_one: Hand, round: Round) -> GameState {
        let board = (hand_zero & hand_one).iter().collect();
        let num_agents = 2;

        // Zero is all in.
        let stacks: Vec<f32> = vec![0.0, 900.0];
        let player_bet = vec![1000.0, 100.0];
        let player_bet_round = vec![900.0, 0.0];
        let round_data =
            RoundData::new_with_bets(100.0, PlayerBitSet::new(num_agents), 1, player_bet_round);
        GameStateBuilder::new()
            .round(round)
            .round_data(round_data)
            .board(board)
            .hands(vec![hand_zero, hand_one])
            .stacks(stacks)
            .player_bet(player_bet)
            .big_blind(5.0)
            .small_blind(0.0)
            .build()
            .unwrap()
    }

    /// Helper to create CFR states for all players from a game state.
    fn make_cfr_states(game_state: &GameState) -> Vec<super::CFRState> {
        (0..game_state.num_players)
            .map(|_| super::CFRState::new(game_state.clone()))
            .collect()
    }

    fn run(game_state: GameState, num_hands: usize) -> HoldemSimulation {
        use rand::{SeedableRng, rngs::StdRng};

        // All agents share the same CFR states and traversal set.
        let cfr_states = make_cfr_states(&game_state);
        let traversal_set = TraversalSet::new(game_state.num_players);
        let iter_config = DepthBasedIteratorGenConfig::new(vec![num_hands, 1]);
        let agents: Vec<_> = (0..game_state.num_players)
            .map(|idx| {
                Box::new(
                    CFRAgentBuilder::<BasicCFRActionGenerator, DepthBasedIteratorGen>::new()
                        .name(format!("CFRAgent-run-{idx}"))
                        .player_idx(idx)
                        .cfr_states(cfr_states.clone())
                        .traversal_set(traversal_set.clone())
                        .gamestate_iterator_gen_config(iter_config.clone())
                        .action_gen_config(())
                        .rng(StdRng::seed_from_u64(12345 + idx as u64))
                        .build(),
                )
            })
            .collect();

        let dyn_agents = agents.into_iter().map(|a| a as Box<dyn Agent>).collect();

        // Use a fixed seed for reproducibility in tests
        let mut rng = StdRng::seed_from_u64(42);

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(dyn_agents)
            .cfr_context(cfr_states, traversal_set, true)
            .build()
            .unwrap();

        sim.run(&mut rng);

        assert_eq!(Round::Complete, sim.game_state.round);

        test_util::assert_valid_game_state(&sim.game_state);

        sim
    }

    /// Debug test to examine CFR convergence in the all-in scenario.
    #[test]
    fn test_debug_all_in_convergence() {
        let num_agents = 2;

        // Player 0 has a pair of tens
        let hand_zero = Hand::new_from_str("JdTcKcAcTh4d8d").unwrap();
        // Player 1 has three of a kind, kings
        let hand_one = Hand::new_from_str("KcKsKdAcTh4d8d").unwrap();

        let board = (hand_zero & hand_one).iter().collect();
        let stacks: Vec<f32> = vec![0.0, 900.0];
        let player_bet = vec![1000.0, 100.0];
        let player_bet_round = vec![900.0, 0.0];
        let round_data =
            RoundData::new_with_bets(100.0, PlayerBitSet::new(num_agents), 1, player_bet_round);
        let game_state = GameStateBuilder::new()
            .round(Round::River)
            .round_data(round_data)
            .board(board)
            .hands(vec![hand_zero, hand_one])
            .stacks(stacks)
            .player_bet(player_bet)
            .big_blind(5.0)
            .small_blind(0.0)
            .build()
            .unwrap();

        // Verify initial state
        println!("current_round_bet = {}", game_state.current_round_bet());
        println!(
            "Player 1 current_round_player_bet = {}",
            game_state.current_round_player_bet(1)
        );
        println!("Player 1 stack = {}", game_state.stacks[1]);
        println!(
            "Player 1 all-in amount = {}",
            game_state.current_round_current_player_bet() + game_state.current_player_stack()
        );

        // Create a CFR agent for player 1 (the one who needs to decide)
        let cfr_states = make_cfr_states(&game_state);
        let traversal_set = TraversalSet::new(game_state.num_players);
        let iter_config = DepthBasedIteratorGenConfig::new(vec![100, 1]);
        let agent = CFRAgentBuilder::<BasicCFRActionGenerator, DepthBasedIteratorGen>::new()
            .name("CFRAgent-debug")
            .player_idx(1)
            .cfr_states(cfr_states)
            .traversal_set(traversal_set)
            .gamestate_iterator_gen_config(iter_config)
            .action_gen_config(())
            .build();

        // Get the valid actions using BasicCFRActionGenerator directly
        use crate::arena::cfr::action_generator::ActionGenerator;
        use crate::arena::cfr::{ActionIndexMapper, ActionIndexMapperConfig, TraversalState};

        let action_gen = BasicCFRActionGenerator::new(
            agent.cfr_state().clone(),
            TraversalState::new_root(1), // Player 1
        );
        let actions = action_gen.gen_possible_actions(&game_state);
        println!("Valid actions: {:?}", actions);

        // Map actions to indices
        let mapper = ActionIndexMapper::new(ActionIndexMapperConfig::from_game_state(&game_state));
        for action in &actions {
            let idx = mapper.action_to_idx(action, &game_state);
            println!("Action {:?} maps to index {}", action, idx);
        }
    }

    /// Test CFR agent starting from Round::Starting where cards are dealt during simulation.
    /// This is the scenario that triggers the bug in agent_comparison.
    #[test]
    fn test_cfr_agent_from_starting_round() {
        use rand::{SeedableRng, rngs::StdRng};

        // Create a starting game state (no cards dealt yet)
        let game_state = GameStateBuilder::new()
            .num_players_with_stack(2, 100.0)
            .blinds(10.0, 5.0)
            .build()
            .unwrap();

        // All agents share the same CFR states and traversal set
        let cfr_states = make_cfr_states(&game_state);
        let traversal_set = TraversalSet::new(game_state.num_players);
        let iter_config = DepthBasedIteratorGenConfig::new(vec![2, 1]);
        let agents: Vec<_> = (0..2)
            .map(|idx| {
                Box::new(
                    CFRAgentBuilder::<BasicCFRActionGenerator, DepthBasedIteratorGen>::new()
                        .name(format!("CFRAgent-starting-{idx}"))
                        .player_idx(idx)
                        .cfr_states(cfr_states.clone())
                        .traversal_set(traversal_set.clone())
                        .gamestate_iterator_gen_config(iter_config.clone())
                        .action_gen_config(())
                        .build(),
                )
            })
            .collect();

        let dyn_agents = agents.into_iter().map(|a| a as Box<dyn Agent>).collect();

        // Use a fixed seed for reproducibility
        let mut rng = StdRng::seed_from_u64(42);

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(dyn_agents)
            .cfr_context(cfr_states, traversal_set, true)
            .build()
            .unwrap();

        sim.run(&mut rng);

        assert_eq!(Round::Complete, sim.game_state.round);
        test_util::assert_valid_game_state(&sim.game_state);
    }

    /// Test CFR agent vs non-CFR agent starting from Round::Starting.
    /// This mimics the agent_comparison scenario.
    #[test]
    fn test_cfr_vs_calling_from_starting_round() {
        use crate::arena::agent::CallingAgent;
        use rand::{SeedableRng, rngs::StdRng};

        // Create a starting game state (no cards dealt yet)
        let game_state = GameStateBuilder::new()
            .num_players_with_stack(2, 100.0)
            .blinds(10.0, 5.0)
            .build()
            .unwrap();

        // Create shared CFR states and traversal set
        let cfr_states = make_cfr_states(&game_state);
        let traversal_set = TraversalSet::new(game_state.num_players);
        let iter_config = DepthBasedIteratorGenConfig::new(vec![2, 1]);
        let cfr_agent = Box::new(
            CFRAgentBuilder::<BasicCFRActionGenerator, DepthBasedIteratorGen>::new()
                .name("CFRAgent")
                .player_idx(0)
                .cfr_states(cfr_states.clone())
                .traversal_set(traversal_set.clone())
                .gamestate_iterator_gen_config(iter_config)
                .action_gen_config(())
                .build(),
        );

        let calling_agent = Box::new(CallingAgent::new("CallingAgent"));

        let agents: Vec<Box<dyn Agent>> = vec![cfr_agent, calling_agent];

        let mut rng = StdRng::seed_from_u64(42);

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .cfr_context(cfr_states, traversal_set, true)
            .build()
            .unwrap();

        sim.run(&mut rng);

        assert_eq!(Round::Complete, sim.game_state.round);
        test_util::assert_valid_game_state(&sim.game_state);
    }

    /// Run multiple games with the same CFR agent to test for tree conflicts.
    /// This simulates what happens in agent_comparison across multiple games.
    #[test]
    fn test_multiple_games_same_cfr_agent() {
        use rand::{SeedableRng, rngs::StdRng};

        let iter_config = DepthBasedIteratorGenConfig::new(vec![2, 1]);

        // Run 5 games with fresh agents each time (like agent_comparison does)
        for game_idx in 0..5 {
            let game_state = GameStateBuilder::new()
                .num_players_with_stack(2, 100.0)
                .blinds(10.0, 5.0)
                .build()
                .unwrap();

            let cfr_states = make_cfr_states(&game_state);
            let traversal_set = TraversalSet::new(game_state.num_players);
            let agents: Vec<Box<dyn Agent>> = (0..2)
                .map(|idx| {
                    Box::new(
                        CFRAgentBuilder::<BasicCFRActionGenerator, DepthBasedIteratorGen>::new()
                            .name(format!("CFRAgent-game{game_idx}-p{idx}"))
                            .player_idx(idx)
                            .cfr_states(cfr_states.clone())
                            .traversal_set(traversal_set.clone())
                            .gamestate_iterator_gen_config(iter_config.clone())
                            .action_gen_config(())
                            .build(),
                    ) as Box<dyn Agent>
                })
                .collect();

            let mut rng = StdRng::seed_from_u64(42 + game_idx as u64);

            let mut sim = HoldemSimulationBuilder::default()
                .game_state(game_state)
                .agents(agents)
                .cfr_context(cfr_states, traversal_set, true)
                .build()
                .unwrap();

            sim.run(&mut rng);

            assert_eq!(
                Round::Complete,
                sim.game_state.round,
                "Game {game_idx} should complete"
            );
            test_util::assert_valid_game_state(&sim.game_state);
        }
    }

    /// Test CFR agent with ConfigurableActionGenerator to verify that:
    /// 1. Actions mapping to the same index are deduplicated
    /// 2. Node type mismatches are handled via allow_node_mutation
    ///
    /// This uses a configuration with 4x raise, half pot, and full pot bet sizing
    /// which can produce actions that map to the same index in the ActionIndexMapper.
    #[test]
    fn test_cfr_with_configurable_action_generator() {
        use crate::arena::cfr::{
            ConfigurableActionConfig, ConfigurableActionGenerator, RoundActionConfig,
        };
        use rand::{SeedableRng, rngs::StdRng};

        // Create a starting game state
        let game_state = GameStateBuilder::new()
            .num_players_with_stack(2, 500.0)
            .blinds(10.0, 5.0)
            .build()
            .unwrap();

        // Configure action generator with 4x raise, half pot, and full pot
        // These can produce bet amounts that map to the same index
        let action_config = ConfigurableActionConfig {
            default: RoundActionConfig {
                call_enabled: true,
                raise_mult: vec![4.0],
                pot_mult: vec![0.5, 1.0],
                setup_shove: false,
                all_in: true,
            },
            preflop: None,
            flop: None,
            turn: None,
            river: None,
        };

        // All agents share the same CFR states and traversal set
        let cfr_states = make_cfr_states(&game_state);
        let traversal_set = TraversalSet::new(game_state.num_players);
        let iter_config = DepthBasedIteratorGenConfig::new(vec![2, 1]);
        let agents: Vec<_> = (0..2)
            .map(|idx| {
                Box::new(
                    CFRAgentBuilder::<ConfigurableActionGenerator, DepthBasedIteratorGen>::new()
                        .name(format!("CFRAgent-configurable-{idx}"))
                        .player_idx(idx)
                        .cfr_states(cfr_states.clone())
                        .traversal_set(traversal_set.clone())
                        .gamestate_iterator_gen_config(iter_config.clone())
                        .action_gen_config(action_config.clone())
                        .allow_node_mutation(true)
                        .build(),
                )
            })
            .collect();

        let dyn_agents = agents.into_iter().map(|a| a as Box<dyn Agent>).collect();

        let mut rng = StdRng::seed_from_u64(42);

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(dyn_agents)
            .cfr_context(cfr_states, traversal_set, true)
            .build()
            .unwrap();

        // This should not panic - both action deduplication and node mutation should work
        sim.run(&mut rng);

        assert_eq!(Round::Complete, sim.game_state.round);
        test_util::assert_valid_game_state(&sim.game_state);
    }
}
