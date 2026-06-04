//! The CFR agent: construction, exploration engine, and `Agent` glue.
//!
//! This module is split by responsibility:
//! - [`CFRAgentBuilder`] (`builder` submodule) constructs an agent and fills in
//!   defaults (in-flight limiter, stopping budget, cancellation token).
//! - The exploration engine (`engine` submodule) owns the [`CFRAgent`] struct
//!   and the async tree walk that accumulates regret.
//! - The `ComputeRewardContext` (`reward_context` submodule) is the cheap,
//!   `Send + 'static` snapshot threaded into each spawned subtree.
//! - This file wires the agent into the arena via the [`Agent`] trait impl
//!   (`act`), which reads the strategy the engine has learned.
//!
//! Lifecycle: build with [`CFRAgentBuilder`] → the engine explores and updates
//! regret → [`Agent::act`] samples the resulting strategy.

mod builder;
mod engine;
mod fast_forward;
mod reward_context;

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use tracing::event;

use crate::arena::{Agent, GameState, action::AgentAction};

use super::{
    ActionPicker, action_generator::ActionGenerator, action_validator::validate_actions,
    get_regret_matcher_from_node,
};

pub use builder::CFRAgentBuilder;
pub use engine::CFRAgent;

/// Aborts the spawned deadline timer when the action returns, so a fast `act`
/// (one that finishes well before its deadline) leaves no lingering timer task.
pub(super) struct AbortOnDrop(pub(super) tokio::task::JoinHandle<()>);

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.abort();
    }
}

#[async_trait::async_trait]
impl<T> Agent for CFRAgent<T>
where
    T: ActionGenerator + Send + 'static,
    T::Config: Send + Sync,
{
    async fn act(&mut self, id: u128, game_state: &GameState) -> AgentAction {
        event!(tracing::Level::TRACE, ?id, "Agent acting");
        assert!(
            game_state.round_data.to_act_idx == self.traversal_state.player_idx() as usize,
            "Agent should only be called when it's the player's turn"
        );

        // make sure that we have at least 2 cards
        let player_idx = self.traversal_state.player_idx() as usize;
        assert!(
            game_state.hands[player_idx].count() == 2 || game_state.hands[player_idx].count() >= 5,
            "Agent should only be called when it has at least 2 cards"
        );

        self.ensure_target_node();

        if let Some(force_action) = self.forced_action.take() {
            event!(
                tracing::Level::DEBUG,
                ?force_action,
                "Playing forced action"
            );

            // Validate that the forced_action is still valid for this game state.
            // If not, we need to find a similar valid action.
            let valid_actions = self.action_generator.gen_possible_actions(game_state);

            // Check if the forced_action is in the valid actions (or close to it for Bet)
            match &force_action {
                AgentAction::Fold => {
                    if valid_actions.contains(&AgentAction::Fold) {
                        force_action
                    } else {
                        // Can't fold when there's nothing to call - this shouldn't happen
                        // but if it does, just call/check instead
                        event!(
                            tracing::Level::WARN,
                            "Forced Fold action invalid, using first valid action"
                        );
                        valid_actions.first().cloned().unwrap_or(AgentAction::Fold)
                    }
                }
                AgentAction::AllIn => {
                    // All-in should always be valid if we have chips
                    force_action
                }
                AgentAction::Call => {
                    // Call should always be valid
                    force_action
                }
                AgentAction::Bet(amount) => {
                    // For Bet, we need to verify the amount is still valid.
                    // The forced_action was generated for a specific game state, and while
                    // we expect the game state to be the same, let's validate to be safe.
                    // Use the ActionIndexMapper for consistent action-to-index mapping.
                    let forced_idx = self
                        .action_index_mapper
                        .action_to_idx(&force_action, game_state);

                    // Find a valid action with the same index
                    if let Some(valid_action) = valid_actions.iter().find(|a| {
                        self.action_index_mapper.action_to_idx(a, game_state) == forced_idx
                    }) {
                        // Found a valid action with the same index - use it
                        // (it might have a slightly different amount due to game state changes)
                        valid_action.clone()
                    } else {
                        // The forced action's index doesn't correspond to a valid action.
                        // This indicates a game state mismatch. Log and use a fallback.
                        event!(
                            tracing::Level::WARN,
                            ?force_action,
                            forced_idx = forced_idx,
                            amount = amount,
                            current_bet = game_state.current_round_bet(),
                            min_raise = game_state.current_round_min_raise(),
                            "Forced Bet action index not valid, using first valid action"
                        );
                        valid_actions.first().cloned().unwrap_or(AgentAction::Fold)
                    }
                }
            }
        } else {
            self.ensure_regret_matcher();

            // Fresh per-act stop flag at the root; sub-agents share the parent's.
            // The engine arms the deadline timer itself when it sees
            // `NextStep::StartTimer` (the `Deadline` budget leaf returns it once
            // at depth 0). No need for a CancellationToken anymore.
            if self.depth == 0 {
                self.stop = Arc::new(AtomicBool::new(false));
            }

            self.explore_all_actions(game_state).await;

            // Use ActionPicker to select an action based on the regret matcher.
            let raw_actions = self.action_generator.gen_possible_actions(game_state);
            let possible_actions = validate_actions(raw_actions, game_state);
            let target_node_idx = self.target_node_idx().unwrap();

            self.cfr_state.with_node_data(target_node_idx, |node_data| {
                let regret_matcher = get_regret_matcher_from_node(node_data);

                let picker = ActionPicker::new(
                    &self.action_index_mapper,
                    &possible_actions,
                    regret_matcher,
                    game_state,
                );
                // `rand::rng()` is fine here — pick happens after all awaits,
                // so the !Send ThreadRng never crosses an .await.
                picker.pick_action(&mut rand::rng())
            })
        }
    }

    fn name(&self) -> &str {
        self.name.as_ref()
    }

    fn historian(&self) -> Option<Box<dyn crate::arena::Historian>> {
        if self.estimator.needs_history() {
            Some(Box::new(
                crate::arena::historian::VecHistorian::new_with_actions(self.log_storage.clone()),
            ))
        } else {
            None
        }
    }
}

/// Test-only estimator that requires history and records, into shared state,
/// how many actions the `GameLog` it received contained. It defers the actual
/// ranges to `KnownHandsEstimator` so sampling still works.
#[cfg(test)]
#[derive(Default)]
pub(crate) struct HistoryNeedingStub {
    pub last_seen_actions: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

#[cfg(test)]
#[async_trait::async_trait]
impl crate::arena::HandDistributionEstimator for HistoryNeedingStub {
    async fn estimate(
        &self,
        game_state: &GameState,
        history: Option<&crate::arena::GameLog<'_>>,
    ) -> crate::arena::OpponentRanges {
        let n = history.map(|h| h.actions.len()).unwrap_or(0);
        self.last_seen_actions
            .store(n, std::sync::atomic::Ordering::SeqCst);
        crate::arena::hand_estimator::KnownHandsEstimator
            .estimate(game_state, None)
            .await
    }

    fn needs_history(&self) -> bool {
        true
    }

}

#[cfg(test)]
mod tests {

    use std::sync::Arc;

    use little_sorry::{PcfrPlusRegretMatcher, RegretMinimizer};
    use rand::{SeedableRng, rngs::StdRng};

    use crate::arena::GameStateBuilder;
    use crate::arena::agent::CallingAgent;
    use crate::arena::cfr::{
        BasicCFRActionGenerator, ConfigurableActionConfig, ConfigurableActionGenerator,
        IterationCount, MaxWidth, MostRestrictive, PerDepth, TraversalSet,
    };
    use crate::arena::{Agent, HoldemSimulationBuilder};

    use super::super::{Budget, CFRState, NUM_ACTION_INDICES};
    use super::fast_forward::*;
    use super::*;

    /// Helper to create CFR states for all players from a game state.
    fn make_cfr_state(game_state: &GameState) -> CFRState {
        CFRState::new(game_state.clone())
    }

    /// Build a unified `Arc<dyn Budget>` from a per-depth iteration schedule
    /// `[a, b, ...]`: recurse to `iters_per_depth.len()` depths, run waves
    /// of width 1, with the per-depth iteration caps from the schedule
    /// (and 1 iteration for the fallback depth). Use this for tests that
    /// drive `act`/`explore_all_actions` — the wave loop is budget-driven,
    /// so an explicit terminating budget is required.
    fn budget_for_schedule(iters_per_depth: &[usize]) -> Arc<dyn Budget> {
        let by_depth: Vec<Arc<dyn Budget>> = iters_per_depth
            .iter()
            .map(|&h| Arc::new(IterationCount::new(h as u64)) as Arc<dyn Budget>)
            .collect();
        let iter_caps = Arc::new(PerDepth::new(by_depth, Arc::new(IterationCount::new(1))));
        let widths = Arc::new(MaxWidth::new(vec![1; iters_per_depth.len()]));
        Arc::new(MostRestrictive::new(vec![iter_caps, widths]))
    }

    /// Test that a CFR agent can play against a non-CFR agent.
    /// This is a regression test for a bug where the CFR agent's reward()
    /// function assumed all players had CFR state initialized.
    ///
    /// The scenario: Player 0 is a CallingAgent (non-CFR), Player 1 is a CFR agent.
    /// The CFR agent uses shared CFR states for ALL players.
    #[tokio::test(flavor = "current_thread")]
    async fn test_cfr_vs_non_cfr_agent() {
        let stacks: Vec<f32> = vec![50.0, 50.0];
        let game_state = GameStateBuilder::new()
            .stacks(stacks)
            .blinds(5.0, 2.5)
            .build()
            .unwrap();

        // Create shared CFR states and TraversalSet
        let cfr_state = make_cfr_state(&game_state);
        let traversal_set = TraversalSet::new(game_state.num_players);

        let budget = budget_for_schedule(&[1]);
        let cfr_agent = Box::new(
            CFRAgentBuilder::<BasicCFRActionGenerator>::new()
                .name("CFRAgent-player1")
                .player_idx(1)
                .cfr_state(cfr_state.clone())
                .traversal_set(traversal_set.clone())
                .budget(budget)
                .action_gen_config(())
                .build(),
        );

        // Player 0 is a simple calling agent (non-CFR)
        let calling_agent = Box::new(CallingAgent::new("CallingAgent-player0"));

        let agents: Vec<Box<dyn Agent>> = vec![calling_agent, cfr_agent];

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .cfr_context(cfr_state.clone(), traversal_set, true)
            .build()
            .unwrap();

        // This should not panic - the CFR agent properly handles
        // mixed-agent simulations
        sim.run().await;
    }

    /// Regression test for B4: `CFRAgent::act` must route the raw generator
    /// output through `validate_actions` before handing it to the picker.
    /// Otherwise the picker can sample a raise after `max_raises_per_round`
    /// is reached, which `do_bet` would then reject.
    ///
    /// We pin the structural invariant: every action `act` returns is one of
    /// the actions present in `validate_actions(gen_possible_actions(state))`.
    /// `ConfigurableActionGenerator` is used because it emits raise sizings
    /// (the basic generator only emits Fold/Call/AllIn so the validator has
    /// nothing to filter).
    #[tokio::test(flavor = "current_thread")]
    async fn test_act_returns_only_validated_actions_when_cap_reached() {
        use crate::arena::cfr::action_validator::validate_actions;
        use crate::core::{Card, Hand, Suit, Value};

        let mut game_state = GameStateBuilder::new()
            .num_players_with_stack(2, 100.0)
            .blinds(10.0, 5.0)
            .max_raises_per_round(Some(2))
            .build()
            .unwrap();

        // Advance to preflop and post blinds.
        game_state.advance_round(); // Starting -> Ante
        game_state.advance_round(); // Ante -> DealPreflop
        game_state.advance_round(); // DealPreflop -> Preflop
        game_state.do_bet(5.0, true).unwrap();
        game_state.do_bet(10.0, true).unwrap();

        let mut hand0 = Hand::default();
        hand0.insert(Card::new(Value::Ace, Suit::Spade));
        hand0.insert(Card::new(Value::King, Suit::Spade));
        let mut hand1 = Hand::default();
        hand1.insert(Card::new(Value::Queen, Suit::Heart));
        hand1.insert(Card::new(Value::Jack, Suit::Heart));
        game_state.hands[0] = hand0;
        game_state.hands[1] = hand1;

        // Cap raises so the validator must filter raise candidates.
        game_state.round_data.total_raise_count = 2;
        assert!(game_state.is_raise_capped());

        let cfr_state = make_cfr_state(&game_state);
        let traversal_set = TraversalSet::new(game_state.num_players);

        let budget = budget_for_schedule(&[1]);
        let mut agent = CFRAgentBuilder::<ConfigurableActionGenerator>::new()
            .name("CFRAgent-cap-test")
            .player_idx(game_state.to_act_idx())
            .cfr_state(cfr_state.clone())
            .traversal_set(traversal_set)
            .budget(budget)
            .action_gen_config(ConfigurableActionConfig::default())
            .build();

        let raw_actions = agent.action_generator.gen_possible_actions(&game_state);
        let validated_actions = validate_actions(raw_actions.clone(), &game_state);
        // Sanity: validation actually removes raise candidates here.
        assert!(
            validated_actions.len() < raw_actions.len(),
            "validate_actions should filter raises once the cap is reached \
             (raw={raw_actions:?}, validated={validated_actions:?})"
        );

        for i in 0..32u128 {
            let action = agent.act(i, &game_state).await;
            assert!(
                validated_actions.contains(&action),
                "CFRAgent returned {action:?}, not in validated set \
                 {validated_actions:?} (raw set was {raw_actions:?})"
            );
        }
    }

    #[test]
    fn test_create_agent() {
        let game_state = GameStateBuilder::new()
            .num_players_with_stack(3, 100.0)
            .blinds(10.0, 5.0)
            .build()
            .unwrap();
        let cfr_state = make_cfr_state(&game_state);
        let traversal_set = TraversalSet::new(game_state.num_players);
        let _ = CFRAgentBuilder::<BasicCFRActionGenerator>::new()
            .name("CFRAgent-test")
            .player_idx(0)
            .cfr_state(cfr_state.clone())
            .traversal_set(traversal_set)
            .action_gen_config(())
            .build();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_run_heads_up() {
        let num_agents = 2;
        let stacks: Vec<f32> = vec![50.0, 50.0];
        let game_state = GameStateBuilder::new()
            .stacks(stacks)
            .blinds(5.0, 2.5)
            .build()
            .unwrap();

        // All CFR agents share the same CFR states and TraversalSet.
        let cfr_state = make_cfr_state(&game_state);
        let traversal_set = TraversalSet::new(game_state.num_players);
        let budget = budget_for_schedule(&[2, 1]);
        let agents: Vec<Box<dyn Agent>> = (0..num_agents)
            .map(|i| {
                Box::new(
                    CFRAgentBuilder::<BasicCFRActionGenerator>::new()
                        .name(format!("CFRAgent-test-{i}"))
                        .player_idx(i)
                        .cfr_state(cfr_state.clone())
                        .traversal_set(traversal_set.clone())
                        .budget(budget.clone())
                        .action_gen_config(())
                        .build(),
                ) as Box<dyn Agent>
            })
            .collect();

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .cfr_context(cfr_state.clone(), traversal_set, true)
            .build()
            .unwrap();

        sim.run().await;
    }

    /// Test that agents sharing CFR states actually share the same CFR tree.
    /// This verifies the shared state pattern works correctly.
    #[test]
    fn test_shared_cfr_states_between_agents() {
        let game_state = GameStateBuilder::new()
            .num_players_with_stack(2, 100.0)
            .blinds(10.0, 5.0)
            .build()
            .unwrap();

        // Create shared CFR states and traversal set
        let cfr_state = make_cfr_state(&game_state);
        let traversal_set = TraversalSet::new(game_state.num_players);

        // Create two agents with the same shared states
        let agent0 = CFRAgentBuilder::<BasicCFRActionGenerator>::new()
            .name("Agent0")
            .player_idx(0)
            .cfr_state(cfr_state.clone())
            .traversal_set(traversal_set.clone())
            .action_gen_config(())
            .build();

        let agent1 = CFRAgentBuilder::<BasicCFRActionGenerator>::new()
            .name("Agent1")
            .player_idx(1)
            .cfr_state(cfr_state.clone())
            .traversal_set(traversal_set)
            .action_gen_config(())
            .build();

        // Both agents should have access to CFR state
        let state0 = agent0.cfr_state();
        let state1 = agent1.cfr_state();

        // Verify both see the same starting state (root node exists)
        assert!(state0.get_node_data(0).is_some());
        assert!(state1.get_node_data(0).is_some());

        // Both agents should be at valid positions
        assert!(state0.get_child(0, 0).is_none()); // No children yet at root
        assert!(state1.get_child(0, 0).is_none());
    }

    /// Test that the concurrent exploration path completes successfully and
    /// produces valid results when driven on a multi-threaded runtime.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_run_heads_up_parallel() {
        let num_agents = 2;
        let stacks: Vec<f32> = vec![50.0, 50.0];
        let game_state = GameStateBuilder::new()
            .stacks(stacks)
            .blinds(5.0, 2.5)
            .build()
            .unwrap();

        let cfr_state = make_cfr_state(&game_state);
        let traversal_set = TraversalSet::new(game_state.num_players);

        let budget = budget_for_schedule(&[2, 1]);
        let agents: Vec<Box<dyn Agent>> = (0..num_agents)
            .map(|i| {
                Box::new(
                    CFRAgentBuilder::<BasicCFRActionGenerator>::new()
                        .name(format!("CFRAgent-par-{i}"))
                        .player_idx(i)
                        .cfr_state(cfr_state.clone())
                        .traversal_set(traversal_set.clone())
                        .budget(budget.clone())
                        .action_gen_config(())
                        .build(),
                ) as Box<dyn Agent>
            })
            .collect();

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .cfr_context(cfr_state.clone(), traversal_set, true)
            .build()
            .unwrap();

        sim.run().await;
    }

    /// Test that the concurrent exploration path builds the CFR tree.
    /// After running, the tree should have nodes beyond just the root.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_parallel_builds_cfr_tree() {
        let stacks: Vec<f32> = vec![50.0, 50.0];
        let game_state = GameStateBuilder::new()
            .stacks(stacks)
            .blinds(5.0, 2.5)
            .build()
            .unwrap();

        let cfr_state = make_cfr_state(&game_state);
        let traversal_set = TraversalSet::new(game_state.num_players);

        let budget = budget_for_schedule(&[2, 1]);
        let agents: Vec<Box<dyn Agent>> = (0..2)
            .map(|i| {
                Box::new(
                    CFRAgentBuilder::<BasicCFRActionGenerator>::new()
                        .name(format!("CFRAgent-par-{i}"))
                        .player_idx(i)
                        .cfr_state(cfr_state.clone())
                        .traversal_set(traversal_set.clone())
                        .budget(budget.clone())
                        .action_gen_config(())
                        .build(),
                ) as Box<dyn Agent>
            })
            .collect();

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .cfr_context(cfr_state.clone(), traversal_set, true)
            .build()
            .unwrap();

        sim.run().await;

        // After running, the shared CFR tree should have grown beyond just the root node
        assert!(
            cfr_state.node_count() > 1,
            "CFR tree should have grown during simulation"
        );
    }

    /// Test that TraversalSet fork() provides proper sub-simulation isolation.
    #[test]
    fn test_sub_simulation_traversal_isolation() {
        let game_state = GameStateBuilder::new()
            .num_players_with_stack(2, 100.0)
            .blinds(10.0, 5.0)
            .build()
            .unwrap();
        let traversal_set = TraversalSet::new(game_state.num_players);

        // Get the initial traversal state position for player 0
        let initial = traversal_set.get(0);
        let initial_node = initial.node_idx();
        let initial_child = initial.chosen_child_idx();

        // Fork the traversal set (simulating sub-simulation start)
        let forked = traversal_set.fork();
        let sub_traversal = forked.get(0);

        // The sub-traversal should start at the same position
        assert_eq!(sub_traversal.node_idx(), initial_node);
        assert_eq!(sub_traversal.chosen_child_idx(), initial_child);

        // Move the sub-traversal
        sub_traversal.move_to(5, 3);

        // The sub-traversal should have moved
        assert_eq!(forked.get(0).node_idx(), 5);
        assert_eq!(forked.get(0).chosen_child_idx(), 3);

        // The original traversal should be unchanged
        assert_eq!(traversal_set.get(0).node_idx(), initial_node);
        assert_eq!(traversal_set.get(0).chosen_child_idx(), initial_child);
    }

    /// Reproduction test for the river call bug.
    ///
    /// Scenario: Player 1 holds K-high (7s Ks) facing an all-in on the river
    /// against Player 2 who has a pair of 9s (9h Qs). Board: 4s 4d 9d Ah Jd.
    /// Player 1 should ALWAYS fold, never call.
    ///
    /// This test verifies that:
    /// 1. The PCFR+ regret matcher correctly learns to fold with the reward structure
    /// 2. The full CFR agent picks fold when given this game state
    #[tokio::test(flavor = "current_thread")]
    async fn test_river_should_fold_king_high_vs_pair() {
        use crate::arena::cfr::action_generator::{
            ConfigurableActionConfig, ConfigurableActionGenerator,
        };
        use crate::arena::game_state::Round;
        use crate::core::{Card, Hand, PlayerBitSet, Suit, Value};

        // Simplified version: 2-player, river, Player 1 facing all-in
        // Player 0 has pair of 9s (winning hand)
        // Player 1 has K-high (losing hand)
        //
        // Setup: preflop/flop/turn betting already happened.
        // Player 0 went all-in on river for 500.
        // Player 1 (with 300 remaining) must decide: fold or call.
        //
        // Board: 4s 4d 9d Ah Jd
        // Player 0 hand: 9h Qs (pair of 9s)
        // Player 1 hand: 7s Ks (K high)

        let board_cards = vec![
            Card::new(Value::Four, Suit::Spade),
            Card::new(Value::Four, Suit::Diamond),
            Card::new(Value::Nine, Suit::Diamond),
            Card::new(Value::Ace, Suit::Heart),
            Card::new(Value::Jack, Suit::Diamond),
        ];

        // Player hands (including board cards for ranking)
        let p0_hole = vec![
            Card::new(Value::Nine, Suit::Heart),
            Card::new(Value::Queen, Suit::Spade),
        ];
        let p1_hole = vec![
            Card::new(Value::Seven, Suit::Spade),
            Card::new(Value::King, Suit::Spade),
        ];

        let mut p0_hand = Hand::new_with_cards(p0_hole.clone());
        p0_hand.extend(board_cards.iter().copied());
        let mut p1_hand = Hand::new_with_cards(p1_hole.clone());
        p1_hand.extend(board_cards.iter().copied());

        // Create game state at river decision point:
        // - Player 0 has gone all-in for 500 this round
        // - Player 1 has 300 left in stack, hasn't bet on river yet
        // - Previous rounds: both put in 200 each
        let stacks = vec![0.0, 300.0]; // P0 is all-in, P1 has 300 left
        let starting_stacks = vec![700.0, 700.0]; // Both started with 700
        let player_bet = vec![700.0, 400.0]; // Total bets over all rounds

        // Round data: P0 bet 500 this round, P1 hasn't bet yet
        let round_player_bet = vec![500.0, 0.0];
        let mut active = PlayerBitSet::new(2);
        // Only P1 is active (P0 is all-in)
        active.disable(0);

        let round_data = crate::arena::game_state::RoundData::new_with_bets(
            10.0, // min_raise
            active,
            1, // P1 to act
            round_player_bet,
        );

        let mut game_state = GameStateBuilder::new()
            .round(Round::River)
            .round_data(round_data)
            .stacks(stacks)
            .player_bet(player_bet)
            .big_blind(10.0)
            .small_blind(5.0)
            .hands(vec![p0_hand, p1_hand])
            .board(board_cards)
            .build()
            .unwrap();
        // Override starting_stacks (builder sets them = stacks)
        game_state.starting_stacks = starting_stacks.into();

        // Verify game state is correct
        assert_eq!(game_state.round, Round::River);
        assert_eq!(game_state.to_act_idx(), 1); // Player 1's turn
        assert_eq!(game_state.current_round_bet(), 500.0); // P0's all-in
        assert_eq!(game_state.current_player_stack(), 300.0); // P1's stack

        // Create CFR agent for Player 1
        let cfr_state = make_cfr_state(&game_state);
        let traversal_set = TraversalSet::new(game_state.num_players);

        // Use the configurable action generator (same as preflop chart post-flop)
        let action_config = ConfigurableActionConfig::default();

        let budget = budget_for_schedule(&[24, 3, 1]);
        let mut agent = CFRAgentBuilder::<ConfigurableActionGenerator>::new()
            .name("TestCFRAgent")
            .player_idx(1)
            .cfr_state(cfr_state.clone())
            .traversal_set(traversal_set.clone())
            .budget(budget)
            .action_gen_config(action_config)
            .build();

        // Check what actions the generator produces
        let possible_actions = agent.action_generator.gen_possible_actions(&game_state);
        println!("Possible actions: {:?}", possible_actions);
        for action in &possible_actions {
            let idx = agent.action_index_mapper.action_to_idx(action, &game_state);
            println!("  {:?} -> index {}", action, idx);
        }

        // Run act() - this explores and picks an action
        let chosen_action = agent.act(0, &game_state).await;
        println!("Chosen action: {:?}", chosen_action);

        // The agent should fold with K-high facing an all-in
        assert!(
            matches!(chosen_action, AgentAction::Fold),
            "Agent should fold K-high facing all-in, but chose {:?}. \
             With a fresh regret matcher, 24 iterations of exploration \
             should overwhelmingly prefer fold over call.",
            chosen_action
        );

        // Also verify the regret matcher weights
        let target_node_idx = agent.target_node_idx().unwrap();
        agent
            .cfr_state
            .with_node_data(target_node_idx, |node_data| {
                let matcher = get_regret_matcher_from_node(node_data).unwrap();
                let weights = matcher.best_weight();
                let fold_weight = weights[0]; // ACTION_IDX_FOLD
                let call_weight = weights[1]; // ACTION_IDX_CALL

                println!(
                    "Weights after exploration: fold={:.6}, call={:.6}",
                    fold_weight, call_weight
                );

                assert!(
                    fold_weight > 0.99,
                    "Fold weight should be >0.99, got {:.6}. Call weight: {:.6}",
                    fold_weight,
                    call_weight
                );
            });
    }

    /// The same K-high river scenario as `test_river_should_fold_king_high_vs_pair`,
    /// but with an empty depth schedule. An empty schedule means the root
    /// agent itself has no recursive-hands budget, so it skips sub-simulation
    /// entirely and takes the fast-forward reward path — a direct end-to-end
    /// test that the fast-forward path still produces a decisive fold.
    #[tokio::test(flavor = "current_thread")]
    async fn test_river_fold_via_fast_forward_at_depth_zero() {
        use crate::arena::cfr::action_generator::{
            ConfigurableActionConfig, ConfigurableActionGenerator,
        };
        use crate::arena::game_state::Round;
        use crate::core::{Card, Hand, PlayerBitSet, Suit, Value};

        let board_cards = vec![
            Card::new(Value::Four, Suit::Spade),
            Card::new(Value::Four, Suit::Diamond),
            Card::new(Value::Nine, Suit::Diamond),
            Card::new(Value::Ace, Suit::Heart),
            Card::new(Value::Jack, Suit::Diamond),
        ];
        let p0_hole = vec![
            Card::new(Value::Nine, Suit::Heart),
            Card::new(Value::Queen, Suit::Spade),
        ];
        let p1_hole = vec![
            Card::new(Value::Seven, Suit::Spade),
            Card::new(Value::King, Suit::Spade),
        ];
        let mut p0_hand = Hand::new_with_cards(p0_hole);
        p0_hand.extend(board_cards.iter().copied());
        let mut p1_hand = Hand::new_with_cards(p1_hole);
        p1_hand.extend(board_cards.iter().copied());

        let stacks = vec![0.0, 300.0];
        let starting_stacks = vec![700.0, 700.0];
        let player_bet = vec![700.0, 400.0];
        let round_player_bet = vec![500.0, 0.0];
        let mut active = PlayerBitSet::new(2);
        active.disable(0);
        let round_data =
            crate::arena::game_state::RoundData::new_with_bets(10.0, active, 1, round_player_bet);

        let mut game_state = GameStateBuilder::new()
            .round(Round::River)
            .round_data(round_data)
            .stacks(stacks)
            .player_bet(player_bet)
            .big_blind(10.0)
            .small_blind(5.0)
            .hands(vec![p0_hand, p1_hand])
            .board(board_cards)
            .build()
            .unwrap();
        game_state.starting_stacks = starting_stacks.into();
        game_state.total_pot = 1100.0;

        let cfr_state = make_cfr_state(&game_state);
        let traversal_set = TraversalSet::new(game_state.num_players);

        // Empty schedule → fast-forward at depth 0 (no recursion), one wave.
        let budget = budget_for_schedule(&[]);
        let mut agent = CFRAgentBuilder::<ConfigurableActionGenerator>::new()
            .name("CFR-ff")
            .player_idx(1)
            .cfr_state(cfr_state.clone())
            .traversal_set(traversal_set)
            .budget(budget)
            .action_gen_config(ConfigurableActionConfig::default())
            .build();

        let chosen = agent.act(0, &game_state).await;
        assert!(
            matches!(chosen, AgentAction::Fold),
            "Agent should fold via fast-forward path, got {:?}",
            chosen
        );
    }

    /// Test that PCFR+ correctly handles the reward structure where call
    /// reward equals the invalid action penalty.
    #[test]
    fn test_pcfr_fold_vs_call_with_penalty_equal_to_call() {
        let mut matcher = PcfrPlusRegretMatcher::new(NUM_ACTION_INDICES);

        // Simulate the exact reward structure from the river call bug:
        // - Fold reward: -604 (lost what was already bet)
        // - Call reward: -1463 (lost entire stack)
        // - Invalid penalty: -1463 (same as call, since losing whole stack)
        let fold_reward = -604.0_f32;
        let call_reward = -1463.0_f32;
        let invalid_penalty = -1463.0_f32;

        for _ in 0..24 {
            let mut rewards = vec![invalid_penalty; NUM_ACTION_INDICES];
            rewards[0] = fold_reward; // Fold
            rewards[1] = call_reward; // Call
            matcher.update_regret(&rewards);
        }

        let weights = matcher.best_weight();
        let fold_weight = weights[0];
        let call_weight = weights[1];

        println!(
            "PCFR+ after 24 iterations: fold={:.6}, call={:.6}",
            fold_weight, call_weight
        );

        // Fold should dominate
        assert!(
            fold_weight > 0.99,
            "Fold should have >99% weight, got {:.4}%",
            fold_weight * 100.0
        );
        assert!(
            call_weight < 0.01,
            "Call should have <1% weight, got {:.4}%",
            call_weight * 100.0
        );
    }

    // -------------------------------------------------------------------------
    // Fast-forward helper tests
    //
    // These exercise the free `fast_forward_*` helpers directly, without
    // going through a CFRAgent. They verify pot distribution, tie-splitting,
    // and the deal-path from mid-hand rounds.
    // -------------------------------------------------------------------------

    /// River, heads-up: P0 has a pair of 9s, P1 has K-high. P0 is all-in for
    /// 500. P1 calls → P1 loses their remaining stack; P0 wins the pot.
    #[test]
    fn fast_forward_river_call_loses_with_worse_hand() {
        use crate::arena::game_state::{Round, RoundData};
        use crate::core::{Card, Hand, PlayerBitSet, Suit, Value};

        let board_cards = vec![
            Card::new(Value::Four, Suit::Spade),
            Card::new(Value::Four, Suit::Diamond),
            Card::new(Value::Nine, Suit::Diamond),
            Card::new(Value::Ace, Suit::Heart),
            Card::new(Value::Jack, Suit::Diamond),
        ];

        let p0_hole = vec![
            Card::new(Value::Nine, Suit::Heart),
            Card::new(Value::Queen, Suit::Spade),
        ];
        let p1_hole = vec![
            Card::new(Value::Seven, Suit::Spade),
            Card::new(Value::King, Suit::Spade),
        ];

        let mut p0_hand = Hand::new_with_cards(p0_hole);
        p0_hand.extend(board_cards.iter().copied());
        let mut p1_hand = Hand::new_with_cards(p1_hole);
        p1_hand.extend(board_cards.iter().copied());

        let stacks = vec![0.0, 300.0];
        let starting_stacks = vec![700.0, 700.0];
        let player_bet = vec![700.0, 400.0];
        let round_player_bet = vec![500.0, 0.0];
        let mut active = PlayerBitSet::new(2);
        active.disable(0);

        let round_data = RoundData::new_with_bets(10.0, active, 1, round_player_bet);

        let mut game_state = GameStateBuilder::new()
            .round(Round::River)
            .round_data(round_data)
            .stacks(stacks)
            .player_bet(player_bet)
            .big_blind(10.0)
            .small_blind(5.0)
            .hands(vec![p0_hand, p1_hand])
            .board(board_cards)
            .build()
            .unwrap();
        game_state.starting_stacks = starting_stacks.into();
        // Total pot reflects what's already in: 700 + 400 = 1100.
        game_state.total_pot = 1100.0;
        game_state.player_all_in = {
            let mut pbs = PlayerBitSet::new(2);
            pbs.disable(1);
            pbs
        };

        let mut rng = StdRng::seed_from_u64(7);

        // Calling with K-high: P1 pays 300 more, goes all-in, loses at showdown.
        let mut call_state = game_state.clone();
        fast_forward_apply_action(&mut call_state, &AgentAction::Call);
        fast_forward_run_to_showdown(&mut call_state, &mut rng);
        fast_forward_distribute_pot(&mut call_state);
        let call_reward = call_state.player_reward(1);
        assert!(
            call_reward < -699.0,
            "Calling with the losing hand should cost ~700 stack, got {}",
            call_reward
        );

        // Folding: P1 forfeits only what's already in the pot (400).
        let mut fold_state = game_state.clone();
        fast_forward_apply_action(&mut fold_state, &AgentAction::Fold);
        fast_forward_run_to_showdown(&mut fold_state, &mut rng);
        fast_forward_distribute_pot(&mut fold_state);
        let fold_reward = fold_state.player_reward(1);
        assert!(
            (fold_reward - (-400.0)).abs() < 0.01,
            "Folding should cost exactly the 400 already committed, got {}",
            fold_reward
        );

        // Folding should be strictly better than calling here.
        assert!(fold_reward > call_reward);
    }

    /// Tie: both players play the board on a paired board → pot split.
    #[test]
    fn fast_forward_split_pot_on_tie() {
        use crate::arena::game_state::{Round, RoundData};
        use crate::core::{Card, Hand, PlayerBitSet, Suit, Value};

        let board_cards = vec![
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::Ace, Suit::Diamond),
            Card::new(Value::King, Suit::Heart),
            Card::new(Value::Queen, Suit::Club),
            Card::new(Value::Jack, Suit::Diamond),
        ];
        // Both players play A-A-K-Q-J from the board.
        let p0_hole = vec![
            Card::new(Value::Two, Suit::Club),
            Card::new(Value::Three, Suit::Club),
        ];
        let p1_hole = vec![
            Card::new(Value::Two, Suit::Heart),
            Card::new(Value::Three, Suit::Spade),
        ];
        let mut p0 = Hand::new_with_cards(p0_hole);
        p0.extend(board_cards.iter().copied());
        let mut p1 = Hand::new_with_cards(p1_hole);
        p1.extend(board_cards.iter().copied());

        let round_data = RoundData::new_with_bets(10.0, PlayerBitSet::new(2), 0, vec![0.0, 0.0]);

        let mut gs = GameStateBuilder::new()
            .round(Round::River)
            .round_data(round_data)
            .stacks(vec![500.0, 500.0])
            .player_bet(vec![500.0, 500.0])
            .big_blind(10.0)
            .small_blind(5.0)
            .hands(vec![p0, p1])
            .board(board_cards)
            .build()
            .unwrap();
        gs.starting_stacks = vec![1000.0, 1000.0].into();
        gs.total_pot = 1000.0;

        let mut rng = StdRng::seed_from_u64(11);
        fast_forward_apply_action(&mut gs, &AgentAction::Call);
        fast_forward_run_to_showdown(&mut gs, &mut rng);
        fast_forward_distribute_pot(&mut gs);
        let reward = gs.player_reward(0);
        // 1000 pot split between both players → ~0 net.
        assert!(
            reward.abs() < 0.01,
            "split should yield ~0 reward, got {}",
            reward
        );
    }

    /// Fast-forward from the flop: the helper must deal the turn and river
    /// before running the showdown rather than panicking on an empty board.
    #[test]
    fn fast_forward_from_flop_deals_turn_and_river() {
        use crate::arena::game_state::{Round, RoundData};
        use crate::core::{Card, Hand, PlayerBitSet, Suit, Value};

        let board_cards = vec![
            Card::new(Value::Two, Suit::Spade),
            Card::new(Value::Seven, Suit::Diamond),
            Card::new(Value::Jack, Suit::Heart),
        ];
        let p0_hole = vec![
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::Ace, Suit::Diamond),
        ];
        let p1_hole = vec![
            Card::new(Value::King, Suit::Spade),
            Card::new(Value::Queen, Suit::Diamond),
        ];
        let mut p0 = Hand::new_with_cards(p0_hole);
        p0.extend(board_cards.iter().copied());
        let mut p1 = Hand::new_with_cards(p1_hole);
        p1.extend(board_cards.iter().copied());

        let round_data = RoundData::new_with_bets(10.0, PlayerBitSet::new(2), 0, vec![0.0, 0.0]);
        let mut gs = GameStateBuilder::new()
            .round(Round::Flop)
            .round_data(round_data)
            .stacks(vec![100.0, 100.0])
            .player_bet(vec![10.0, 10.0])
            .big_blind(10.0)
            .small_blind(5.0)
            .hands(vec![p0, p1])
            .board(board_cards)
            .build()
            .unwrap();
        gs.starting_stacks = vec![110.0, 110.0].into();
        gs.total_pot = 20.0;

        let mut rng = StdRng::seed_from_u64(3);
        fast_forward_apply_action(&mut gs, &AgentAction::Call);
        fast_forward_run_to_showdown(&mut gs, &mut rng);
        fast_forward_distribute_pot(&mut gs);
        // With random turn/river cards we don't assert an exact reward — the
        // point is that we reached showdown (board has 5 cards) instead of
        // panicking on an empty deck / missing cards.
        assert_eq!(gs.board.len(), 5);
    }

    /// Test that regret-based pruning produces the same correct result
    /// as the unpruned path. With enough iterations, the agent should
    /// still fold K-high facing an all-in.
    #[tokio::test(flavor = "current_thread")]
    async fn test_rbp_preserves_fold_decision() {
        use crate::arena::cfr::action_generator::{
            ConfigurableActionConfig, ConfigurableActionGenerator,
        };
        use crate::arena::game_state::Round;
        use crate::core::{Card, Hand, PlayerBitSet, Suit, Value};

        let board_cards = vec![
            Card::new(Value::Four, Suit::Spade),
            Card::new(Value::Four, Suit::Diamond),
            Card::new(Value::Nine, Suit::Diamond),
            Card::new(Value::Ace, Suit::Heart),
            Card::new(Value::Jack, Suit::Diamond),
        ];
        let p0_hole = vec![
            Card::new(Value::Nine, Suit::Heart),
            Card::new(Value::Queen, Suit::Spade),
        ];
        let p1_hole = vec![
            Card::new(Value::Seven, Suit::Spade),
            Card::new(Value::King, Suit::Spade),
        ];
        let mut p0_hand = Hand::new_with_cards(p0_hole);
        p0_hand.extend(board_cards.iter().copied());
        let mut p1_hand = Hand::new_with_cards(p1_hole);
        p1_hand.extend(board_cards.iter().copied());

        let stacks = vec![0.0, 300.0];
        let starting_stacks = vec![700.0, 700.0];
        let player_bet = vec![700.0, 400.0];
        let round_player_bet = vec![500.0, 0.0];
        let mut active = PlayerBitSet::new(2);
        active.disable(0);
        let round_data =
            crate::arena::game_state::RoundData::new_with_bets(10.0, active, 1, round_player_bet);

        let mut game_state = GameStateBuilder::new()
            .round(Round::River)
            .round_data(round_data)
            .stacks(stacks)
            .player_bet(player_bet)
            .big_blind(10.0)
            .small_blind(5.0)
            .hands(vec![p0_hand, p1_hand])
            .board(board_cards)
            .build()
            .unwrap();
        game_state.starting_stacks = starting_stacks.into();

        // Use 24 iterations — enough for RBP to kick in (warmup=3, reprobe every 4th).
        // Pruning should skip computing rewards for clearly-bad actions after warmup.
        let cfr_state = make_cfr_state(&game_state);
        let traversal_set = TraversalSet::new(game_state.num_players);

        let budget = budget_for_schedule(&[24, 3, 1]);
        let mut agent = CFRAgentBuilder::<ConfigurableActionGenerator>::new()
            .name("CFR-RBP-test")
            .player_idx(1)
            .cfr_state(cfr_state.clone())
            .traversal_set(traversal_set)
            .budget(budget)
            .action_gen_config(ConfigurableActionConfig::default())
            .build();

        let chosen = agent.act(0, &game_state).await;
        assert!(
            matches!(chosen, AgentAction::Fold),
            "RBP should preserve the fold decision for K-high vs pair, got {:?}",
            chosen
        );
    }

    /// Test that regret-based pruning actually activates by verifying
    /// that the pruning info bitset becomes sparse after warmup.
    #[tokio::test(flavor = "current_thread")]
    async fn test_rbp_reduces_active_actions() {
        use crate::arena::cfr::action_generator::{
            ConfigurableActionConfig, ConfigurableActionGenerator,
        };
        use crate::arena::game_state::Round;
        use crate::core::{Card, Hand, PlayerBitSet, Suit, Value};

        let board_cards = vec![
            Card::new(Value::Four, Suit::Spade),
            Card::new(Value::Four, Suit::Diamond),
            Card::new(Value::Nine, Suit::Diamond),
            Card::new(Value::Ace, Suit::Heart),
            Card::new(Value::Jack, Suit::Diamond),
        ];
        let p0_hole = vec![
            Card::new(Value::Nine, Suit::Heart),
            Card::new(Value::Queen, Suit::Spade),
        ];
        let p1_hole = vec![
            Card::new(Value::Seven, Suit::Spade),
            Card::new(Value::King, Suit::Spade),
        ];
        let mut p0_hand = Hand::new_with_cards(p0_hole);
        p0_hand.extend(board_cards.iter().copied());
        let mut p1_hand = Hand::new_with_cards(p1_hole);
        p1_hand.extend(board_cards.iter().copied());

        let stacks = vec![0.0, 300.0];
        let starting_stacks = vec![700.0, 700.0];
        let player_bet = vec![700.0, 400.0];
        let round_player_bet = vec![500.0, 0.0];
        let mut active = PlayerBitSet::new(2);
        active.disable(0);
        let round_data =
            crate::arena::game_state::RoundData::new_with_bets(10.0, active, 1, round_player_bet);

        let mut game_state = GameStateBuilder::new()
            .round(Round::River)
            .round_data(round_data)
            .stacks(stacks)
            .player_bet(player_bet)
            .big_blind(10.0)
            .small_blind(5.0)
            .hands(vec![p0_hand, p1_hand])
            .board(board_cards)
            .build()
            .unwrap();
        game_state.starting_stacks = starting_stacks.into();

        let cfr_state = make_cfr_state(&game_state);
        let traversal_set = TraversalSet::new(game_state.num_players);

        let budget = budget_for_schedule(&[24, 3, 1]);
        let mut agent = CFRAgentBuilder::<ConfigurableActionGenerator>::new()
            .name("CFR-RBP-sparse")
            .player_idx(1)
            .cfr_state(cfr_state.clone())
            .traversal_set(traversal_set)
            .budget(budget)
            .action_gen_config(ConfigurableActionConfig::default())
            .build();

        // Run exploration
        let _ = agent.act(0, &game_state).await;

        // After 24 iterations with clear fold > call, the active set
        // should have fewer actions than the total valid set.
        let target_node_idx = agent.target_node_idx().unwrap();
        let (active_set, num_updates) = agent.cfr_state.get_pruning_info(target_node_idx);

        println!(
            "After exploration: {} active actions, {} updates",
            active_set.count(),
            num_updates
        );

        // The regret matcher must have run long enough for pruning to be
        // meaningful — PRUNE_WARMUP=3, plus a few extra updates. In a
        // lopsided scenario the strategy-stability early exit may bail
        // before 24 iters, which is fine; we just need enough updates
        // that pruning had a chance to fire.
        assert!(
            num_updates >= 6,
            "Expected >= 6 updates (warmup + a few prunes), got {}",
            num_updates
        );

        // With a clear fold vs call scenario, not all actions should remain active.
        // Fold should dominate, so call and other actions should have 0 weight.
        // We expect at most 2 active actions (fold + possibly one other)
        // in this lopsided scenario.
        assert!(
            active_set.count() <= 3,
            "Expected <= 3 active actions after pruning, got {}. \
             In this lopsided fold-vs-call scenario, most actions should be pruned.",
            active_set.count()
        );
    }

    /// Test that multi-sample flop + exhaustive runout produces lower
    /// variance than single-sample for a preflop scenario.
    ///
    /// Runs both approaches many times with different RNG seeds and
    /// compares the standard deviation of the rewards.
    #[test]
    fn test_flop_sample_variance_vs_single_sample() {
        use crate::arena::game_state::Round;
        use crate::core::{Card, Hand, Suit, Value};
        use rand::SeedableRng;

        // Set up a preflop-like game state where both players are all-in
        // and we need to deal flop+turn+river.
        let p0_hole = vec![
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Spade),
        ];
        let p1_hole = vec![
            Card::new(Value::Queen, Suit::Heart),
            Card::new(Value::Jack, Suit::Heart),
        ];
        let p0_hand = Hand::new_with_cards(p0_hole);
        let p1_hand = Hand::new_with_cards(p1_hole);

        // stacks=0 + bets>0 in a non-Starting round → both players all-in
        let mut game_state = GameStateBuilder::new()
            .round(Round::DealFlop)
            .stacks(vec![0.0, 0.0])
            .player_bet(vec![500.0, 500.0])
            .big_blind(10.0)
            .small_blind(5.0)
            .hands(vec![p0_hand, p1_hand])
            .build()
            .unwrap();
        game_state.starting_stacks = vec![500.0, 500.0].into();

        let num_trials = 50;

        // Multi-sample flop + exhaustive runout (current approach)
        let multi_results: Vec<f32> = (0..num_trials)
            .map(|seed| {
                let mut rng = StdRng::seed_from_u64(seed);
                fast_forward_sample_flop_enumerate_runout(&game_state, 0, &mut rng)
            })
            .collect();

        // Single-sample baseline (old approach: random runout)
        let single_results: Vec<f32> = (0..num_trials)
            .map(|seed| {
                let mut gs = game_state.clone();
                let mut rng = StdRng::seed_from_u64(seed);
                fast_forward_run_to_showdown(&mut gs, &mut rng);
                fast_forward_distribute_pot(&mut gs);
                gs.player_reward(0)
            })
            .collect();

        fn std_dev(vals: &[f32]) -> f64 {
            let n = vals.len() as f64;
            let mean = vals.iter().map(|&v| v as f64).sum::<f64>() / n;
            let var = vals.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
            var.sqrt()
        }

        let multi_std = std_dev(&multi_results);
        let single_std = std_dev(&single_results);

        let multi_mean: f64 =
            multi_results.iter().map(|&v| v as f64).sum::<f64>() / num_trials as f64;
        let single_mean: f64 =
            single_results.iter().map(|&v| v as f64).sum::<f64>() / num_trials as f64;

        println!(
            "Multi-sample flop (k={}): mean={:.2}, std={:.2}",
            FLOP_SAMPLES, multi_mean, multi_std
        );
        println!(
            "Single-sample runout:     mean={:.2}, std={:.2}",
            single_mean, single_std
        );
        println!(
            "Variance reduction: {:.1}x",
            if multi_std > 0.0 {
                single_std / multi_std
            } else {
                f64::INFINITY
            }
        );

        // The multi-sample approach should have meaningfully lower variance.
        // With k=3 flops and exact runout, we expect at least 2x reduction.
        assert!(
            multi_std < single_std,
            "Multi-sample flop should have lower variance than single-sample. \
             multi_std={:.2}, single_std={:.2}",
            multi_std,
            single_std
        );

        // Both means should be in roughly the same ballpark (same underlying EV).
        // AKs vs QJs preflop is roughly 60/40, so EV for player 0 ≈ +200 (win 1000 * 0.6 - 500).
        // Allow a wide range since we're measuring with limited trials.
        assert!(
            (multi_mean - single_mean).abs() < 200.0,
            "Means should be broadly similar: multi={:.2}, single={:.2}",
            multi_mean,
            single_mean
        );
    }

    /// Test that multi-sample flop enumeration produces correct sign
    /// for a strongly dominated hand. AKs should beat 72o.
    #[test]
    fn test_flop_sample_dominated_hand() {
        use crate::arena::game_state::Round;
        use crate::core::{Card, Hand, Suit, Value};
        use rand::SeedableRng;

        let p0_hole = vec![
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Spade),
        ];
        let p1_hole = vec![
            Card::new(Value::Seven, Suit::Diamond),
            Card::new(Value::Two, Suit::Club),
        ];
        let p0_hand = Hand::new_with_cards(p0_hole);
        let p1_hand = Hand::new_with_cards(p1_hole);

        // stacks=0 + bets>0 in a non-Starting round → both players all-in
        let mut game_state = GameStateBuilder::new()
            .round(Round::DealFlop)
            .stacks(vec![0.0, 0.0])
            .player_bet(vec![500.0, 500.0])
            .big_blind(10.0)
            .small_blind(5.0)
            .hands(vec![p0_hand, p1_hand])
            .build()
            .unwrap();
        game_state.starting_stacks = vec![500.0, 500.0].into();

        // Average over multiple seeds — AKs should consistently beat 72o.
        let num_trials = 20;
        let total_reward: f64 = (0..num_trials)
            .map(|seed| {
                let mut rng = StdRng::seed_from_u64(seed + 100);
                fast_forward_sample_flop_enumerate_runout(&game_state, 0, &mut rng) as f64
            })
            .sum();
        let avg_reward = total_reward / num_trials as f64;

        println!("AKs vs 72o avg reward for AKs: {:.2}", avg_reward);

        // AKs vs 72o has ~66% equity, so EV ≈ 1000 * 0.66 - 500 = +160.
        // Should be solidly positive.
        assert!(
            avg_reward > 50.0,
            "AKs should have positive EV vs 72o, got {:.2}",
            avg_reward
        );
    }

    /// Compare variance and cost across different flop sample counts.
    /// Run with `--nocapture` to see the table:
    ///   cargo test --all-features test_flop_sample_count_comparison -- --nocapture
    #[test]
    fn test_flop_sample_count_comparison() {
        use crate::arena::game_state::Round;
        use crate::core::{Card, Hand, Suit, Value};
        use rand::SeedableRng;
        use std::time::Instant;

        let p0_hole = vec![
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Spade),
        ];
        let p1_hole = vec![
            Card::new(Value::Queen, Suit::Heart),
            Card::new(Value::Jack, Suit::Heart),
        ];
        let p0_hand = Hand::new_with_cards(p0_hole);
        let p1_hand = Hand::new_with_cards(p1_hole);

        let mut game_state = GameStateBuilder::new()
            .round(Round::DealFlop)
            .stacks(vec![0.0, 0.0])
            .player_bet(vec![500.0, 500.0])
            .big_blind(10.0)
            .small_blind(5.0)
            .hands(vec![p0_hand, p1_hand])
            .build()
            .unwrap();
        game_state.starting_stacks = vec![500.0, 500.0].into();

        let num_trials = 100;

        fn std_dev(vals: &[f32]) -> f64 {
            let n = vals.len() as f64;
            let mean = vals.iter().map(|&v| v as f64).sum::<f64>() / n;
            let var = vals.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
            var.sqrt()
        }

        println!(
            "\n{:<8} {:>10} {:>10} {:>12} {:>10}",
            "k", "mean", "std", "var_red", "time_us"
        );
        println!("{}", "-".repeat(56));

        // Single-sample baseline
        let start = Instant::now();
        let single_results: Vec<f32> = (0..num_trials)
            .map(|seed| {
                let mut gs = game_state.clone();
                let mut rng = StdRng::seed_from_u64(seed);
                fast_forward_run_to_showdown(&mut gs, &mut rng);
                fast_forward_distribute_pot(&mut gs);
                gs.player_reward(0)
            })
            .collect();
        let single_time = start.elapsed();
        let single_std = std_dev(&single_results);
        let single_mean: f64 =
            single_results.iter().map(|&v| v as f64).sum::<f64>() / num_trials as f64;

        println!(
            "{:<8} {:>10.2} {:>10.2} {:>12} {:>10}",
            "1-samp",
            single_mean,
            single_std,
            "baseline",
            single_time.as_micros()
        );

        // Test k = 1, 2, 3, 5, 8, 13
        for k in [1, 2, 3, 5, 8, 13] {
            let start = Instant::now();
            let results: Vec<f32> = (0..num_trials)
                .map(|seed| {
                    let mut rng = StdRng::seed_from_u64(seed);
                    fast_forward_sample_flop_enumerate_runout_n(&game_state, 0, &mut rng, k)
                })
                .collect();
            let elapsed = start.elapsed();

            let multi_std = std_dev(&results);
            let multi_mean: f64 =
                results.iter().map(|&v| v as f64).sum::<f64>() / num_trials as f64;
            let var_reduction = if multi_std > 0.0 {
                single_std / multi_std
            } else {
                f64::INFINITY
            };

            println!(
                "{:<8} {:>10.2} {:>10.2} {:>11.1}x {:>10}",
                format!("k={k}"),
                multi_mean,
                multi_std,
                var_reduction,
                elapsed.as_micros()
            );
        }

        // Sanity: k=3 should reduce variance vs single sample
        let k3_results: Vec<f32> = (0..num_trials)
            .map(|seed| {
                let mut rng = StdRng::seed_from_u64(seed);
                fast_forward_sample_flop_enumerate_runout_n(&game_state, 0, &mut rng, 3)
            })
            .collect();
        assert!(
            std_dev(&k3_results) < single_std,
            "k=3 should reduce variance vs single sample"
        );
    }

    /// Test helper: a `tracing-subscriber` layer that captures every
    /// `cfr_diag` event's field set into a shared `Vec<CapturedEvent>` so
    /// tests can assert on what the engine emitted.
    #[derive(Clone, Default)]
    struct CapturedEvent {
        depth: u64,
        stop_cause: String,
        final_iterations: u64,
        final_elapsed_us: u64,
        timer_armed: bool,
        actions_considered: u64,
        regret_series: String,
    }

    #[derive(Default)]
    struct CapturingDiagLayer {
        events: std::sync::Arc<std::sync::Mutex<Vec<CapturedEvent>>>,
    }

    impl CapturingDiagLayer {
        fn new() -> Self {
            Self::default()
        }
        fn events(&self) -> std::sync::Arc<std::sync::Mutex<Vec<CapturedEvent>>> {
            self.events.clone()
        }
    }

    impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for CapturingDiagLayer {
        fn on_event(
            &self,
            event: &tracing::Event<'_>,
            _ctx: tracing_subscriber::layer::Context<'_, S>,
        ) {
            if event.metadata().target() != "cfr_diag" {
                return;
            }
            let mut captured = CapturedEvent::default();
            struct V<'a>(&'a mut CapturedEvent);
            impl tracing::field::Visit for V<'_> {
                fn record_u64(&mut self, f: &tracing::field::Field, value: u64) {
                    match f.name() {
                        "depth" => self.0.depth = value,
                        "final_iterations" => self.0.final_iterations = value,
                        "final_elapsed_us" => self.0.final_elapsed_us = value,
                        "actions_considered" => self.0.actions_considered = value,
                        _ => {}
                    }
                }
                fn record_bool(&mut self, f: &tracing::field::Field, value: bool) {
                    if f.name() == "timer_armed" {
                        self.0.timer_armed = value;
                    }
                }
                fn record_debug(&mut self, f: &tracing::field::Field, value: &dyn std::fmt::Debug) {
                    match f.name() {
                        "stop_cause" => self.0.stop_cause = format!("{value:?}"),
                        "regret_series" => self.0.regret_series = format!("{value:?}"),
                        _ => {}
                    }
                }
            }
            event.record(&mut V(&mut captured));
            self.events.lock().unwrap().push(captured);
        }
    }

    /// When the budget allows N iterations with no deadline, the emitted
    /// summary must report stop_cause=budget_stop, final_iterations=N,
    /// and a regret_series with N entries.
    ///
    /// Uses `budget_for_schedule(&[5])`: `MostRestrictive([IterationCount(5),
    /// MaxWidth([1])])`. `MaxWidth` provides `Wave { width: 1 }` at depth 0
    /// and `IterationCount` provides `Pass` until the cap, after which it
    /// returns `Stop`. `MostRestrictive` returns `Wave` until the cap, then
    /// `Stop` — so the wave loop runs exactly 5 times and stops.
    #[tokio::test(flavor = "current_thread")]
    async fn diag_event_records_iteration_bound_stop() {
        use tracing_subscriber::layer::SubscriberExt;

        let layer = CapturingDiagLayer::new();
        let events = layer.events();
        let subscriber = tracing_subscriber::registry()
            .with(
                tracing_subscriber::filter::Targets::new()
                    .with_target("cfr_diag", tracing::Level::TRACE),
            )
            .with(layer);
        let _guard = tracing::subscriber::set_default(subscriber);

        let (game_state, cfr_state, traversal_set) = setup_tiny_heads_up();
        // budget_for_schedule(&[5]): 5 iterations at depth 0, fast-forward at
        // depth 1. MaxWidth provides Wave so the loop runs; IterationCount
        // provides Stop after 5 completions.
        let budget = budget_for_schedule(&[5]);

        let mut agent = CFRAgentBuilder::<ConfigurableActionGenerator>::new()
            .name("CFRAgent-iter-bound")
            .player_idx(game_state.to_act_idx())
            .cfr_state(cfr_state)
            .traversal_set(traversal_set)
            .action_gen_config(ConfigurableActionConfig::default())
            .budget(budget)
            .build();

        let _ = agent.act(0, &game_state).await;

        let events = events.lock().unwrap();
        let root = events
            .iter()
            .find(|e| e.depth == 0)
            .expect("expected at least one depth=0 event");
        assert_eq!(root.stop_cause, "budget_stop");
        assert_eq!(root.final_iterations, 5);
        let series_len = if root.regret_series == "[]" {
            0
        } else {
            root.regret_series.matches(',').count() + 1
        };
        assert_eq!(
            series_len, 5,
            "expected 5 entries in regret_series, got '{}'",
            root.regret_series
        );
    }

    /// A 1ms deadline against an absurd iteration cap must report
    /// stop_cause=deadline with final_iterations < cap.
    ///
    /// Uses `MostRestrictive([Deadline(1ms), MaxWidth([1,1,1])])`:
    /// `Deadline` emits `StartTimer` on the first wave, then `Pass`; `MaxWidth`
    /// provides `Wave { width: 1 }` at every depth. Once the timer fires the
    /// stop flag, the engine breaks with `StopCause::Deadline`.
    #[tokio::test(flavor = "current_thread")]
    async fn diag_event_records_deadline_stop() {
        use crate::arena::cfr::{Deadline, MostRestrictive};
        use tracing_subscriber::layer::SubscriberExt;

        let layer = CapturingDiagLayer::new();
        let events = layer.events();
        let subscriber = tracing_subscriber::registry()
            .with(
                tracing_subscriber::filter::Targets::new()
                    .with_target("cfr_diag", tracing::Level::TRACE),
            )
            .with(layer);
        let _guard = tracing::subscriber::set_default(subscriber);

        let (game_state, cfr_state, traversal_set) = setup_tiny_heads_up();
        // Deadline(1ms) arms the timer; MaxWidth([1,1,1]) provides Wave so the
        // loop keeps running until the stop flag flips.
        let budget: Arc<dyn Budget> = Arc::new(MostRestrictive::new(vec![
            Arc::new(Deadline::new(std::time::Duration::from_millis(1))),
            Arc::new(MaxWidth::new(vec![1, 1, 1])),
        ]));

        let mut agent = CFRAgentBuilder::<ConfigurableActionGenerator>::new()
            .name("CFRAgent-deadline")
            .player_idx(game_state.to_act_idx())
            .cfr_state(cfr_state)
            .traversal_set(traversal_set)
            .action_gen_config(ConfigurableActionConfig::default())
            .budget(budget)
            .build();

        let _ = agent.act(0, &game_state).await;

        let events = events.lock().unwrap();
        let root = events
            .iter()
            .find(|e| e.depth == 0)
            .expect("expected at least one depth=0 event");
        assert_eq!(root.stop_cause, "deadline");
        assert!(
            root.final_iterations < 1_000_000,
            "deadline should have stopped well before any theoretical cap, got {}",
            root.final_iterations
        );
    }

    /// A recursive budget with PerDepth iteration caps at depths 0 and 1
    /// must emit summary events from at least depth 0 AND depth 1.
    ///
    /// Uses `budget_for_schedule(&[2, 1])`: this is
    /// `MostRestrictive([PerDepth([IterationCount(2), IterationCount(1)],
    /// fallback), MaxWidth([1, 1])])`. MaxWidth provides `Wave` at each depth
    /// so sub-agents actually run; PerDepth caps iterations per depth.
    #[tokio::test(flavor = "current_thread")]
    async fn diag_event_emitted_at_every_depth() {
        use tracing_subscriber::layer::SubscriberExt;

        let layer = CapturingDiagLayer::new();
        let events = layer.events();
        let subscriber = tracing_subscriber::registry()
            .with(
                tracing_subscriber::filter::Targets::new()
                    .with_target("cfr_diag", tracing::Level::TRACE),
            )
            .with(layer);
        let _guard = tracing::subscriber::set_default(subscriber);

        let (game_state, cfr_state, traversal_set) = setup_tiny_heads_up();
        // budget_for_schedule(&[2, 1]): 2 waves at depth 0, 1 wave at depth 1,
        // fast-forward at depth 2. MaxWidth ensures sub-agents get Wave signals
        // so they recurse one level and emit their own cfr_diag events.
        let budget = budget_for_schedule(&[2, 1]);

        let mut agent = CFRAgentBuilder::<ConfigurableActionGenerator>::new()
            .name("CFRAgent-perdepth")
            .player_idx(game_state.to_act_idx())
            .cfr_state(cfr_state)
            .traversal_set(traversal_set)
            .action_gen_config(ConfigurableActionConfig::default())
            .budget(budget)
            .build();

        let _ = agent.act(0, &game_state).await;

        let events = events.lock().unwrap();
        let depths_seen: std::collections::BTreeSet<u64> = events.iter().map(|e| e.depth).collect();
        assert!(depths_seen.contains(&0), "expected a depth=0 event");
        assert!(
            depths_seen.contains(&1),
            "expected at least one depth=1 event from recursive sub-agents; saw depths {:?}",
            depths_seen
        );
    }

    /// With a very generous iteration cap, the strategy-stability early
    /// exit path must be reachable — at some point the per-wave L1
    /// strategy delta drops below `EARLY_EXIT_EPSILON` for
    /// `EARLY_EXIT_STABLE_ITERS` consecutive waves. Asserts the
    /// stop_cause="stable_strategy" string appears in the captured diag
    /// stream, proving the code path is wired and the field
    /// serialization round-trips.
    #[tokio::test(flavor = "current_thread")]
    async fn diag_event_records_stable_strategy_stop() {
        use tracing_subscriber::layer::SubscriberExt;

        let layer = CapturingDiagLayer::new();
        let events = layer.events();
        let subscriber = tracing_subscriber::registry()
            .with(
                tracing_subscriber::filter::Targets::new()
                    .with_target("cfr_diag", tracing::Level::TRACE),
            )
            .with(layer);
        let _guard = tracing::subscriber::set_default(subscriber);

        let (game_state, cfr_state, traversal_set) = setup_tiny_heads_up();
        // Cap of 4096 is enough for PCFR+ to stabilize at EPSILON=0.001
        // in this tiny heads-up setup; the cfr_diag tests are sensitive
        // to the production EPSILON, not arbitrary values.
        let budget = budget_for_schedule(&[4096, 1]);

        let mut agent = CFRAgentBuilder::<ConfigurableActionGenerator>::new()
            .name("CFRAgent-stable")
            .player_idx(game_state.to_act_idx())
            .cfr_state(cfr_state)
            .traversal_set(traversal_set)
            .action_gen_config(ConfigurableActionConfig::default())
            .budget(budget)
            .build();

        let _ = agent.act(0, &game_state).await;

        let events = events.lock().unwrap();
        let any_stable = events.iter().any(|e| e.stop_cause == "stable_strategy");
        assert!(
            any_stable,
            "expected at least one stable_strategy event across all depths; saw causes: {:?}",
            events.iter().map(|e| &e.stop_cause).collect::<Vec<_>>()
        );
    }

    /// Build a tiny heads-up preflop game state plus the shared CFR state and
    /// traversal set used by the budget tests below. Returns everything the
    /// caller needs to construct a `CFRAgentBuilder` with a custom budget.
    fn setup_tiny_heads_up() -> (GameState, CFRState, TraversalSet) {
        use crate::core::{Card, Hand, Suit, Value};

        let mut game_state = GameStateBuilder::new()
            .num_players_with_stack(2, 100.0)
            .blinds(10.0, 5.0)
            .build()
            .unwrap();

        // Advance to preflop and post blinds so the to-act player has cards.
        game_state.advance_round(); // Starting -> Ante
        game_state.advance_round(); // Ante -> DealPreflop
        game_state.advance_round(); // DealPreflop -> Preflop
        game_state.do_bet(5.0, true).unwrap();
        game_state.do_bet(10.0, true).unwrap();

        // Deterministic hole cards (independent of the dealer's RNG).
        let mut hand0 = Hand::default();
        hand0.insert(Card::new(Value::Ace, Suit::Spade));
        hand0.insert(Card::new(Value::King, Suit::Spade));
        let mut hand1 = Hand::default();
        hand1.insert(Card::new(Value::Queen, Suit::Heart));
        hand1.insert(Card::new(Value::Jack, Suit::Heart));
        game_state.hands[0] = hand0;
        game_state.hands[1] = hand1;

        let cfr_state = make_cfr_state(&game_state);
        let traversal_set = TraversalSet::new(game_state.num_players);

        (game_state, cfr_state, traversal_set)
    }

    /// An iteration budget of 1 must terminate `act` promptly (it runs at most
    /// ~1 exploration iteration) and still return a legal `AgentAction`. The
    /// point is termination under a tight budget without erroring.
    #[tokio::test(flavor = "current_thread")]
    async fn act_respects_iteration_budget() {
        let (game_state, cfr_state, traversal_set) = setup_tiny_heads_up();

        let budget: Arc<dyn Budget> = Arc::new(IterationCount::new(1));
        let mut agent = CFRAgentBuilder::<ConfigurableActionGenerator>::new()
            .name("CFRAgent-budget")
            .player_idx(game_state.to_act_idx())
            .cfr_state(cfr_state)
            .traversal_set(traversal_set)
            .action_gen_config(ConfigurableActionConfig::default())
            .budget(budget)
            .build();

        let action = agent.act(0, &game_state).await;

        // A valid action is one of the four AgentAction variants; matching
        // exhaustively documents that `act` returned a legal action rather than
        // hanging or erroring under the tight budget.
        match action {
            AgentAction::Fold | AgentAction::Call | AgentAction::Bet(_) | AgentAction::AllIn => {}
        }
    }

    /// The engine must populate `ExplorationStats::avg_regret` from the node's
    /// regret matcher: `None` on the very first budget check (no completed
    /// update yet), then `Some(_)` once the node has been updated. A recording
    /// budget captures every value the engine produces while bounding the wave
    /// loop to 8 waves at the root (since the loop is now budget-driven).
    #[tokio::test(flavor = "current_thread")]
    async fn engine_populates_avg_regret_after_updates() {
        use crate::arena::cfr::{ExplorationStats, NextStep};
        use std::sync::{Arc, Mutex};

        #[derive(Clone)]
        struct RecordingBudget {
            seen: Arc<Mutex<Vec<Option<f32>>>>,
        }
        impl Budget for RecordingBudget {
            fn next_step(&self, stats: &ExplorationStats) -> NextStep {
                self.seen.lock().unwrap().push(stats.avg_regret);
                // Bound the wave loop ourselves: 8 waves at the root, then stop.
                // Deeper agents (sub-sims) stop after their first wave so the
                // test stays fast.
                let cap = if stats.depth == 0 { 8 } else { 1 };
                if stats.iterations < cap {
                    NextStep::Wave { width: 1 }
                } else {
                    NextStep::Stop
                }
            }
        }

        let (game_state, cfr_state, traversal_set) = setup_tiny_heads_up();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let budget: Arc<dyn Budget> = Arc::new(RecordingBudget { seen: seen.clone() });

        let mut agent = CFRAgentBuilder::<ConfigurableActionGenerator>::new()
            .name("CFRAgent-record")
            .player_idx(game_state.to_act_idx())
            .cfr_state(cfr_state)
            .traversal_set(traversal_set)
            .action_gen_config(ConfigurableActionConfig::default())
            .budget(budget)
            .build();

        let _ = agent.act(0, &game_state).await;

        let seen = seen.lock().unwrap();
        assert!(
            !seen.is_empty(),
            "the budget must be consulted at least once"
        );
        assert_eq!(
            seen[0], None,
            "the first budget check happens before any completed update"
        );
        assert!(
            seen.iter().any(Option::is_some),
            "avg_regret must be populated once the root node has been updated"
        );
    }

    /// Solve a small tree on a real multi-threaded runtime and assert that
    /// `act` returns a legal `AgentAction` without panicking. A panic inside a
    /// spawned task re-panics at the `JoinHandle::await` site and would fail the
    /// test, so a clean return proves no spawned work panicked. The solve is
    /// repeated a small number of times to exercise scheduling variation.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn multi_thread_exploration_is_sound() {
        use crate::core::{Card, Hand, Suit, Value};

        for run in 0u64..10 {
            let mut game_state = GameStateBuilder::new()
                .num_players_with_stack(2, 100.0)
                .blinds(10.0, 5.0)
                .build()
                .unwrap();

            game_state.advance_round();
            game_state.advance_round();
            game_state.advance_round();
            game_state.do_bet(5.0, true).unwrap();
            game_state.do_bet(10.0, true).unwrap();

            let mut hand0 = Hand::default();
            hand0.insert(Card::new(Value::Ace, Suit::Spade));
            hand0.insert(Card::new(Value::King, Suit::Spade));
            let mut hand1 = Hand::default();
            hand1.insert(Card::new(Value::Queen, Suit::Heart));
            hand1.insert(Card::new(Value::Jack, Suit::Heart));
            game_state.hands[0] = hand0;
            game_state.hands[1] = hand1;

            let cfr_state = make_cfr_state(&game_state);
            let traversal_set = TraversalSet::new(game_state.num_players);
            // Keep depth modest to stay fast while genuinely exercising spawning.
            let budget = budget_for_schedule(&[4, 1]);

            let mut agent = CFRAgentBuilder::<ConfigurableActionGenerator>::new()
                .name(format!("CFRAgent-mt-{run}"))
                .player_idx(game_state.to_act_idx())
                .cfr_state(cfr_state)
                .traversal_set(traversal_set)
                .budget(budget)
                .action_gen_config(ConfigurableActionConfig::default())
                .build();

            let action = agent.act(0, &game_state).await;

            // Assert legality — any of the four variants is valid.
            match action {
                AgentAction::Fold
                | AgentAction::Call
                | AgentAction::Bet(_)
                | AgentAction::AllIn => {}
            }
        }
    }

    /// M=1 structural test: with `wave_width == 1`, the budget alone governs the
    /// wave count. An `IterationCount(5)` budget must produce exactly 5 waves,
    /// i.e. 5 matcher updates at the root node (5 waves => 5 updates).
    #[tokio::test(flavor = "current_thread")]
    async fn wave_loop_runs_exactly_budget_waves_at_m_one() {
        let (game_state, cfr_state, traversal_set) = setup_tiny_heads_up();

        // M=1 (sequential waves), recurse one level so the root explores but
        // children fast-forward (keeps the test cheap).
        let budget: Arc<dyn Budget> = Arc::new(MostRestrictive::new(vec![
            Arc::new(IterationCount::new(5)),
            Arc::new(MaxWidth::new(vec![1])),
        ]));

        let mut agent = CFRAgentBuilder::<ConfigurableActionGenerator>::new()
            .name("CFRAgent-m1")
            .player_idx(game_state.to_act_idx())
            .cfr_state(cfr_state.clone())
            .traversal_set(traversal_set)
            .budget(budget)
            .action_gen_config(ConfigurableActionConfig::default())
            .build();

        // Drive the engine directly (no forced action): ensure the node + its
        // regret matcher exist, then run the wave loop.
        agent.ensure_target_node();
        agent.ensure_regret_matcher();
        let target_node_idx = agent.target_node_idx().unwrap();
        agent.explore_all_actions(&game_state).await;

        // `get_pruning_info` returns (active set, num_updates).
        let (_active, num_updates) = cfr_state.get_pruning_info(target_node_idx);
        assert_eq!(
            num_updates, 5,
            "five waves at M=1 must produce exactly five matcher updates"
        );
    }

    /// Inline-path structural test: with a size-1 `InFlightLimiter`, only one
    /// `try_acquire` can ever succeed at a time, so most samples take the inline
    /// path. The engine must still complete every wave and apply the budgeted
    /// number of updates (this replaces coverage lost when the forced-inline
    /// determinism test was removed in Phase 0).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn wave_loop_completes_with_size_one_limiter() {
        use tokio::sync::Semaphore;

        let (game_state, cfr_state, traversal_set) = setup_tiny_heads_up();

        let budget: Arc<dyn Budget> = Arc::new(MostRestrictive::new(vec![
            Arc::new(IterationCount::new(3)),
            Arc::new(MaxWidth::new(vec![1])),
        ]));
        // A semaphore with a single permit: at most one sample is ever spawned;
        // the rest fall back to the inline path.
        let limiter: Arc<Semaphore> = Arc::new(Semaphore::new(1));

        let mut agent = CFRAgentBuilder::<ConfigurableActionGenerator>::new()
            .name("CFRAgent-inline")
            .player_idx(game_state.to_act_idx())
            .cfr_state(cfr_state.clone())
            .traversal_set(traversal_set)
            .budget(budget)
            .limiter(limiter)
            .action_gen_config(ConfigurableActionConfig::default())
            .build();

        agent.ensure_target_node();
        agent.ensure_regret_matcher();
        let target_node_idx = agent.target_node_idx().unwrap();
        agent.explore_all_actions(&game_state).await;

        let (_active, num_updates) = cfr_state.get_pruning_info(target_node_idx);
        assert_eq!(
            num_updates, 3,
            "the wave loop must complete every budgeted wave even on the inline path"
        );
    }

    /// Real-time enforcement: a depth-0 agent whose budget would otherwise loop
    /// forever (`MaxWidth` recursing three levels with no `IterationCount` cap)
    /// must still return promptly once the wall-clock `Deadline` elapses. The
    /// `Deadline` leaf emits `StartTimer` at the root; the engine arms a stop
    /// flag, and every recursive level breaks at its next wave boundary. `act`
    /// returns ~at the deadline with a best-known (legal) action.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn act_returns_within_act_deadline() {
        use crate::arena::cfr::Deadline;

        let (game_state, cfr_state, traversal_set) = setup_tiny_heads_up();

        // MaxWidth([1,1,1]) recurses three levels with no iteration cap, so
        // only the deadline can terminate exploration.
        let budget: Arc<dyn Budget> = Arc::new(MostRestrictive::new(vec![
            Arc::new(Deadline::new(std::time::Duration::from_millis(150))),
            Arc::new(MaxWidth::new(vec![1, 1, 1])),
        ]));

        let mut agent = CFRAgentBuilder::<ConfigurableActionGenerator>::new()
            .name("CFRAgent-deadline")
            .player_idx(game_state.to_act_idx())
            .cfr_state(cfr_state)
            .traversal_set(traversal_set)
            .budget(budget)
            .action_gen_config(ConfigurableActionConfig::default())
            .build();

        let start = std::time::Instant::now();
        let action = agent.act(0, &game_state).await;
        let elapsed = start.elapsed();

        assert!(
            elapsed < std::time::Duration::from_secs(2),
            "act must stop ~at the 150ms deadline (unbounded budget would loop forever), \
             but took {elapsed:?}"
        );

        // A best-known action is returned, never a panic. Any of the four
        // variants is legal.
        match action {
            AgentAction::Fold | AgentAction::Call | AgentAction::Bet(_) | AgentAction::AllIn => {}
        }
    }

    /// Run one wave of CFR exploration on a Turn game state (4 board cards
    /// dealt, 1 card to deal = the river) with `RecursionConfig::new(0, vec![1])`
    /// (fast-forward at depth 0) and `IterationCount::new(1)`.
    ///
    /// Returns the post-update `best_weight()` vector bit-cast to `Vec<u32>`
    /// for exact equality comparison. The Turn scenario causes
    /// `fast_forward_advance_betting` to advance the state to `DealRiver`
    /// (`cards_needed = 1`) which routes to `fast_forward_enumerate_showdowns` —
    /// the RNG-independent enumeration branch.
    async fn run_turn_enumeration_once() -> Vec<u32> {
        use crate::arena::game_state::{Round, RoundData};
        use crate::core::{Card, Hand, PlayerBitSet, Suit, Value};

        // Board: Ah Kd Qc Js (4 cards dealt — Turn is complete).
        let board_cards = vec![
            Card::new(Value::Ace, Suit::Heart),
            Card::new(Value::King, Suit::Diamond),
            Card::new(Value::Queen, Suit::Club),
            Card::new(Value::Jack, Suit::Spade),
        ];

        // P0 has pocket aces → trip aces with the Ah on board (the nuts here).
        let p0_hole = vec![
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::Ace, Suit::Diamond),
        ];
        // P1 has 9h 2c → barely misses the board.
        let p1_hole = vec![
            Card::new(Value::Nine, Suit::Heart),
            Card::new(Value::Two, Suit::Club),
        ];

        let mut p0 = Hand::new_with_cards(p0_hole);
        p0.extend(board_cards.iter().copied());
        let mut p1 = Hand::new_with_cards(p1_hole);
        p1.extend(board_cards.iter().copied());

        // Both players active (need action), P1 to act, equal bets so far.
        let round_data = RoundData::new_with_bets(10.0, PlayerBitSet::new(2), 1, vec![0.0, 0.0]);
        let mut game_state = GameStateBuilder::new()
            .round(Round::Turn)
            .round_data(round_data)
            .stacks(vec![100.0, 100.0])
            .player_bet(vec![50.0, 50.0])
            .big_blind(10.0)
            .small_blind(5.0)
            .hands(vec![p0, p1])
            .board(board_cards)
            .build()
            .unwrap();
        game_state.starting_stacks = vec![150.0, 150.0].into();
        game_state.total_pot = 100.0;

        // Empty MaxWidth (depth 0 ≥ len 0) returns FastForward at depth 0.
        // One wave (IterationCount(1)) → fast-forward computation per action.
        // After fast_forward_advance_betting the state reaches DealRiver
        // (cards_needed=1) → fast_forward_enumerate_showdowns — no RNG.
        let cfr_state = make_cfr_state(&game_state);
        let traversal_set = TraversalSet::new(game_state.num_players);
        let budget: Arc<dyn Budget> = Arc::new(MostRestrictive::new(vec![
            Arc::new(IterationCount::new(1)),
            Arc::new(MaxWidth::new(vec![])),
        ]));

        let mut agent = CFRAgentBuilder::<BasicCFRActionGenerator>::new()
            .name("CFR-enum-stability")
            .player_idx(1) // P1 acts
            .cfr_state(cfr_state.clone())
            .traversal_set(traversal_set)
            .budget(budget)
            .action_gen_config(())
            .build();

        agent.ensure_target_node();
        agent.ensure_regret_matcher();
        let target_node_idx = agent.target_node_idx().unwrap();
        agent.explore_all_actions(&game_state).await;

        // Read back the post-update strategy weights as exact bit patterns so
        // two runs can be compared with assert_eq! (f32 has no Eq).
        cfr_state.with_node_data(target_node_idx, |node_data| {
            let matcher = get_regret_matcher_from_node(node_data).unwrap();
            matcher.best_weight().iter().map(|w| w.to_bits()).collect()
        })
    }

    /// The ≤2-card board-enumeration path (`fast_forward_enumerate_showdowns`,
    /// called when `cards_needed ≤ 2`) does NO sampling — it iterates a fixed
    /// set of card combinations from the remaining deck. Because the set of
    /// remaining cards is determined solely by the fixed game state (holes +
    /// board), the reward signal and therefore the PCFR+ update are fully
    /// deterministic.
    ///
    /// This test documents and locks that invariant: with `max_recursion_depth=0`
    /// (fast-forward at the root) on a Turn game state (4 board cards → 1 card
    /// to deal = `DealRiver`, `cards_needed=1`), two independent runs of the
    /// 1-wave exploration produce bit-identical `best_weight` vectors with no
    /// seed — no RNG is consulted on the enumeration path.
    #[tokio::test(flavor = "current_thread")]
    async fn enumeration_path_is_value_stable_without_seed() {
        let a = run_turn_enumeration_once().await;
        let b = run_turn_enumeration_once().await;
        assert_eq!(
            a, b,
            "the ≤2-card enumeration path is RNG-free, so best_weight vectors \
             must be bit-identical across two independent runs with no seed"
        );
    }

    /// Without a `Deadline` leaf and with a terminating `IterationCount`
    /// budget, the agent completes exactly its budgeted waves — no timer is
    /// armed, the stop flag never flips, and the wave loop runs to its
    /// iteration cap.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn no_act_deadline_completes_budgeted_waves() {
        let (game_state, cfr_state, traversal_set) = setup_tiny_heads_up();

        // Recurse one level (root explores, children fast-forward) so the run is
        // cheap, and bound the root to exactly 5 waves via the budget.
        let budget: Arc<dyn Budget> = Arc::new(MostRestrictive::new(vec![
            Arc::new(IterationCount::new(5)),
            Arc::new(MaxWidth::new(vec![1])),
        ]));

        let mut agent = CFRAgentBuilder::<ConfigurableActionGenerator>::new()
            .name("CFRAgent-no-deadline")
            .player_idx(game_state.to_act_idx())
            .cfr_state(cfr_state.clone())
            .traversal_set(traversal_set)
            .budget(budget)
            .action_gen_config(ConfigurableActionConfig::default())
            .build();

        // Drive exploration directly so we can assert the wave count exactly.
        agent.ensure_target_node();
        agent.ensure_regret_matcher();
        let target_node_idx = agent.target_node_idx().unwrap();
        agent.explore_all_actions(&game_state).await;

        let (_active, num_updates) = cfr_state.get_pruning_info(target_node_idx);
        assert_eq!(
            num_updates, 5,
            "without a deadline the wave loop runs to its iteration cap: \
             all five budgeted waves complete and apply their updates"
        );
    }

    /// Smoke test: `explore_all_actions` completes without panicking when the
    /// agent is built with `UniformRandomEstimator`. This exercises the
    /// per-act estimate + per-wave world-sampling paths introduced in Task 10.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn explore_runs_with_uniform_estimator() {
        use crate::arena::hand_estimator::UniformRandomEstimator;

        let (game_state, cfr_state, traversal_set) = setup_tiny_heads_up();

        let budget = budget_for_schedule(&[4, 2, 1]);

        let mut agent = CFRAgentBuilder::<BasicCFRActionGenerator>::new()
            .name("uniform-test")
            .player_idx(game_state.to_act_idx())
            .cfr_state(cfr_state)
            .traversal_set(traversal_set)
            .budget(budget)
            .action_gen_config(())
            .estimator(Arc::new(UniformRandomEstimator))
            .build();

        agent.ensure_target_node();
        agent.ensure_regret_matcher();
        agent.explore_all_actions(&game_state).await;
        // Test passes if no panic occurs.
    }

    #[tokio::test]
    async fn historian_present_only_when_estimator_needs_history() {
        use crate::arena::Agent;
        use std::sync::Arc;

        let game_state = crate::arena::GameStateBuilder::default()
            .num_players_with_stack(2, 100.0)
            .big_blind(2.0)
            .build()
            .unwrap();
        let cfr_state = make_cfr_state(&game_state);
        let traversal_set = TraversalSet::new(game_state.num_players);

        // Default (KnownHands) does not need history → no historian.
        let agent = CFRAgentBuilder::<BasicCFRActionGenerator>::new()
            .name("no-hist")
            .player_idx(0)
            .cfr_state(cfr_state.clone())
            .traversal_set(traversal_set.clone())
            .action_gen_config(())
            .build();
        assert!(agent.historian().is_none());

        // A needs_history estimator → historian present.
        let agent2 = CFRAgentBuilder::<BasicCFRActionGenerator>::new()
            .name("hist")
            .player_idx(0)
            .cfr_state(cfr_state)
            .traversal_set(traversal_set)
            .action_gen_config(())
            .estimator(Arc::new(HistoryNeedingStub::default()))
            .build();
        assert!(agent2.historian().is_some());
    }
}
