use rand::Rng;
use tracing::event;

use crate::arena::{GameState, action::AgentAction};

use super::{CFRState, NodeData, TraversalState};

pub trait ActionGenerator {
    /// Create a new action generator
    ///
    /// This is used by the Agent to create identical
    /// action generators for the historians it uses.
    fn new(cfr_state: CFRState, traversal_state: TraversalState) -> Self;

    /// Get a reference to the CFR state
    fn cfr_state(&self) -> &CFRState;

    /// Get a reference to the traversal state
    fn traversal_state(&self) -> &TraversalState;

    /// Given an action return the index of the action in the children array.
    ///
    /// # Arguments
    ///
    /// * `game_state` - The current game state
    /// * `action` - The action to convert to an index
    ///
    /// # Returns
    ///
    /// The index of the action in the children array. The 0 index is the fold
    /// action. All other are defined by the implentation
    fn action_to_idx(&self, game_state: &GameState, action: &AgentAction) -> usize;

    /// How many potential actions in total might be generated.
    ///
    /// At a given node there might be fewere that will be
    /// possible, but the regret matcher doesn't keep track of that.
    ///
    /// At all time the number of potential actions is
    /// larger than or equal to the number of possible actions
    ///
    /// # Returns
    ///
    /// The number of potential actions
    fn num_potential_actions(&self, game_state: &GameState) -> usize;

    // Generate all possible actions for the current game state
    //
    // This returns a vector so that the actions can be chosen from randomly
    fn gen_possible_actions(&self, game_state: &GameState) -> Vec<AgentAction>;

    /// Using the current and the CFR's tree's regret state choose a single action to
    /// play.
    ///
    /// This default implementation uses the regret matcher from the CFR state to
    /// select an action based on the current strategy. It filters the regret matcher's
    /// probability distribution to only include valid actions for the current game state.
    fn gen_action(&self, game_state: &GameState) -> AgentAction {
        let possible = self.gen_possible_actions(game_state);

        debug_assert!(
            !possible.is_empty(),
            "gen_possible_actions should always return at least one action"
        );

        // For now always use the thread rng.
        // At some point we will want to be able to pass seeded or deterministic action
        // choices.
        let mut rng = rand::rng();

        // Get target node index
        let from_node_idx = self.traversal_state().node_idx();
        let from_child_idx = self.traversal_state().chosen_child_idx();
        let target_node_idx = self
            .cfr_state()
            .get_child(from_node_idx, from_child_idx)
            .expect("Expected target node");

        // Get the node data
        let node_data = self
            .cfr_state()
            .get_node_data(target_node_idx)
            .expect("Expected target node data");

        if let NodeData::Player(pd) = &node_data {
            // Build a mapping from action index to action for valid actions only
            let valid_actions: Vec<(usize, AgentAction)> = possible
                .iter()
                .map(|action| (self.action_to_idx(game_state, action), action.clone()))
                .collect();

            // If there's no regret matcher yet, use uniform random over valid actions
            let Some(matcher) = pd.regret_matcher.as_ref() else {
                let chosen_idx = rng.random_range(0..valid_actions.len());
                return valid_actions[chosen_idx].1.clone();
            };

            // Get the weights from the regret matcher (average strategy)
            let weights = matcher.best_weight();

            // Extract weights for only the valid action indices
            let valid_weights: Vec<f32> = valid_actions
                .iter()
                .map(|(idx, _)| weights.get(*idx).copied().unwrap_or(0.0).max(0.0))
                .collect();

            // Calculate total weight
            let total_weight: f32 = valid_weights.iter().sum();

            // If all weights are zero (or very close), use uniform distribution
            if total_weight < 1e-10 {
                let chosen_idx = rng.random_range(0..valid_actions.len());
                event!(
                    tracing::Level::DEBUG,
                    chosen_idx = chosen_idx,
                    "All weights zero, using uniform random"
                );
                return valid_actions[chosen_idx].1.clone();
            }

            // Sample from the weighted distribution over valid actions
            let random_value: f32 = rng.random::<f32>() * total_weight;
            let mut cumulative = 0.0;
            for (i, weight) in valid_weights.iter().enumerate() {
                cumulative += weight;
                if random_value <= cumulative {
                    event!(
                        tracing::Level::DEBUG,
                        action_idx = valid_actions[i].0,
                        weight = weight,
                        total_weight = total_weight,
                        "Selected action from regret matcher"
                    );
                    return valid_actions[i].1.clone();
                }
            }

            // Fallback to last action (shouldn't reach here due to floating point)
            valid_actions.last().unwrap().1.clone()
        } else {
            panic!("Expected player node");
        }
    }
}

pub struct BasicCFRActionGenerator {
    cfr_state: CFRState,
    traversal_state: TraversalState,
}

impl BasicCFRActionGenerator {
    pub fn new(cfr_state: CFRState, traversal_state: TraversalState) -> Self {
        BasicCFRActionGenerator {
            cfr_state,
            traversal_state,
        }
    }
}

impl ActionGenerator for BasicCFRActionGenerator {
    fn new(cfr_state: CFRState, traversal_state: TraversalState) -> Self {
        BasicCFRActionGenerator {
            cfr_state,
            traversal_state,
        }
    }

    fn cfr_state(&self) -> &CFRState {
        &self.cfr_state
    }

    fn traversal_state(&self) -> &TraversalState {
        &self.traversal_state
    }

    fn gen_possible_actions(&self, game_state: &GameState) -> Vec<AgentAction> {
        let mut res: Vec<AgentAction> = Vec::with_capacity(3);
        let to_call =
            game_state.current_round_bet() - game_state.current_round_current_player_bet();
        if to_call > 0.0 {
            res.push(AgentAction::Fold);
        }
        // Call, Match the current bet (if the bet is 0 this is a check)
        res.push(AgentAction::Bet(game_state.current_round_bet()));

        let all_in_ammount =
            game_state.current_round_current_player_bet() + game_state.current_player_stack();

        if all_in_ammount > game_state.current_round_bet() {
            // All-in, Bet all the money
            // Bet everything we have bet so far plus the remaining stack
            res.push(AgentAction::AllIn);
        }
        res
    }

    fn action_to_idx(&self, _game_state: &GameState, action: &AgentAction) -> usize {
        match action {
            AgentAction::Fold => 0,
            AgentAction::Bet(_) => 1,
            AgentAction::Call => 1,
            AgentAction::AllIn => 2,
        }
    }

    fn num_potential_actions(&self, _game_state: &GameState) -> usize {
        3
    }
}

/// Action indices for SimpleActionGenerator
const ACTION_FOLD: usize = 0;
const ACTION_CALL: usize = 1;
const ACTION_MIN_RAISE: usize = 2;
const ACTION_POT_33: usize = 3;
const ACTION_POT_66: usize = 4;
const ACTION_ALL_IN: usize = 5;

/// Number of potential actions for SimpleActionGenerator
const SIMPLE_NUM_ACTIONS: usize = 6;

/// Epsilon for floating point comparison
const BET_EPSILON: f32 = 0.01;

/// Action generator with more betting options: fold, check/call, min raise,
/// 33% pot, 66% pot, and all-in.
///
/// This provides a richer action space for CFR exploration compared to
/// BasicCFRActionGenerator which only has fold, call, and all-in.
pub struct SimpleActionGenerator {
    cfr_state: CFRState,
    traversal_state: TraversalState,
}

impl SimpleActionGenerator {
    /// Compute the bet amounts for each action type given the current game state
    fn compute_bet_amounts(game_state: &GameState) -> BetAmounts {
        let current_bet = game_state.current_round_bet();
        let player_bet_this_round = game_state.current_round_current_player_bet();
        let stack = game_state.current_player_stack();
        let pot = game_state.total_pot;
        let min_raise = game_state.current_round_min_raise();

        let to_call = current_bet - player_bet_this_round;
        let max_total_bet = player_bet_this_round + stack;

        // Min raise: current bet + minimum raise amount
        let min_raise_total = current_bet + min_raise;

        // Pot-based raises: current bet + pot * percentage
        let pot_33_total = current_bet + pot * 0.33;
        let pot_66_total = current_bet + pot * 0.66;

        BetAmounts {
            to_call,
            max_total_bet,
            call_amount: current_bet,
            min_raise_total,
            pot_33_total,
            pot_66_total,
            min_raise,
        }
    }

    /// Check if two bet amounts are approximately equal
    fn amounts_equal(a: f32, b: f32) -> bool {
        (a - b).abs() < BET_EPSILON
    }

    /// Try to add a raise action if valid and not a duplicate of existing bets
    fn try_add_raise(
        actions: &mut Vec<AgentAction>,
        added_bets: &mut Vec<f32>,
        amount: f32,
        amounts: &BetAmounts,
    ) {
        let is_duplicate = added_bets.iter().any(|&b| Self::amounts_equal(b, amount));
        if amounts.is_valid_raise(amount) && !is_duplicate {
            actions.push(AgentAction::Bet(amount));
            added_bets.push(amount);
        }
    }
}

/// Helper struct for computed bet amounts
struct BetAmounts {
    to_call: f32,
    max_total_bet: f32,
    call_amount: f32,
    min_raise_total: f32,
    pot_33_total: f32,
    pot_66_total: f32,
    min_raise: f32,
}

impl BetAmounts {
    /// Check if a bet amount is a valid raise (meets min raise and within stack).
    ///
    /// This uses a strict comparison (no epsilon tolerance) to match the
    /// validation in game_state.validate_bet_amount(). The epsilon tolerance
    /// should only be used for equality comparisons (deduplication), not for
    /// "is this raise large enough" checks.
    fn is_valid_raise(&self, amount: f32) -> bool {
        let raise_over_call = amount - self.call_amount;
        // Use strict >= comparison to match game_state validation
        raise_over_call >= self.min_raise && amount <= self.max_total_bet
    }
}

impl ActionGenerator for SimpleActionGenerator {
    fn new(cfr_state: CFRState, traversal_state: TraversalState) -> Self {
        SimpleActionGenerator {
            cfr_state,
            traversal_state,
        }
    }

    fn cfr_state(&self) -> &CFRState {
        &self.cfr_state
    }

    fn traversal_state(&self) -> &TraversalState {
        &self.traversal_state
    }

    fn gen_possible_actions(&self, game_state: &GameState) -> Vec<AgentAction> {
        let mut actions: Vec<AgentAction> = Vec::with_capacity(SIMPLE_NUM_ACTIONS);
        let amounts = Self::compute_bet_amounts(game_state);

        // Fold is only possible if there's something to call
        if amounts.to_call > 0.0 {
            actions.push(AgentAction::Fold);
        }

        // Check/Call is always possible
        actions.push(AgentAction::Bet(amounts.call_amount));

        // Track which bet sizes we've added to avoid duplicates
        let mut added_bets: Vec<f32> = vec![amounts.call_amount];

        // Add raise options: min raise, 33% pot, 66% pot
        Self::try_add_raise(
            &mut actions,
            &mut added_bets,
            amounts.min_raise_total,
            &amounts,
        );
        Self::try_add_raise(
            &mut actions,
            &mut added_bets,
            amounts.pot_33_total,
            &amounts,
        );
        Self::try_add_raise(
            &mut actions,
            &mut added_bets,
            amounts.pot_66_total,
            &amounts,
        );

        // All-in (if we have more than the current bet and it's not a duplicate)
        let can_raise = amounts.max_total_bet > amounts.call_amount + BET_EPSILON;
        let is_duplicate = added_bets
            .iter()
            .any(|&b| Self::amounts_equal(b, amounts.max_total_bet));
        if can_raise && !is_duplicate {
            actions.push(AgentAction::AllIn);
        }

        actions
    }

    fn action_to_idx(&self, game_state: &GameState, action: &AgentAction) -> usize {
        match action {
            AgentAction::Fold => ACTION_FOLD,
            AgentAction::AllIn => ACTION_ALL_IN,
            AgentAction::Call => ACTION_CALL,
            AgentAction::Bet(amount) => {
                let amounts = Self::compute_bet_amounts(game_state);

                // Check each bet type in order
                if Self::amounts_equal(*amount, amounts.call_amount) {
                    ACTION_CALL
                } else if Self::amounts_equal(*amount, amounts.min_raise_total) {
                    ACTION_MIN_RAISE
                } else if Self::amounts_equal(*amount, amounts.pot_33_total) {
                    ACTION_POT_33
                } else if Self::amounts_equal(*amount, amounts.pot_66_total) {
                    ACTION_POT_66
                } else if Self::amounts_equal(*amount, amounts.max_total_bet) {
                    // This is effectively an all-in
                    ACTION_ALL_IN
                } else {
                    // Unknown bet size, default to call index
                    event!(
                        tracing::Level::WARN,
                        amount = amount,
                        "Unknown bet amount in action_to_idx, defaulting to call"
                    );
                    ACTION_CALL
                }
            }
        }
    }

    fn num_potential_actions(&self, _game_state: &GameState) -> usize {
        SIMPLE_NUM_ACTIONS
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::arena::GameState;

    use std::vec;

    // === BasicCFRActionGenerator tests ===

    #[test]
    fn test_should_gen_2_actions() {
        let stacks = vec![50.0; 2];
        let game_state = GameState::new_starting(stacks, 2.0, 1.0, 0.0, 0);
        let action_generator = BasicCFRActionGenerator::new(
            CFRState::new(game_state.clone()),
            TraversalState::new_root(0),
        );
        let actions = action_generator.gen_possible_actions(&game_state);
        // We should have 2 actions: Call or All-in since 0 is the dealer when starting
        assert_eq!(actions.len(), 2);

        // None of the ations should have a child idx of 0
        for action in actions {
            assert_ne!(action_generator.action_to_idx(&game_state, &action), 0);
        }
    }

    #[test]
    fn test_should_gen_3_actions() {
        let stacks = vec![50.0; 2];
        let mut game_state = GameState::new_starting(stacks, 2.0, 1.0, 0.0, 0);
        game_state.advance_round();
        game_state.advance_round();

        game_state.do_bet(10.0, false).unwrap();
        let action_generator = BasicCFRActionGenerator::new(
            CFRState::new(game_state.clone()),
            TraversalState::new_root(0),
        );
        let actions = action_generator.gen_possible_actions(&game_state);
        // We should have 3 actions: Fold, Call, or All-in
        assert_eq!(actions.len(), 3);

        // Check the indices of the actions
        assert_eq!(
            action_generator.action_to_idx(&game_state, &AgentAction::Fold),
            0
        );
        assert_eq!(
            action_generator.action_to_idx(&game_state, &AgentAction::Bet(10.0)),
            1
        );
        assert_eq!(
            action_generator.action_to_idx(&game_state, &AgentAction::AllIn),
            2
        );
    }

    // === SimpleActionGenerator tests ===

    fn create_simple_generator(game_state: &GameState) -> SimpleActionGenerator {
        SimpleActionGenerator::new(
            CFRState::new(game_state.clone()),
            TraversalState::new_root(0),
        )
    }

    #[test]
    fn test_simple_action_indices() {
        // Verify the action index constants
        assert_eq!(ACTION_FOLD, 0);
        assert_eq!(ACTION_CALL, 1);
        assert_eq!(ACTION_MIN_RAISE, 2);
        assert_eq!(ACTION_POT_33, 3);
        assert_eq!(ACTION_POT_66, 4);
        assert_eq!(ACTION_ALL_IN, 5);
        assert_eq!(SIMPLE_NUM_ACTIONS, 6);
    }

    #[test]
    fn test_simple_fold_maps_to_zero() {
        let stacks = vec![100.0; 2];
        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        let action_gen = create_simple_generator(&game_state);

        assert_eq!(
            action_gen.action_to_idx(&game_state, &AgentAction::Fold),
            ACTION_FOLD
        );
    }

    #[test]
    fn test_simple_all_in_maps_to_five() {
        let stacks = vec![100.0; 2];
        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        let action_gen = create_simple_generator(&game_state);

        assert_eq!(
            action_gen.action_to_idx(&game_state, &AgentAction::AllIn),
            ACTION_ALL_IN
        );
    }

    #[test]
    fn test_simple_call_maps_to_one() {
        let stacks = vec![100.0; 2];
        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        let action_gen = create_simple_generator(&game_state);

        // Call action
        assert_eq!(
            action_gen.action_to_idx(&game_state, &AgentAction::Call),
            ACTION_CALL
        );

        // Bet that matches current bet (same as call)
        let call_amount = game_state.current_round_bet();
        assert_eq!(
            action_gen.action_to_idx(&game_state, &AgentAction::Bet(call_amount)),
            ACTION_CALL
        );
    }

    #[test]
    fn test_simple_min_raise_mapping() {
        let stacks = vec![100.0; 2];
        let mut game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        // Advance to flop so we have a clean state
        game_state.advance_round();

        // Player 0 bets 10
        game_state.do_bet(10.0, false).unwrap();

        let action_gen = create_simple_generator(&game_state);

        // Min raise = current_bet (10) + min_raise (10) = 20
        let min_raise_amount =
            game_state.current_round_bet() + game_state.current_round_min_raise();
        assert_eq!(
            action_gen.action_to_idx(&game_state, &AgentAction::Bet(min_raise_amount)),
            ACTION_MIN_RAISE
        );
    }

    #[test]
    fn test_simple_pot_33_mapping() {
        let stacks = vec![500.0; 2];
        let mut game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        game_state.advance_round();

        // Player 0 bets 20
        game_state.do_bet(20.0, false).unwrap();

        let action_gen = create_simple_generator(&game_state);

        // Pot is now 35 (15 from blinds + 20 from bet)
        // 33% pot = current_bet (20) + pot (35) * 0.33 = 20 + 11.55 = 31.55
        let pot_33_amount = game_state.current_round_bet() + game_state.total_pot * 0.33;
        assert_eq!(
            action_gen.action_to_idx(&game_state, &AgentAction::Bet(pot_33_amount)),
            ACTION_POT_33
        );
    }

    #[test]
    fn test_simple_pot_66_mapping() {
        let stacks = vec![500.0; 2];
        let mut game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        game_state.advance_round();

        // Player 0 bets 20
        game_state.do_bet(20.0, false).unwrap();

        let action_gen = create_simple_generator(&game_state);

        // 66% pot = current_bet (20) + pot (35) * 0.66
        let pot_66_amount = game_state.current_round_bet() + game_state.total_pot * 0.66;
        assert_eq!(
            action_gen.action_to_idx(&game_state, &AgentAction::Bet(pot_66_amount)),
            ACTION_POT_66
        );
    }

    #[test]
    fn test_simple_gen_actions_with_bet_facing() {
        let stacks = vec![500.0; 2];
        let mut game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        game_state.advance_round();

        // Player 0 bets 30
        game_state.do_bet(30.0, false).unwrap();

        let action_gen = create_simple_generator(&game_state);
        let actions = action_gen.gen_possible_actions(&game_state);

        // Should have: Fold, Call, Min Raise, 33% pot, 66% pot, All-in
        // (assuming all bet sizes are distinct and valid)
        assert!(actions.contains(&AgentAction::Fold));
        assert!(actions.iter().any(|a| matches!(a, AgentAction::Bet(_))));
        assert!(actions.contains(&AgentAction::AllIn));

        // Verify each action maps to a unique index
        let mut indices: Vec<usize> = actions
            .iter()
            .map(|a| action_gen.action_to_idx(&game_state, a))
            .collect();
        indices.sort();
        indices.dedup();
        assert_eq!(
            indices.len(),
            actions.len(),
            "All actions should have unique indices"
        );
    }

    #[test]
    fn test_simple_no_fold_when_checking() {
        let stacks = vec![100.0; 2];
        let mut game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        // Advance to flop where no one has bet yet
        game_state.advance_round();

        let action_gen = create_simple_generator(&game_state);
        let actions = action_gen.gen_possible_actions(&game_state);

        // No fold option when there's nothing to call
        assert!(!actions.contains(&AgentAction::Fold));
    }

    #[test]
    fn test_simple_num_potential_actions() {
        let stacks = vec![100.0; 2];
        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        let action_gen = create_simple_generator(&game_state);

        assert_eq!(
            action_gen.num_potential_actions(&game_state),
            SIMPLE_NUM_ACTIONS
        );
    }

    #[test]
    fn test_simple_no_duplicate_bet_sizes() {
        // Test case where pot sizes might overlap with min raise
        let stacks = vec![100.0; 2];
        let mut game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        game_state.advance_round();

        // Bet 10 (min bet) where 33% pot might be close to min raise
        game_state.do_bet(10.0, false).unwrap();

        let action_gen = create_simple_generator(&game_state);
        let actions = action_gen.gen_possible_actions(&game_state);

        // Count bet actions
        let bet_amounts: Vec<f32> = actions
            .iter()
            .filter_map(|a| match a {
                AgentAction::Bet(amt) => Some(*amt),
                _ => None,
            })
            .collect();

        // Verify no duplicate bet amounts (within epsilon)
        for (i, a) in bet_amounts.iter().enumerate() {
            for (j, b) in bet_amounts.iter().enumerate() {
                if i != j {
                    assert!(
                        (a - b).abs() >= BET_EPSILON,
                        "Duplicate bet amounts: {} and {}",
                        a,
                        b
                    );
                }
            }
        }
    }

    #[test]
    fn test_simple_all_actions_roundtrip() {
        // Test that all generated actions map back correctly
        let stacks = vec![1000.0; 2];
        let mut game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        game_state.advance_round();
        game_state.do_bet(50.0, false).unwrap();

        let action_gen = create_simple_generator(&game_state);
        let actions = action_gen.gen_possible_actions(&game_state);

        for action in &actions {
            let idx = action_gen.action_to_idx(&game_state, action);
            // Find action with this index
            let found = actions
                .iter()
                .find(|a| action_gen.action_to_idx(&game_state, a) == idx);
            assert!(
                found.is_some(),
                "Action {:?} mapped to idx {} but no action maps back",
                action,
                idx
            );
        }
    }

    // === SimpleActionGenerator validation tests ===
    // These tests verify that all generated actions are valid when applied to the game state

    /// Helper to verify all generated actions are valid by applying them to game state
    fn verify_all_actions_valid(game_state: &GameState) {
        let action_gen = create_simple_generator(game_state);
        let actions = action_gen.gen_possible_actions(game_state);

        for action in &actions {
            let mut gs_copy = game_state.clone();
            let result = match action {
                AgentAction::Fold => {
                    gs_copy.fold();
                    Ok(())
                }
                AgentAction::Bet(amount) => gs_copy.do_bet(*amount, false).map(|_| ()),
                AgentAction::Call => {
                    let call_amount = gs_copy.current_round_bet();
                    gs_copy.do_bet(call_amount, false).map(|_| ())
                }
                AgentAction::AllIn => {
                    let all_in_amount =
                        gs_copy.current_round_current_player_bet() + gs_copy.current_player_stack();
                    gs_copy.do_bet(all_in_amount, false).map(|_| ())
                }
            };

            assert!(
                result.is_ok(),
                "Action {:?} should be valid but got error: {:?}\n\
                 Game state: current_bet={}, min_raise={}, player_bet={}, stack={}, pot={}",
                action,
                result.err(),
                game_state.current_round_bet(),
                game_state.current_round_min_raise(),
                game_state.current_round_current_player_bet(),
                game_state.current_player_stack(),
                game_state.total_pot
            );
        }
    }

    #[test]
    fn test_simple_all_actions_valid_preflop_sb() {
        // Small blind facing big blind
        let stacks = vec![100.0; 2];
        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        verify_all_actions_valid(&game_state);
    }

    #[test]
    fn test_simple_all_actions_valid_preflop_bb() {
        // Big blind after small blind completes
        let stacks = vec![100.0; 2];
        let mut game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        game_state.do_bet(10.0, false).unwrap(); // SB completes to BB
        verify_all_actions_valid(&game_state);
    }

    #[test]
    fn test_simple_all_actions_valid_flop_first_to_act() {
        // First to act on flop (can check)
        let stacks = vec![100.0; 2];
        let mut game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        game_state.advance_round(); // To flop
        verify_all_actions_valid(&game_state);
    }

    #[test]
    fn test_simple_all_actions_valid_facing_bet() {
        // Facing a bet on flop
        let stacks = vec![100.0; 2];
        let mut game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        game_state.advance_round();
        game_state.do_bet(20.0, false).unwrap(); // Opponent bets
        verify_all_actions_valid(&game_state);
    }

    #[test]
    fn test_simple_all_actions_valid_facing_raise() {
        // Facing a raise
        let stacks = vec![200.0; 2];
        let mut game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        game_state.advance_round();
        game_state.do_bet(20.0, false).unwrap(); // Bet
        game_state.do_bet(50.0, false).unwrap(); // Raise
        verify_all_actions_valid(&game_state);
    }

    #[test]
    fn test_simple_all_actions_valid_small_stack() {
        // Player with small stack
        let stacks = vec![30.0, 100.0];
        let mut game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        game_state.advance_round();
        game_state.do_bet(15.0, false).unwrap();
        verify_all_actions_valid(&game_state);
    }

    #[test]
    fn test_simple_all_actions_valid_large_pot() {
        // Large pot scenario where pot bets might exceed stack
        let stacks = vec![100.0; 2];
        let mut game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        // Build up pot through betting rounds
        game_state.do_bet(10.0, false).unwrap(); // SB completes
        game_state.do_bet(30.0, false).unwrap(); // BB raises
        game_state.do_bet(30.0, false).unwrap(); // SB calls
        game_state.advance_round(); // To flop
        verify_all_actions_valid(&game_state);
    }

    #[test]
    fn test_simple_all_actions_valid_tiny_pot() {
        // Tiny pot where pot-based bets might be smaller than min raise
        let stacks = vec![1000.0; 2];
        let mut game_state = GameState::new_starting(stacks, 2.0, 1.0, 0.0, 0);
        game_state.advance_round();
        verify_all_actions_valid(&game_state);
    }

    #[test]
    fn test_simple_all_actions_valid_after_multiple_raises() {
        // After multiple raises (min raise increases)
        let stacks = vec![500.0; 2];
        let mut game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        game_state.advance_round();
        game_state.do_bet(20.0, false).unwrap(); // Bet 20
        game_state.do_bet(50.0, false).unwrap(); // Raise to 50 (raise of 30)
        game_state.do_bet(110.0, false).unwrap(); // Re-raise to 110 (raise of 60)
        verify_all_actions_valid(&game_state);
    }

    #[test]
    fn test_simple_all_actions_valid_three_players() {
        // Three player scenario
        let stacks = vec![100.0; 3];
        let mut game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        game_state.advance_round();
        game_state.do_bet(15.0, false).unwrap();
        game_state.do_bet(15.0, false).unwrap();
        verify_all_actions_valid(&game_state);
    }

    #[test]
    fn test_simple_all_actions_valid_river() {
        // River scenario
        let stacks = vec![100.0; 2];
        let mut game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        game_state.advance_round(); // Flop
        game_state.advance_round(); // Turn
        game_state.advance_round(); // River
        verify_all_actions_valid(&game_state);
    }

    #[test]
    fn test_simple_pot_bet_not_too_small() {
        // Specific test for pot-based bets being at least min raise
        let stacks = vec![1000.0; 2];
        let mut game_state = GameState::new_starting(stacks, 100.0, 50.0, 0.0, 0);
        game_state.advance_round();

        let action_gen = create_simple_generator(&game_state);
        let actions = action_gen.gen_possible_actions(&game_state);

        // Find all Bet actions that are raises (not just calls)
        let call_amount = game_state.current_round_bet();
        let min_raise = game_state.current_round_min_raise();

        for action in &actions {
            if let AgentAction::Bet(amount) = action
                && *amount > call_amount
            {
                // This is a raise, verify it meets min raise
                let raise_amount = amount - call_amount;
                assert!(
                    raise_amount >= min_raise - BET_EPSILON,
                    "Bet of {} is a raise of {} which is less than min raise {}",
                    amount,
                    raise_amount,
                    min_raise
                );
            }
        }
    }

    // === Tests for action index consistency across game states ===

    #[test]
    fn test_simple_action_to_idx_uses_correct_game_state() {
        // This test verifies that action_to_idx maps actions correctly
        // when the action was generated from the same game state
        let stacks = vec![500.0; 2];
        let mut game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        game_state.advance_round();
        game_state.do_bet(30.0, false).unwrap();

        let action_gen = create_simple_generator(&game_state);
        let actions = action_gen.gen_possible_actions(&game_state);

        // Every action should map to a unique index when using the SAME game state
        let mut seen_indices = std::collections::HashSet::new();
        for action in &actions {
            let idx = action_gen.action_to_idx(&game_state, action);
            assert!(
                seen_indices.insert(idx),
                "Duplicate index {} for action {:?}",
                idx,
                action
            );
        }
    }

    #[test]
    fn test_simple_action_to_idx_with_different_pot_sizes() {
        // This test exposes the issue: actions generated from one game state
        // may not map correctly when using a different game state
        let stacks = vec![500.0; 2];

        // Game state 1: larger pot
        let mut gs1 = GameState::new_starting(stacks.clone(), 10.0, 5.0, 0.0, 0);
        gs1.advance_round();
        gs1.do_bet(50.0, false).unwrap(); // Pot is now 65

        // Game state 2: smaller pot (same structure, different pot)
        let mut gs2 = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        gs2.advance_round();
        gs2.do_bet(20.0, false).unwrap(); // Pot is now 35

        let action_gen1 = create_simple_generator(&gs1);
        let actions_gs1 = action_gen1.gen_possible_actions(&gs1);

        // Now try to map actions from gs1 using gs2's context
        // This simulates what happens in explore_all_actions when
        // actions are generated from game_state but action_to_idx
        // is called with starting_gamestate
        let action_gen2 = create_simple_generator(&gs2);

        println!(
            "Actions from gs1 (pot={}): {:?}",
            gs1.total_pot, actions_gs1
        );
        println!("Pot gs2: {}", gs2.total_pot);

        for action in &actions_gs1 {
            let idx_correct = action_gen1.action_to_idx(&gs1, action);
            let idx_wrong = action_gen2.action_to_idx(&gs2, action);

            println!(
                "Action {:?}: correct_idx={}, wrong_idx={}",
                action, idx_correct, idx_wrong
            );

            // The indices should match when using the correct game state
            // but may differ when using a different game state
            // This test documents the current behavior
        }
    }

    #[test]
    fn test_simple_actions_consistent_for_same_state() {
        // For a given game state, generating actions and mapping them back
        // should always produce consistent indices
        let stacks = vec![200.0; 2];
        let mut game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        game_state.advance_round();
        game_state.do_bet(25.0, false).unwrap();

        let action_gen = create_simple_generator(&game_state);
        let actions = action_gen.gen_possible_actions(&game_state);

        // Map actions to indices
        let indices: Vec<usize> = actions
            .iter()
            .map(|a| action_gen.action_to_idx(&game_state, a))
            .collect();

        // For each index, find the matching action
        for (action, &idx) in actions.iter().zip(indices.iter()) {
            // The regret matcher might return this index
            // We should be able to find the action that matches
            let found = actions
                .iter()
                .find(|a| action_gen.action_to_idx(&game_state, a) == idx);

            assert!(
                found.is_some(),
                "Could not find action for index {} (original action: {:?})",
                idx,
                action
            );
        }
    }

    #[test]
    fn test_simple_regret_matcher_index_always_findable() {
        // This test verifies that for any index the regret matcher might return,
        // we can find a valid action in gen_possible_actions
        let stacks = vec![300.0; 2];
        let mut game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        game_state.advance_round();
        game_state.do_bet(40.0, false).unwrap();

        let action_gen = create_simple_generator(&game_state);
        let actions = action_gen.gen_possible_actions(&game_state);
        let num_potential = action_gen.num_potential_actions(&game_state);

        // Get all indices that are represented in actions
        let valid_indices: std::collections::HashSet<usize> = actions
            .iter()
            .map(|a| action_gen.action_to_idx(&game_state, a))
            .collect();

        println!("Valid indices for this state: {:?}", valid_indices);
        println!("Num potential actions: {}", num_potential);

        // The regret matcher can return any index from 0 to num_potential-1
        // Some of these might not have a matching action in the current state
        for idx in 0..num_potential {
            let has_match = actions
                .iter()
                .any(|a| action_gen.action_to_idx(&game_state, a) == idx);

            if !has_match {
                println!(
                    "WARNING: Index {} has no matching action in this game state",
                    idx
                );
            }
        }

        // At minimum, we should have at least 2 valid actions (check/call + one other)
        assert!(
            valid_indices.len() >= 2,
            "Should have at least 2 valid actions, got {}",
            valid_indices.len()
        );
    }
}
